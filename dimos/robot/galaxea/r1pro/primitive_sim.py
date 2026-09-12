# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native physics and measured outcomes for independent bimanual ACT primitives."""

from dataclasses import asdict
from pathlib import Path
import threading
import time
from typing import Any, cast
from uuid import uuid4

import mujoco
import numpy as np
from pydantic import Field, TypeAdapter
from reactivex.disposable import Disposable

from dimos.constants import RECORDINGS_DIR
from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.imitation.observation import VectorObservation
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.robot.galaxea.r1pro.grasping_sim import VIRTUAL_BASE_JOINTS
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.navigation_base import PlanarVelocityServo
from dimos.robot.galaxea.r1pro.navigation_sim import pose_message
from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout, sample_layout
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState
from dimos.robot.galaxea.r1pro.object_primitive_task import ObjectPrimitiveTask
from dimos.robot.galaxea.r1pro.object_primitives import (
    ARMS,
    PRIMITIVES,
    Arm,
    Primitive,
    active_indices,
    primitive_observation,
)
from dimos.robot.galaxea.r1pro.placement_regions import PlacementRegion
from dimos.robot.galaxea.r1pro.primitive_scene import (
    bilateral_layout,
    choose_placement,
    prepare_primitive_scene,
)
from dimos.simulation.engines.mujoco_engine import MujocoEngine
from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule, MujocoSimModuleConfig


class R1ProPrimitiveSimConfig(MujocoSimModuleConfig):
    seed: int = Field(default=340000, ge=0)
    generate_scene: bool = True
    occupied: int = Field(default=0, ge=0, le=3)
    bilateral_layout: bool = True
    output: Path = Field(default_factory=lambda: RECORDINGS_DIR / "r1pro-primitives" / uuid4().hex)


class R1ProPrimitiveSim(MujocoSimModule):
    config: R1ProPrimitiveSimConfig
    left_wrist: Out[Image]
    right_wrist: Out[Image]
    pick_left_goal: Out[VectorObservation]
    pick_right_goal: Out[VectorObservation]
    place_left_goal: Out[VectorObservation]
    place_right_goal: Out[VectorObservation]
    base_cmd_vel: In[Twist]
    base_odom: Out[PoseStamped]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._preparation_lock = threading.Lock()
        self._session: dict[str, Any] | None = None
        self._layout: ObjectLayout | None = None
        self._scene_state: PrimitiveSceneState | None = None
        self._active: tuple[Primitive, Arm] | None = None
        self._initial: list[dict[str, Any]] | None = None
        self._region: PlacementRegion | None = None
        self._regions: dict[str, PlacementRegion] = {}
        self._error: str | None = None
        self._servo: PlanarVelocityServo | None = None
        self._last_odom = float("-inf")
        self._base_commands = 0
        self._last_check = float("-inf")

    @rpc
    def prepare_primitive_session(self) -> dict[str, Any]:
        """Resolve the generated physical scene and calibrated reset posture once."""
        with self._preparation_lock:
            if self._session is not None:
                return dict(self._session)
            if self.config.generate_scene:
                layout = sample_layout(self.config.seed, occupied=self.config.occupied)
                if self.config.bilateral_layout:
                    layout = bilateral_layout(layout)
                output = self.config.output.expanduser().resolve()
                if (output / "scene.xml").exists():
                    raise FileExistsError(f"Choose a new session output: {output}")
                scene, layout = prepare_primitive_scene(output / "scene.xml", layout, "right")
                self.config.address = scene
            scene = Path(self.config.address).expanduser().resolve()
            self._layout = TypeAdapter(ObjectLayout).validate_json(
                scene.with_suffix(".objects.json").read_text()
            )
            with ObjectPrimitiveTask(scene, self._layout, arm="right", images=False) as task:
                self.config.reset_joint_positions = task.home.tolist()
                limits = [task.model.joint(n).range.tolist() for n in R1PRO_PICK_PLACE_JOINTS]
            self._session = dict(
                scene=str(scene), output=str(scene.parent), seed=self._layout.seed, limits=limits
            )
            return dict(self._session)

    @rpc
    def build(self) -> None:
        self.prepare_primitive_session()
        super().build()

    @rpc
    def start(self) -> None:
        self.register_disposable(Disposable(self.color_image.subscribe(self._publish_goal)))
        self.register_disposable(Disposable(self.base_cmd_vel.subscribe(self._on_twist)))
        super().start()

    def _state(self, engine: MujocoEngine) -> PrimitiveSceneState:
        if self._scene_state is None:
            assert self._layout is not None and self.config.reset_joint_positions is not None
            self._scene_state = PrimitiveSceneState(
                engine.model,
                engine.data,
                self._layout,
                np.asarray(self.config.reset_joint_positions),
            )
        return self._scene_state

    def _on_twist(self, twist: Twist) -> None:
        if self._engine is not None:
            with self._engine._lock:
                if self._servo is not None:
                    self._base_commands += 1
                    self._servo.command_twist(
                        np.array([twist.linear.x, twist.linear.y, twist.angular.z]),
                        time.monotonic(),
                    )

    def _publish_goal(self, frame: Image) -> None:
        engine = self._engine
        if engine is None:
            return
        with engine._lock:
            if self._active is None:
                return
            primitive, arm = self._active
            state = self._state(engine).arms[arm]
            values = primitive_observation(
                primitive, arm, engine.data.qpos[state.qids], state.goal()
            )["observation.environment_state"]
        port = getattr(self, f"{primitive}_{arm}_goal")
        port.publish(VectorObservation(ts=frame.ts, values=tuple(map(float, values))))

    def _publish_shm_and_lcm(self, engine: MujocoEngine) -> None:
        super()._publish_shm_and_lcm(engine)
        with engine._lock:
            state = self._state(engine)
            pose = np.array([engine.data.joint(n).qpos[0] for n in VIRTUAL_BASE_JOINTS])
            if self._servo is None:
                self._servo = PlanarVelocityServo(pose, max_speed=0.12, max_accel=0.12)
            targets = self._servo.step(pose, float(engine.model.opt.timestep), time.monotonic())
            for name, target in zip(VIRTUAL_BASE_JOINTS, targets, strict=True):
                engine.data.actuator(name).ctrl[0] = target
            now = time.monotonic()
            if now - self._last_odom >= 0.02:
                self._last_odom = now
                self.base_odom.publish(pose_message(pose.tolist()))
            if (
                self._active is not None
                and self._initial is not None
                and now - self._last_check >= 0.05
            ):
                self._last_check = now
                state.observe()
                arm = self._active[1]
                try:
                    state.validate(self._initial, arm=arm, selected=state.arms[arm].selected)
                except RuntimeError as exc:
                    self._error = str(exc)

    @rpc
    def stop_primitive_base(self) -> None:
        """Hold the current base actuator targets after cancelling SDK execution."""
        engine = self._engine
        if engine is not None:
            with engine._lock:
                if self._servo is not None:
                    self._servo.stop(
                        np.array([engine.data.joint(n).qpos[0] for n in VIRTUAL_BASE_JOINTS])
                    )

    @rpc
    def is_simulation_running(self) -> bool:
        engine = self._engine
        return bool(engine and engine._sim_thread and engine._sim_thread.is_alive())

    @rpc
    def define_placement_region(
        self, name: str, x: float, y: float, width: float, depth: float
    ) -> dict[str, Any]:
        """Register a measured rectangle on the physical worktable, without moving anything."""
        if name in ("table", "tray") or not name.strip():
            raise ValueError("Choose a new nonempty region name")
        if not np.isfinite([x, y, width, depth]).all() or min(width, depth) <= 0:
            raise ValueError("Region dimensions must be finite and positive")
        engine = self._engine
        if engine is None:
            raise RuntimeError("Simulation is not ready")
        with engine._lock:
            model, data = engine.model, engine.data
            gid = next(
                i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body("task_table").id
            )
            center = data.geom_xpos[gid]
            half = model.geom_size[gid]
            if np.any(
                np.abs(np.array([x, y]) - center[:2]) + np.array([width, depth]) / 2
                > half[:2] - 0.008
            ):
                raise ValueError("The whole region must lie inside the measured worktable")
            region = PlacementRegion(
                name,
                (x, y, float(center[2] + half[2])),
                (width / 2, depth / 2),
                (model.geom(gid).name,),
            )
            self._regions[name] = region
            return asdict(region)

    @rpc
    def prepare_primitive(
        self, primitive: str, arm: str, index: int = -1, region: str = "tray"
    ) -> dict[str, Any]:
        """Select a supported source or explicit empty region without commanding motion."""
        if primitive not in PRIMITIVES or arm not in ARMS:
            raise ValueError("Choose pick/place and left/right")
        engine = self._engine
        if engine is None:
            raise RuntimeError("Simulation is not ready")
        with engine._lock:
            if self._error is not None:
                raise RuntimeError(self._error)
            scene = self._state(engine)
            if primitive == "pick":
                state = scene.select_pick(cast("Arm", arm), index)
                target = engine.data.body(state.bottle_id).xpos.copy()
                self._region = None
            else:
                held = scene.held_objects()[cast("Arm", arm)]
                if held is None:
                    raise RuntimeError(f"The {arm} hand does not hold an object")
                state = scene.arms[cast("Arm", arm)]
                target, self._region = choose_placement(
                    state, self._regions.get(region, region), None
                )
                scene.select_place(cast("Arm", arm), target)
            self._active = (cast("Primitive", primitive), cast("Arm", arm))
            self._initial = scene.inventory()
            path = scene.preposition_path(cast("Arm", arm), target)
            return dict(
                object=f"object_{state.selected + 1}",
                arm=arm,
                primitive=primitive,
                target=target.tolist(),
                base_target=path[-1],
                base_waypoints=path,
                region=asdict(self._region) if self._region else None,
            )

    @rpc
    def validate_primitive_base_plan(self, trajectory: JointTrajectory) -> None:
        """Check the actual DimOS base trajectory with the current posture and held objects."""
        engine = self._engine
        if engine is None or self._active is None:
            raise RuntimeError("No selected primitive to preposition")
        if set(trajectory.joint_names) != set(VIRTUAL_BASE_JOINTS) or not trajectory.points:
            raise ValueError("Prepositioning requires a nonempty base-only trajectory")
        columns = [trajectory.joint_names.index(name) for name in VIRTUAL_BASE_JOINTS]
        with engine._lock:
            planner = self._state(engine).transport_planner()
            start = planner.start
            for point in trajectory.points:
                target = np.asarray(point.positions)[columns]
                if not np.isfinite(target).all() or not planner.clear_pose_segment(start, target):
                    raise RuntimeError("DimOS base plan is obstructed with the held objects")
                start = target

    @rpc
    def primitive_state(self) -> dict[str, Any]:
        """Read current physical ownership, completion and protected-object evidence."""
        engine = self._engine
        if engine is None:
            raise RuntimeError("Simulation is not ready")
        with engine._lock:
            scene = self._state(engine)
            scene.observe()
            rows = scene.inventory()
            base = engine.data.body("base_link")
            for i, row in enumerate(rows):
                relative = base.xmat.reshape(3, 3).T @ (np.asarray(row["position"]) - base.xpos)
                row.update(
                    index=i,
                    id=f"object_{i + 1}",
                    rgba=list(scene.layout.objects[i].rgba),
                    forward_m=float(relative[0]),
                    left_m=float(relative[1]),
                    distance_m=float(np.linalg.norm(relative[:2])),
                )
            complete = False
            if self._active is not None:
                primitive, arm = self._active
                state = scene.arms[arm]
                row = rows[state.selected]
                if primitive == "pick":
                    complete = state.holding()
                elif self._region is not None:
                    obj = scene.layout.objects[state.selected]
                    complete = bool(
                        row["upright"]
                        and row["released"]
                        and row["settled"]
                        and set(self._region.support_geoms).intersection(row["support_geoms"])
                        and self._region.contains(
                            tuple(row["position"]), obj.radius, obj.half_size[2]
                        )
                        and engine.data.site(f"{arm}_tcp").xpos[2]
                        > row["position"][2] + obj.half_size[2] + 0.035
                    )
            return dict(
                seed=scene.layout.seed,
                source="simulator_ground_truth",
                supported_arms=list(ARMS),
                objects=rows,
                held_objects={
                    arm: f"object_{i + 1}" if i is not None else None
                    for arm, i in scene.held_objects().items()
                },
                active=self._active,
                complete=complete and self._error is None,
                error=self._error,
                base_pose=[float(engine.data.joint(n).qpos[0]) for n in VIRTUAL_BASE_JOINTS],
                sim_time=float(engine.data.time),
                joint_positions={
                    name: float(engine.data.joint(name).qpos[0]) for name in R1PRO_PICK_PLACE_JOINTS
                },
                tcp_positions={arm: engine.data.site(f"{arm}_tcp").xpos.tolist() for arm in ARMS},
                base_command_count=self._base_commands,
                base_command=self._servo.command.tolist() if self._servo else None,
                base_target=self._servo.target.tolist() if self._servo else None,
                base_ctrl=[float(engine.data.actuator(n).ctrl[0]) for n in VIRTUAL_BASE_JOINTS],
                regions=["tray", "table", *self._regions],
                defined_regions={name: asdict(region) for name, region in self._regions.items()},
            )

    @rpc
    def primitive_recovery(self) -> dict[str, Any]:
        """Inspect recovery without motion; preserve a confirmed hold and protect the other hand."""
        engine = self._engine
        if engine is None or self._active is None:
            raise RuntimeError("No selected primitive to recover")
        with engine._lock:
            scene = self._state(engine)
            arm = self._active[1]
            state = scene.arms[arm]
            if self._initial is not None:
                scene.validate(self._initial, arm=arm, selected=state.selected)
            rows = scene.inventory()
            held = scene.held_objects()[arm]
            if held is not None:
                return dict(mode="hold", arm=arm, object=f"object_{held + 1}")
            if any(
                arm in row["contacting_arms"] and (not row["upright"] or not row["support_geoms"])
                for row in rows
            ):
                raise RuntimeError(
                    "Unsupported contact: preserve the grip; automatic release is not safe"
                )
            if not rows[state.selected]["upright"] or not rows[state.selected]["support_geoms"]:
                raise RuntimeError("Selected object has lost stable support; holding position")
            indices = list(active_indices(arm))
            return dict(
                mode="empty_hand",
                arm=arm,
                joints=[R1PRO_PICK_PLACE_JOINTS[i] for i in indices],
                home=state.home[indices].tolist(),
            )

    @rpc
    def validate_primitive_recovery_plan(self, trajectory: JointTrajectory) -> None:
        """Check the SDK's planned empty-arm sweep against the physical environment and objects."""
        engine = self._engine
        if engine is None or self._active is None:
            raise RuntimeError("No selected primitive to recover")
        with engine._lock:
            scene = self._state(engine)
            arm = self._active[1]
            expected = {R1PRO_PICK_PLACE_JOINTS[i] for i in active_indices(arm)[:-1]}
            if not set(trajectory.joint_names) <= expected:
                raise RuntimeError("Recovery plan may only move the selected arm")
            probe = mujoco.MjData(engine.model)
            probe.qpos[:] = engine.data.qpos
            qids = [engine.model.joint(n).qposadr[0] for n in trajectory.joint_names]
            start = probe.qpos[qids].copy()
            guard = scene.arms[arm].guard
            robot = guard.robot_bodies - guard.cargo_ids - {guard.tray_id}
            # The other hand's existing held contacts remain permitted.
            held_other = {
                engine.model.body(row["object"]).id
                for row in scene.inventory()
                if row["held_by"] not in (None, arm)
            }
            for point in trajectory.points:
                target = np.asarray(point.positions)
                for t in np.linspace(0, 1, max(2, int(np.max(np.abs(target - start)) / 0.01) + 1)):
                    probe.qpos[qids] = start + t * (target - start)
                    mujoco.mj_forward(engine.model, probe)
                    for contact in probe.contact:
                        if contact.dist > 0 or contact.pos[2] < 0.06:
                            continue
                        a, b = map(int, engine.model.geom_bodyid[contact.geom])
                        if (a in robot) != (b in robot):
                            other = b if a in robot else a
                            if other not in held_other:
                                raise RuntimeError(
                                    f"Recovery path contacts {engine.model.body(other).name}"
                                )
                start = target

    @rpc
    def finish_primitive_recovery(self) -> dict[str, Any]:
        """Clear a stopped action's error only after its measured protected state is sound."""
        recovery = self.primitive_recovery()
        engine = self._engine
        assert engine is not None
        with engine._lock:
            if recovery["mode"] == "empty_hand":
                arm = recovery["arm"]
                state = self._state(engine).arms[arm]
                indices = list(active_indices(arm))
                if (
                    np.max(np.abs(engine.data.qpos[state.qids][indices] - state.home[indices]))
                    > 0.015
                ):
                    raise RuntimeError("Recovery has not reached the calibrated arm posture")
            self._active = None
            self._initial = None
            self._region = None
            self._error = None
        return recovery

    @rpc
    def reset(self) -> bool:
        applied = super().reset()
        if applied and self._engine is not None:
            with self._engine._lock:
                self._scene_state = None
                self._active = None
                self._initial = None
                self._region = None
                self._error = None
            self.stop_primitive_base()
        return applied
