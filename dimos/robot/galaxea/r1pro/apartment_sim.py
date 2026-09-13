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

"""Apartment lifecycle and cargo-aware navigation checks for independent ACT skills."""

from dataclasses import asdict
import json
from pathlib import Path
import secrets
import time
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.core.core import rpc
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.robot.galaxea.r1pro.apartment_route import refine_apartment_route
from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport
from dimos.robot.galaxea.r1pro.home_surfaces import station_name
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.navigation_cloud import save_environment_cloud
from dimos.robot.galaxea.r1pro.navigation_sim import carrying_offset
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, Arm, Primitive
from dimos.robot.galaxea.r1pro.object_reachability import ObjectReachability
from dimos.robot.galaxea.r1pro.primitive_scene import placement_options
from dimos.robot.galaxea.r1pro.primitive_sim import R1ProPrimitiveSim, R1ProPrimitiveSimConfig
from dimos.robot.galaxea.r1pro.primitive_workspace import PrimitiveWorkspace
from dimos.simulation.engines.mujoco_engine import MujocoEngine


class R1ProApartmentSimConfig(R1ProPrimitiveSimConfig):
    seed: int = Field(default_factory=lambda: secrets.randbelow(2**31), ge=0)
    everyday_objects: bool = True
    randomize_locations: bool = True
    policy_neighbor_distance: float | None = 0.8
    workspace_file: Path | None = None
    scene_package: Path | None = DIMOS_PROJECT_ROOT / "dimos/data/scene_packages/hssd_102344115"


class R1ProApartmentSim(R1ProPrimitiveSim):
    config: R1ProApartmentSimConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._transport_initial: list[dict[str, Any]] | None = None
        self._last_transport_check = float("-inf")

    @rpc
    def prepare_primitive_session(self) -> dict[str, Any]:
        session = super().prepare_primitive_session()
        cloud = Path(session["output"]) / "navigation-cloud.npy"
        with self._preparation_lock:
            if not cloud.exists():
                save_environment_cloud(Path(session["scene"]), cloud)
        return {**session, "cloud": str(cloud)}

    def _publish_shm_and_lcm(self, engine: MujocoEngine) -> None:
        super()._publish_shm_and_lcm(engine)
        now = time.monotonic()
        if self._transport_initial is not None and now - self._last_transport_check >= 0.05:
            with engine._lock:
                self._last_transport_check = now
                try:
                    self._state(engine).validate(self._transport_initial, arm="right", selected=-1)
                except RuntimeError as exc:
                    self._error = str(exc)

    @rpc
    def assess_object_reachability(
        self, primitive: str, index: int, arm: str = "auto", region: str = "tray"
    ) -> dict[str, Any]:
        """Compare physical manipulation corridors and audited ACT coverage without motion."""
        if self._engine is None or primitive not in ("pick", "place") or arm not in (*ARMS, "auto"):
            raise ValueError("A live scene, pick/place and auto/left/right are required")
        with self._engine._lock:
            live = self._state(self._engine)
            scene = live.snapshot()
            regions = dict(self._regions)
        rows = scene.inventory()
        if not 0 <= index < len(rows):
            raise ValueError("Unknown object index")
        if primitive == "pick":
            if not (
                rows[index]["released"] and rows[index]["upright"] and rows[index]["support_geoms"]
            ):
                raise ValueError("Pick source must be supported, upright and unheld")
            targets = [np.asarray(rows[index]["position"])]
            placement = None
        else:
            owner = rows[index]["held_by"]
            if owner is None or arm not in ("auto", owner):
                raise ValueError("Placement must preserve the object's holding hand")
            state = scene.arms[owner]
            state.selected = index
            state.bottle_id = scene.model.body(scene.layout.objects[index].name).id
            # First check object fit. Reachability checks the actual open-hand
            # sweep at each proposed heading, rather than assuming today's yaw.
            points, placement = placement_options(
                state, regions.get(region, region), None, check_gripper=False
            )
            targets = [np.asarray(point) for point in points]
        workspaces = {}
        if self.config.workspace_file is not None:
            workspaces = {
                name: PrimitiveWorkspace(**value)
                for name, value in json.loads(self.config.workspace_file.read_text()).items()
            }
        candidates = ObjectReachability(scene, workspaces).candidates(
            cast("Primitive", primitive), index, targets, arm=arm
        )
        return dict(
            object=f"object_{index + 1}",
            primitive=primitive,
            requested_arm=arm,
            candidates=[asdict(candidate) for candidate in candidates],
            region=asdict(placement) if placement is not None else None,
            source="simulator_geometry_and_dimos_ik",
            note="Kinematic feasibility and training bounds do not establish ACT success; paths require execution-time validation.",
        )

    @rpc
    def validate_apartment_posture(self, trajectory: JointTrajectory, index: int, arm: str) -> None:
        """Validate a full SDK torso/arm posture plan before granting execution."""
        if self._engine is None or arm not in ARMS:
            raise ValueError("A live scene and selected arm are required")
        with self._engine._lock:
            live = self._state(self._engine)
            scene = live.snapshot()
        ObjectReachability(scene).validate_posture(trajectory, index, cast("Arm", arm))

    @rpc
    def prepare_reachable_primitive(
        self, primitive: str, arm: str, index: int, region: str, stance: dict[str, Any]
    ) -> dict[str, Any]:
        """Recheck the selected object/empty spot and route to its evaluated stance."""
        engine = self._engine
        if engine is None or arm not in ARMS or primitive not in ("pick", "place"):
            raise ValueError("A live scene, primitive and requested hand are required")
        target = np.asarray(stance["target"], dtype=float)
        pose = np.asarray(stance["base_pose"], dtype=float)
        if target.shape != (3,) or pose.shape != (3,) or not np.isfinite([target, pose]).all():
            raise ValueError("Reachability target and base pose must be finite XYZ/XY-yaw")
        if stance["arm"] != arm:
            raise ValueError("Reachability cannot substitute the requested hand")
        with engine._lock:
            if self._error:
                raise RuntimeError(self._error)
            scene = self._state(engine).snapshot()
            registered_region = self._regions.get(region, region)
        # Search on a snapshot: a rejected request must not overwrite the live
        # action selection, and route planning must not stall the physics loop.
        if not 0 <= index < len(scene.layout.objects):
            raise ValueError("Unknown object index")
        initial = scene.inventory()
        if primitive == "pick":
            state = scene.select_pick(cast("Arm", arm), index)
            if np.linalg.norm(scene.data.body(state.bottle_id).xpos - target) > 0.005:
                raise RuntimeError("The source moved after reachability assessment")
            placement = None
        else:
            state = scene.select_place(cast("Arm", arm), target)
            if state.selected != index:
                raise RuntimeError("The holding hand changed after reachability assessment")
            points, placement = placement_options(
                state, registered_region, None, check_gripper=False
            )
            if not any(np.linalg.norm(np.asarray(point) - target) < 0.005 for point in points):
                raise RuntimeError("The selected placement spot is no longer empty and supported")
        planner = scene.transport_planner()
        path = self._reachable_base_path(planner, pose)
        with engine._lock:
            if self._error:
                raise RuntimeError(self._error)
            live = self._state(engine)
            live.validate(initial, arm=cast("Arm", arm), selected=-1)
            qids = [engine.model.joint(n).qposadr[0] for n in R1PRO_PICK_PLACE_JOINTS]
            if (
                np.max(np.abs(engine.data.qpos[qids] - scene.data.qpos[qids])) > 0.005
                or np.linalg.norm(
                    engine.data.body("base_link").xpos - scene.data.body("base_link").xpos
                )
                > 0.005
                or np.max(
                    np.abs(engine.data.body("base_link").xmat - scene.data.body("base_link").xmat)
                )
                > 0.005
            ):
                raise RuntimeError(
                    "Robot moved during reachability preparation; reassess before execution"
                )
            if primitive == "pick":
                live.select_pick(cast("Arm", arm), index)
            else:
                live.select_place(cast("Arm", arm), target)
            self._active = cast("Primitive", primitive), cast("Arm", arm)
            self._initial = live.inventory()
            self._region = placement
        return dict(
            object=f"object_{index + 1}",
            arm=arm,
            primitive=primitive,
            target=target.tolist(),
            base_target=path[-1],
            base_waypoints=path,
            region=asdict(placement) if placement else None,
            reachability=stance,
        )

    def _reachable_base_path(
        self, planner: PlanarTransport, pose: NDArray[Any]
    ) -> list[list[float]]:
        if (
            abs(np.arctan2(np.sin(pose[2] - planner.start[2]), np.cos(pose[2] - planner.start[2])))
            > 0.02
        ):
            raise RuntimeError("Navigate to the assessed approach heading before local positioning")
        path = planner.plan(tuple(pose[:2]), resolution=0.025, max_distance=1.5, timeout=10)
        return planner.shorten_path([*path, pose.tolist()])

    @rpc
    def prepare_object_navigation(
        self, destination: str, arm: str = "right", stance: list[float] | None = None
    ) -> dict[str, Any]:
        """Plan a clear departure and a dock outside the requested support or object."""
        if arm not in ARMS or self._engine is None:
            raise ValueError("A live scene and valid arm are required")
        with self._engine._lock:
            if self._error:
                raise RuntimeError(self._error)
            scene = self._state(self._engine)
            name = station_name(destination)
            if name.startswith("object_"):
                index = int(name.removeprefix("object_")) - 1
                rows = scene.inventory()
                if not 0 <= index < len(rows) or rows[index]["held_by"] is not None:
                    raise ValueError("Navigation source must be an unheld object")
                target = np.asarray(rows[index]["position"])
                region_name = next(
                    (
                        n
                        for n, r in self._regions.items()
                        if set(r.support_geoms) & set(rows[index]["support_geoms"])
                    ),
                    None,
                )
                if region_name is None:
                    raise RuntimeError("Object has no registered apartment support")
            else:
                region_name = "worktable" if name in ("worktop", "table") else name
                if region_name not in self._regions:
                    raise ValueError("Use a measured apartment region or an object ID")
                target = np.asarray(self._regions[region_name].center)
            yaw = {"worktable": 0.0, "dining_table": -np.pi / 2, "kitchen": np.pi}[region_name]
            goal = scene.preposition_pose(cast("Arm", arm), target, yaw=float(yaw))
            if stance is not None:
                goal = np.asarray(stance, dtype=float)
                if goal.shape != (3,) or not np.isfinite(goal).all():
                    raise ValueError("Expected a finite assessed base pose")
                yaw = float(goal[2])
            # Transit stops clear of the furniture; local SDK prepositioning docks later.
            goal[:2] -= 0.28 * np.array([np.cos(yaw), np.sin(yaw)])
            planner = scene.transport_planner()
            if not planner.clear_pose_segment(goal, goal):
                raise RuntimeError("Apartment approach pose is obstructed with current cargo")
            docking = goal.copy()
            # Holonomic travel can keep the narrower fore/aft profile across a
            # passage. Face the support after reaching its open approach area.
            goal[2] = planner.start[2]
            if not planner.clear_pose_segment(goal, docking):
                raise RuntimeError("No clear arrival turn with the current hands and cargo")
            yaw = float(goal[2])
            offset = carrying_offset(self._engine.model, self._engine.data, planner.robot_bodies)
            departure = None
            if (
                abs(np.arctan2(np.sin(yaw - planner.start[2]), np.cos(yaw - planner.start[2])))
                < 0.01
            ):
                departure = [planner.start.tolist()]
            else:
                rotation = np.array(
                    [
                        [np.cos(planner.start[2]), -np.sin(planner.start[2])],
                        [np.sin(planner.start[2]), np.cos(planner.start[2])],
                    ]
                )
                for retreat in (0.2, 0.3, 0.4, 0.1):
                    backed = planner.start.copy()
                    backed[:2] -= rotation @ np.array([retreat, 0])
                    turned = np.r_[
                        backed[:2],
                        planner.start[2]
                        + np.arctan2(
                            np.sin(yaw - planner.start[2]), np.cos(yaw - planner.start[2])
                        ),
                    ]
                    if planner.clear_pose_segment(
                        planner.start, backed
                    ) and planner.clear_pose_segment(backed, turned):
                        departure = [planner.start.tolist(), backed.tolist(), turned.tolist()]
                        break
            if departure is None:
                raise RuntimeError("No clear departure turn with the current hands and cargo")
            self._transport_initial = scene.inventory()
            self._active = None
            self._initial = None
            return dict(
                destination=region_name,
                goal=goal.tolist(),
                departure=departure,
                arrival=[goal.tolist(), docking.tolist()],
                footprint_offset=offset.tolist(),
                cloud=str(Path(self.config.output) / "navigation-cloud.npy"),
            )

    @rpc
    def validate_object_navigation(self, path: list[list[float]]) -> list[list[float]]:
        """Reject incomplete or colliding base paths with both hands' measured cargo."""
        poses = np.asarray(path, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2 or not np.isfinite(poses).all():
            raise ValueError("Expected at least two finite planar poses")
        if self._engine is None or self._transport_initial is None:
            raise RuntimeError("No prepared apartment transport")
        with self._engine._lock:
            scene = self._state(self._engine)
            scene.validate(self._transport_initial, arm="right", selected=-1)
            planner = scene.transport_planner()
            # Native graph nodes describe the footprint centre, not the entire
            # carried geometry. Retain only shortcuts whose complete sweeps are
            # clear; reject if no such connection reaches the requested end.
        return refine_apartment_route(planner, path)

    @rpc
    def finish_object_navigation(self) -> None:
        """Confirm cargo ownership before ending a transport monitoring session."""
        if self._engine is not None:
            with self._engine._lock:
                if self._transport_initial is not None:
                    self._state(self._engine).validate(
                        self._transport_initial, arm="right", selected=-1
                    )
                self._transport_initial = None

    @rpc
    def primitive_recovery(self) -> dict[str, Any]:
        """Recover a stopped route in place when every object's protected state is intact."""
        if self._transport_initial is None:
            return super().primitive_recovery()
        if self._engine is None:
            raise RuntimeError("No live apartment to inspect")
        with self._engine._lock:
            scene = self._state(self._engine)
            scene.validate(self._transport_initial, arm="right", selected=-1)
            return dict(mode="transport_hold", held_objects=scene.held_objects())

    @rpc
    def finish_primitive_recovery(self) -> dict[str, Any]:
        """Clear a transport error only after checking cargo, contacts and a stopped base."""
        if self._transport_initial is None:
            return super().finish_primitive_recovery()
        if self._engine is None:
            raise RuntimeError("No live apartment to inspect")
        with self._engine._lock:
            state = self._state(self._engine)
            state.validate(self._transport_initial, arm="right", selected=-1)
            speed = self._engine.data.body("base_link").cvel
            if np.linalg.norm(speed[:3]) > 0.02 or np.linalg.norm(speed[3:]) > 0.01:
                raise RuntimeError("Wait for the base to stop before completing transport recovery")
            result = dict(mode="transport_hold", held_objects=state.held_objects())
            self._transport_initial = None
            self._active = None
            self._initial = None
            self._region = None
            self._error = None
            return result

    @rpc
    def reset(self) -> bool:
        self._transport_initial = None
        return super().reset()
