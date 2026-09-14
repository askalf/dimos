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

"""Classical apartment sensing, feasibility and physical outcome checks."""

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Protocol, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.core.core import rpc
from dimos.msgs.manipulation_msgs.GraspCandidateArray import GraspCandidateArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.robot.galaxea.r1pro.apartment_navigation import ApartmentSimSpec
from dimos.robot.galaxea.r1pro.apartment_route import apartment_approach, apartment_departure
from dimos.robot.galaxea.r1pro.apartment_sim import R1ProApartmentSim
from dimos.robot.galaxea.r1pro.classical_perception import segmented_object_cloud
from dimos.robot.galaxea.r1pro.classical_planning import ClassicalGraspPlanner
from dimos.robot.galaxea.r1pro.grasping_sim import VIRTUAL_BASE_JOINTS
from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport
from dimos.robot.galaxea.r1pro.home_surfaces import station_name
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.navigation_base import PlanarVelocityServo
from dimos.robot.galaxea.r1pro.navigation_sim import carrying_offset
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, Arm
from dimos.robot.galaxea.r1pro.primitive_scene import placement_options
from dimos.simulation.engines.mujoco_engine import MujocoEngine


class ClassicalSimSpec(ApartmentSimSpec, Protocol):
    def save_classical_state(self) -> str: ...
    def classical_carry_posture(self) -> list[list[float]]: ...
    def classical_align(
        self, index: int, arm: str, target: list[list[float]]
    ) -> list[list[float]]: ...
    def classical_posture(
        self, index: int, arm: str, positions: list[float], target: list[list[float]] | None = None
    ) -> list[list[float]]: ...
    def classical_object_cloud(self, index: int) -> PointCloud2: ...
    def assess_classical_pick(
        self, index: int, candidates: GraspCandidateArray, arm: str
    ) -> list[dict[str, Any]]: ...
    def classical_line(
        self, index: int, arm: str, target: list[list[float]]
    ) -> list[list[float]]: ...
    def classical_place_goals(self, arm: str, region: str) -> list[dict[str, Any]]: ...


class R1ProClassicalSim(R1ProApartmentSim):
    """Keep the physical apartment while replacing ACT with measured Cartesian plans."""

    def _reachable_base_path(
        self, planner: PlanarTransport, pose: NDArray[Any]
    ) -> list[list[float]]:
        yaw = np.arctan2(np.sin(pose[2] - planner.start[2]), np.cos(pose[2] - planner.start[2]))
        if abs(yaw) > 0.02:
            return planner.plan_stance(pose.tolist(), separate_turns=True)
        return super()._reachable_base_path(planner, pose)

    @rpc
    def save_classical_state(self) -> str:
        """Record exact physical state and runtime model parameters for replay, including failures."""
        if self._engine is None:
            raise RuntimeError("Simulation is not ready")
        output = self.config.output / f"classical-state-{time.time_ns()}.npz"
        with self._engine._lock:
            model_path = self.config.output / "classical-model.mjb"
            if not model_path.exists():
                mujoco.mj_saveModel(self._engine.model, str(model_path), None)
            scene = self._state(self._engine)
            np.savez(
                output,
                qpos=scene.data.qpos,
                qvel=scene.data.qvel,
                ctrl=scene.data.ctrl,
                time=scene.data.time,
                selected_left=scene.arms["left"].selected,
                selected_right=scene.arms["right"].selected,
            )
        return str(output)

    def _publish_shm_and_lcm(self, engine: MujocoEngine) -> None:
        with engine._lock:
            if self._servo is None:
                pose = np.array([engine.data.joint(name).qpos[0] for name in VIRTUAL_BASE_JOINTS])
                self._servo = PlanarVelocityServo(
                    pose, max_speed=0.3, max_accel=0.2, max_yaw_rate=0.4, max_yaw_accel=0.4
                )
        super()._publish_shm_and_lcm(engine)

    @rpc
    def primitive_state(self) -> dict[str, Any]:
        state = super().primitive_state()
        assert self._engine is not None
        with self._engine._lock:
            for row in state["objects"]:
                row["orientation_wxyz"] = self._engine.data.body(row["object"]).xquat.tolist()
            state["gripper_commands"] = {
                side: float(self._engine.data.actuator(f"r1pro/{side}_gripper").ctrl[0])
                for side in ARMS
            }
            state["joint_commands"] = {
                name: float(self._engine.data.actuator(name).ctrl[0])
                for name in R1PRO_PICK_PLACE_JOINTS
            }
            state["joint_velocities"] = {
                name: float(self._engine.data.joint(name).qvel[0])
                for name in R1PRO_PICK_PLACE_JOINTS
            }
            state["tcp_poses"] = {}
            for side in ARMS:
                site = self._engine.data.site(f"{side}_tcp")
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = site.xmat.reshape(3, 3), site.xpos
                state["tcp_poses"][side] = pose.tolist()
        return state

    def _snapshot(self) -> PrimitiveSceneState:
        if self._engine is None:
            raise RuntimeError("Simulation is not ready")
        with self._engine._lock:
            if self._error:
                raise RuntimeError(self._error)
            return self._state(self._engine).snapshot()

    @rpc
    def classical_object_cloud(self, index: int) -> PointCloud2:
        """Acquire current raycast depth with simulation instance labels."""
        scene = self._snapshot()
        if not 0 <= index < len(scene.layout.objects):
            raise ValueError("Unknown object index")
        return segmented_object_cloud(
            scene.model, scene.data, scene.model.body(scene.layout.objects[index].name).id
        )

    @rpc
    def assess_classical_pick(
        self, index: int, candidates: GraspCandidateArray, arm: str
    ) -> list[dict[str, Any]]:
        """Rank actual GraspGenX TCP poses by collision-checked DimOS reachability."""
        if candidates.header.frame_id != "world":
            raise ValueError("Grasp proposals must be expressed in world")
        scene = self._snapshot()
        poses, scores = [], []
        for candidate in candidates.candidates:
            matrix = np.eye(4)
            matrix[:3, :3] = candidate.pose.orientation.to_rotation_matrix()
            matrix[:3, 3] = candidate.pose.position.to_numpy()
            poses.append(matrix)
            scores.append(candidate.score)
        if not poses:
            return []
        np.savez(
            self.config.output / f"classical-grasps-{time.time_ns()}.npz",
            poses=poses,
            scores=scores,
            qpos=scene.data.qpos,
            qvel=scene.data.qvel,
            ctrl=scene.data.ctrl,
            index=index,
            arm=arm,
        )
        planner = ClassicalGraspPlanner(scene)
        return [
            asdict(plan)
            for plan in planner.rank(index, np.asarray(poses), np.asarray(scores), arm=arm)
        ]

    @rpc
    def classical_carry_posture(self) -> list[list[float]]:
        """Prepare compact hands for navigation and guard every held object during motion."""
        scene = self._snapshot()
        points = ClassicalGraspPlanner(scene).carry_posture()
        assert self._engine is not None
        with self._engine._lock:
            self._transport_initial = scene.inventory()
            self._active = None
            self._initial = None
        return points

    @rpc
    def primitive_recovery(self) -> dict[str, Any]:
        if self._active is not None or self._transport_initial is None:
            return super().primitive_recovery()
        assert self._engine is not None
        with self._engine._lock:
            scene = self._state(self._engine)
            scene.validate(self._transport_initial, arm="right", selected=-1)
            return dict(mode="navigation_hold", held_objects=scene.held_objects())

    @rpc
    def finish_primitive_recovery(self) -> dict[str, Any]:
        if self._active is not None or self._transport_initial is None:
            return super().finish_primitive_recovery()
        recovery = self.primitive_recovery()
        assert self._engine is not None
        with self._engine._lock:
            self._transport_initial = None
            self._error = None
        return recovery

    @rpc
    def classical_align(self, index: int, arm: str, target: list[list[float]]) -> list[list[float]]:
        if arm not in ARMS:
            raise ValueError("Choose left or right")
        return ClassicalGraspPlanner(self._snapshot()).align_pose(
            index, cast("Arm", arm), np.asarray(target, dtype=float)
        )

    @rpc
    def classical_posture(
        self, index: int, arm: str, positions: list[float], target: list[list[float]] | None = None
    ) -> list[list[float]]:
        if arm not in ARMS:
            raise ValueError("Choose left or right")
        planner = ClassicalGraspPlanner(self._snapshot())
        if target is not None:
            try:
                return planner.transfer_path(index, cast("Arm", arm), np.asarray(target))
            except RuntimeError as transfer_error:
                try:
                    return planner.posture_path(index, cast("Arm", arm), positions)
                except RuntimeError as posture_error:
                    raise RuntimeError(f"{transfer_error}; {posture_error}") from posture_error
        return planner.posture_path(index, cast("Arm", arm), positions)

    @rpc
    def classical_line(self, index: int, arm: str, target: list[list[float]]) -> list[list[float]]:
        """Replan the local Cartesian leg from fresh measured joints and contacts."""
        if arm not in ARMS:
            raise ValueError("Choose left or right")
        scene = self._snapshot()
        planner = ClassicalGraspPlanner(scene)
        try:
            planner.initialize_local_probe(index, cast("Arm", arm))
            return planner.segment(cast("Arm", arm), index, np.asarray(target, dtype=float))
        except RuntimeError:
            np.savez(
                self.config.output / f"classical-line-failure-{time.time_ns()}.npz",
                qpos=scene.data.qpos,
                qvel=scene.data.qvel,
                ctrl=scene.data.ctrl,
                target=target,
                arm=arm,
                index=index,
            )
            raise

    @rpc
    def classical_place_goals(self, arm: str, region: str) -> list[dict[str, Any]]:
        """Intersect empty support footprints with the holding hand's measured transform."""
        if arm not in ARMS:
            raise ValueError("Choose left or right")
        scene = self._snapshot()
        side = cast("Arm", arm)
        index = scene.held_objects()[side]
        if index is None:
            raise ValueError(f"The {arm} hand does not hold an object")
        state = scene.arms[side]
        points, surface = placement_options(
            state, self._regions.get(region, region), None, check_gripper=False
        )
        np.savez(
            self.config.output / f"classical-place-{side}-{time.time_ns()}.npz",
            qpos=scene.data.qpos,
            qvel=scene.data.qvel,
            ctrl=scene.data.ctrl,
        )
        planner = ClassicalGraspPlanner(scene)
        return [
            dict(plan, region=asdict(surface))
            for plan in planner.rank_places(index, side, [np.asarray(point) for point in points])
        ]

    @rpc
    def validate_primitive_base_plan(self, trajectory: JointTrajectory) -> None:
        """Check SDK local motion with held cargo during manipulation or departure."""
        if self._active is None and self._transport_initial is None:
            raise RuntimeError("No selected primitive or prepared departure")
        if set(trajectory.joint_names) != set(VIRTUAL_BASE_JOINTS) or not trajectory.points:
            raise ValueError("Prepositioning requires a nonempty base-only trajectory")
        scene = self._snapshot()
        if self._transport_initial is not None:
            scene.validate(self._transport_initial, arm="right", selected=-1)
        planner = scene.transport_planner()
        columns = [trajectory.joint_names.index(name) for name in VIRTUAL_BASE_JOINTS]
        start = planner.start
        for point in trajectory.points:
            target = np.asarray(point.positions)[columns]
            if not np.isfinite(target).all() or not planner.clear_pose_segment(start, target):
                raise RuntimeError("DimOS base plan is obstructed with the held objects")
            start = target

    @rpc
    def prepare_object_navigation(
        self, destination: str, arm: str = "right", stance: list[float] | None = None
    ) -> dict[str, Any]:
        """Plan a clear departure and a dock outside the requested support or object."""
        if arm not in ARMS or self._engine is None:
            raise ValueError("A live scene and valid arm are required")
        scene = self._snapshot()
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
        # Leave an extra 2 cm at the arrival stop for measured settling error,
        # so the next checked turn still has the full transit clearance.
        planner = scene.transport_planner(
            collision_margin=self.config.navigation_clearance_m + 0.02
        )
        local = scene.transport_planner()
        departure = apartment_departure(local, planner)
        # The approach turn begins from the departure heading, not the grasp's
        # diagonal heading. Rebase the planning copy including measured cargo.
        pose = np.asarray(departure[-1])
        planner.clear_pose_segment(pose, pose)
        aligned = PlanarTransport(
            scene.model,
            planner.probe,
            cargo_bodies=planner.cargo_bodies,
            carry_tray=False,
            sweep_spacing=planner.sweep_spacing,
            collision_margin=planner.collision_margin,
        )
        goal, docking = apartment_approach(aligned, goal)
        offset = carrying_offset(scene.model, scene.data, planner.robot_bodies)
        with self._engine._lock:
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
