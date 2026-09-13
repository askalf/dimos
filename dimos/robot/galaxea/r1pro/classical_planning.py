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

"""Collision-checked whole-body reachability for actual GraspGen TCP poses.

The same DimOS IK is used for candidate assessment and Cartesian execution.
All kinematic object attachments here live only in the planning snapshot.
"""

from dataclasses import dataclass, replace
import time
from typing import Any, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.manipulation.planning.groups.models import PlanningGroupSelection
from dimos.manipulation.planning.planners.rrt_planner import RRTConnectPlanner
from dimos.manipulation.planning.spec.protocols import WorldSpec
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, Arm, active_indices
from dimos.robot.galaxea.r1pro.object_reachability import ObjectReachability, stance_candidates


def preserves_cargo_tilt(initial_up_z: float, current_up_z: float) -> bool:
    """Keep a verified upright hold within 2 degrees of its initial tilt, capped at 15."""
    initial_tilt = float(np.arccos(np.clip(initial_up_z, -1, 1)))
    limit = min(np.deg2rad(15), max(np.deg2rad(5), initial_tilt + np.deg2rad(2)))
    return bool(current_up_z >= np.cos(limit))


@dataclass(frozen=True)
class ClassicalGraspPlan:
    arm: Arm
    index: int
    base_pose: list[float]
    source_position: list[float]
    tcp: list[list[float]]
    pregrasp: list[list[float]]
    ready_joints: list[float]
    grasp_score: float
    joint_margin: float
    manipulability: float
    cost: float
    refinement: str = "graspgenx"


class ClassicalGraspPlanner(ObjectReachability):
    allow_target_contact: bool = False
    sweep_error: str = ""

    def _joint_margin(self, arm: Arm, joints: NDArray[Any]) -> float:
        margins = []
        for column in active_indices(arm)[:-1]:
            lower, upper = self.model.joint(R1PRO_PICK_PLACE_JOINTS[column]).range
            margins.append(min(joints[column] - lower, upper - joints[column]) / (upper - lower))
        return float(min(margins))

    def _joint_clearance(self, arm: Arm, joints: NDArray[Any]) -> float:
        """Minimum distance to an arm stop, in radians rather than range percentage."""
        return min(
            float(min(joints[column] - limits[0], limits[1] - joints[column]))
            for column in active_indices(arm)[:-1]
            for limits in [self.model.joint(R1PRO_PICK_PLACE_JOINTS[column]).range]
        )

    def _body_poses(self, target: NDArray[Any], arm: Arm) -> list[NDArray[np.float64]]:
        """Try the current stance, then center the target in the arm's working area."""
        current = self.transport.start
        candidates = stance_candidates(target, current, arm)
        preferred = np.array([0.42, 0.32 if arm == "left" else -0.32])

        def cost(pose: NDArray[Any]) -> float:
            c, s = np.cos(pose[2]), np.sin(pose[2])
            local = np.array([[c, s], [-s, c]]) @ (target[:2] - pose[:2])
            return float(
                np.linalg.norm(local - preferred)
                + 0.15 * np.linalg.norm(pose[:2] - current[:2])
                + 0.03 * abs(pose[2] - current[2])
            )

        result = [current.copy()]
        for pose in sorted(candidates, key=cost):
            if any(
                np.linalg.norm(pose[:2] - old[:2]) < 0.02 and abs(pose[2] - old[2]) < 0.02
                for old in result
            ):
                continue
            result.append(pose)
        return result

    def _collisions(
        self, *, selected: int, arm: Arm, allow_selected_contact: bool = False
    ) -> list[str]:
        return super()._collisions(
            selected=selected,
            arm=arm,
            allow_selected_contact=allow_selected_contact or self.allow_target_contact,
        )

    def _sweep(self, goal: NDArray[Any], selected: int, arm: Arm) -> bool:
        """Preserve existing cargo tilt; do not reject an unchanged, verified hold."""
        start = self.probe.qpos[self.qids].copy()
        count = max(2, int(np.ceil(np.max(np.abs(goal - start)) / 0.025)))
        for t in np.linspace(0, 1, count + 1):
            self.probe.qpos[self.qids] = start + t * (goal - start)
            self._forward()
            collisions = self._collisions(selected=selected, arm=arm)
            if collisions:
                self.sweep_error = f"contacts {collisions}"
                return False
            for index in self.attachments:
                name = self.scene.layout.objects[index].name
                if not preserves_cargo_tilt(
                    self.initial.body(name).xmat[8], self.probe.body(name).xmat[8]
                ):
                    self.sweep_error = f"cargo tilt changed for {name}"
                    return False
        return True

    def solve_pose(
        self, arm: Arm, tcp: NDArray[Any], *, torso: bool = False, preserve_other: bool = False
    ) -> NDArray[np.float64]:
        base = self.probe.body("base_link")
        transform = self.reference_rotation @ base.xmat.reshape(3, 3).T
        targets: dict[str, NDArray[Any]] = {
            arm: self.reference_position + transform @ (tcp[:3, 3] - base.xpos)
        }
        orientations: dict[str, Quaternion] = {
            arm: Quaternion.from_rotation_matrix(transform @ tcp[:3, :3])
        }
        other: Arm = "right" if arm == "left" else "left"
        if torso and (preserve_other or any(row["held_by"] == other for row in self.rows)):
            site = self.probe.site(f"{other}_tcp")
            targets[other] = self.reference_position + transform @ (site.xpos - base.xpos)
            orientations[other] = Quaternion.from_rotation_matrix(
                transform @ site.xmat.reshape(3, 3)
            )
        result = self.kinematics.solve(
            self.probe,
            targets,
            orientations=orientations,
            allow_torso=torso,
            position_tolerance=0.0002,
            orientation_tolerance=0.008,
            max_attempts=1,
        )
        result[-2:] = self.probe.qpos[self.qids[-2:]]
        return result

    def segment(self, arm: Arm, index: int, target: NDArray[Any]) -> list[list[float]]:
        """Solve and sweep a fixed-orientation line from the measured TCP, including its start."""
        site = self.probe.site(f"{arm}_tcp")
        start = site.xpos.copy()
        if np.linalg.norm(site.xmat.reshape(3, 3) - target[:3, :3]) > 0.03:
            raise RuntimeError("Cartesian approach requires the staged grasp orientation")
        # Plan collisions at measured finger positions, but retain the actuator
        # closure targets in execution. Commanding the measured contact width
        # would remove the grip force as soon as a new arm segment starts.
        points = [self._arm_command(self.probe.qpos[self.qids], arm)]
        steps = max(2, int(np.ceil(np.linalg.norm(target[:3, 3] - start) / 0.004)))
        for t in np.linspace(0, 1, steps + 1)[1:]:
            pose = target.copy()
            pose[:3, 3] = start + t * (target[:3, 3] - start)
            before = self.probe.qpos[self.qids].copy()
            goal = self.solve_pose(arm, pose)
            if np.max(np.abs(goal - before)) > 0.20:
                raise RuntimeError("Cartesian sweep has discontinuous IK")
            if not self._sweep(goal, index, arm):
                raise RuntimeError(f"Cartesian sweep rejected: {self.sweep_error}")
            points.append(self._arm_command(goal, arm))
        return points

    def _arm_command(self, joints: NDArray[Any], arm: Arm) -> list[float]:
        """Keep static load compensation on moving joints and targets on stationary joints."""
        command = [
            float(self.scene.data.actuator(name).ctrl[0]) for name in R1PRO_PICK_PLACE_JOINTS
        ]
        for column in active_indices(arm)[:-1]:
            measured = float(self.scene.data.qpos[self.qids[column]])
            bias = command[column] - measured
            if abs(bias) > 0.03:
                raise RuntimeError(
                    "Arm has not settled enough to estimate static load compensation"
                )
            command[column] = float(joints[column]) + bias
        return command

    def align_pose(self, index: int, arm: Arm, target: NDArray[Any]) -> list[list[float]]:
        """Correct measured TCP error while retaining each joint's static load compensation."""
        self.initialize_local_probe(index, arm)
        before = self.probe.qpos[self.qids].copy()
        try:
            goal = self.solve_pose(arm, target)
        except RuntimeError:
            goal = self.solve_pose(arm, target, torso=True, preserve_other=True)
        if np.max(np.abs(goal - before)) > 0.20 or not self._sweep(goal, index, arm):
            raise RuntimeError("Staging correction exceeds the local collision-checked range")
        command = np.array(
            [float(self.scene.data.actuator(name).ctrl[0]) for name in R1PRO_PICK_PLACE_JOINTS]
        )
        corrected = command.copy()
        corrected[:18] += (goal - before)[:18]
        return [command.tolist(), corrected.tolist()]

    def _check_closure(self, index: int, arm: Arm) -> None:
        """Require opposed finger-pad contacts at one attainable jaw opening."""
        pads = self.scene.arms[arm].pad_ids
        body = self.model.body(self.scene.layout.objects[index].name).id
        driver = self.model.joint(f"r1pro/{arm}_gripper").qposadr[0]
        for opening in np.linspace(0.05, 0, 101):
            self.probe.qpos[driver] = opening
            self._forward()
            touching = set()
            for contact in self.probe.contact:
                if not -0.002 <= contact.dist <= 0.0005:
                    continue
                a, b = map(int, contact.geom)
                if a in pads and self.model.geom_bodyid[b] == body:
                    touching.add(a)
                if b in pads and self.model.geom_bodyid[a] == body:
                    touching.add(b)
            if touching == pads and not self._collisions(selected=index, arm=arm):
                return
        raise RuntimeError("Generated grasp cannot establish two-pad contact at a valid opening")

    def initialize_probe(self, pose: NDArray[Any]) -> None:
        self.allow_target_contact = False
        if not self.transport.clear_pose_segment(pose, pose):
            raise RuntimeError("Candidate body pose is obstructed")
        self.probe.qpos[:] = self.transport.probe.qpos
        mujoco.mj_forward(self.model, self.probe)
        self.attachments.clear()
        for i, row in enumerate(self.rows):
            if row["held_by"]:
                self._attach(i, row["held_by"])

    def initialize_local_probe(self, index: int, arm: Arm) -> None:
        """Preserve measured supported grasps during the first lift, without base motion."""
        self.probe.qpos[:] = self.initial.qpos
        mujoco.mj_forward(self.model, self.probe)
        self.attachments.clear()
        for i, row in enumerate(self.rows):
            if row["held_by"]:
                self._attach(i, row["held_by"])
            elif i == index and row["grasped"] and arm in row["contacting_arms"]:
                # Before lift the table still supports the object. Two-pad
                # contact, rather than unsupported ownership, authorizes this
                # planning attachment. Live physics must prove the actual lift.
                self._attach(i, arm)
        self.allow_target_contact = True
        if self._collisions(selected=index, arm=arm):
            raise RuntimeError("Measured local manipulation start is obstructed")

    def evaluate_grasp(
        self, index: int, arm: Arm, tcp: NDArray[Any], pose: NDArray[Any], confidence: float
    ) -> ClassicalGraspPlan:
        if self.rows[index]["held_by"] or any(row["held_by"] == arm for row in self.rows):
            raise ValueError("A pick requires a free hand and an unheld object")
        if np.linalg.norm(np.asarray(self.rows[index]["position"])[:2] - pose[:2]) > 0.85:
            raise RuntimeError("Target outside bounded arm reach search")
        pregrasp = tcp.copy()
        pregrasp[:3, 3] += 0.10 * tcp[:3, 2]
        failures = []
        for torso in (False, True):
            self.initialize_probe(pose)
            try:
                ready = self.solve_pose(arm, pregrasp, torso=torso)
                self.probe.qpos[self.qids] = ready
                self._forward()
                if self._collisions(selected=index, arm=arm):
                    raise RuntimeError("Pregrasp is obstructed")
                # Only finger pads may touch the selected object during approach.
                self.allow_target_contact = True
                self.segment(arm, index, tcp)
                self._check_closure(index, arm)
                columns = list(active_indices(arm))[:-1]
                joints = self.probe.qpos[self.qids[columns]]
                lower = np.array(
                    [
                        self.model.joint(self.kinematics.config.joint_names[i]).range[0]
                        for i in columns
                    ]
                )
                upper = np.array(
                    [
                        self.model.joint(self.kinematics.config.joint_names[i]).range[1]
                        for i in columns
                    ]
                )
                margin = float(np.min(np.minimum(joints - lower, upper - joints) / (upper - lower)))
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(  # type: ignore[attr-defined]
                    self.model, self.probe, jac, None, self.model.site(f"{arm}_tcp").id
                )
                dofs = [
                    self.model.joint(self.kinematics.config.joint_names[i]).dofadr[0]
                    for i in columns
                ]
                manipulability = float(np.prod(np.linalg.svd(jac[:, dofs], compute_uv=False)))
                self._attach(index, arm)
                lifted = tcp.copy()
                lifted[:3, 3] += [0, 0, 0.12]
                self.segment(arm, index, lifted)
                distance = float(np.linalg.norm(pose[:2] - self.transport.start[:2]))
                yaw = abs(float(pose[2] - self.transport.start[2]))
                cost = (
                    distance + 0.15 * yaw + 0.3 * (1 - confidence) - 0.5 * margin - manipulability
                )
                return ClassicalGraspPlan(
                    arm,
                    index,
                    pose.tolist(),
                    list(self.rows[index]["position"]),
                    tcp.tolist(),
                    pregrasp.tolist(),
                    ready.tolist(),
                    confidence,
                    margin,
                    manipulability,
                    cost,
                )
            except RuntimeError as exc:
                failures.append(str(exc))
        raise RuntimeError("; ".join(failures))

    def evaluate_place(
        self,
        index: int,
        arm: Arm,
        target: NDArray[Any],
        pose: NDArray[Any],
        *,
        yaw_offset: float = 0,
    ) -> dict[str, Any]:
        """Prove approach, support contact, finger opening and retreat at a candidate stance."""
        if self.rows[index]["held_by"] != arm:
            raise ValueError("Placement must use the holding hand")
        if np.linalg.norm(target[:2] - pose[:2]) > 0.85:
            raise RuntimeError("Placement outside bounded arm reach search")
        errors = []
        for torso in (False, True):
            self.initialize_probe(pose)
            site = self.probe.site(f"{arm}_tcp")
            c, s = np.cos(yaw_offset), np.sin(yaw_offset)
            turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            tcp = np.eye(4)
            tcp[:3, :3] = turn @ site.xmat.reshape(3, 3)
            tcp[:3, 3] = target + turn @ (
                site.xpos - self.probe.body(self.scene.layout.objects[index].name).xpos
            )
            above = tcp.copy()
            above[2, 3] = max(
                tcp[2, 3] + 0.10,
                target[2] + self.scene.layout.objects[index].half_size[2] + 0.05,
            )
            try:
                ready = self.solve_pose(arm, above, torso=torso)
                margin = self._joint_margin(arm, ready)
                clearance = self._joint_clearance(arm, ready)
                # The joint ranges differ substantially. Requiring 4% of each
                # range rejected poses with >10 degrees of available motion.
                # Keep at least 0.06 rad away from the physical stop (twice the
                # maximum accepted load-compensation bias) on every arm joint.
                if clearance < 0.06:
                    raise RuntimeError("Placement posture leaves insufficient joint-limit margin")
                self.probe.qpos[self.qids] = ready
                self._forward()
                self.allow_target_contact = True
                if self._collisions(selected=index, arm=arm):
                    raise RuntimeError("Preplace is obstructed")
                self.segment(arm, index, tcp)
                self.attachments.pop(index)
                self.probe.qpos[self.qids[18 if arm == "left" else 19]] = 0.05
                self._forward()
                if self._collisions(selected=index, arm=arm):
                    raise RuntimeError("The hand cannot open at this supported spot")
                self.segment(arm, index, above)
                cost = float(
                    np.linalg.norm(pose[:2] - self.transport.start[:2])
                    + 0.15 * abs(pose[2] - self.transport.start[2])
                    - 0.5 * margin
                    + 0.02 * abs(yaw_offset)
                )
                return dict(
                    index=index,
                    arm=arm,
                    target=target.tolist(),
                    base_pose=pose.tolist(),
                    tcp=tcp.tolist(),
                    preplace=above.tolist(),
                    ready_joints=ready.tolist(),
                    joint_margin=margin,
                    joint_clearance_rad=clearance,
                    yaw_offset=yaw_offset,
                    cost=cost,
                )
            except RuntimeError as exc:
                errors.append(str(exc))
        raise RuntimeError("; ".join(errors))

    def rank_places(
        self, index: int, arm: Arm, targets: list[NDArray[Any]], *, seconds: float = 30
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + seconds
        results: list[dict[str, Any]] = []
        # Prefer a nearby empty spot, but examine multiple support locations and
        # body poses before deciding the region is unreachable.
        ordered = sorted(
            targets,
            key=lambda p: float(np.linalg.norm(p[:2] - self.initial.site(f"{arm}_tcp").xpos[:2])),
        )
        for target in ordered[:24]:
            for pose in self._body_poses(target, arm)[:8]:
                # Placement may turn an upright item about gravity. A fixed
                # wrist heading can otherwise force a joint against its stop.
                for yaw in (0.0, -np.pi / 2, np.pi / 2, np.pi):
                    if time.monotonic() > deadline:
                        return sorted(results, key=lambda row: row["cost"])
                    try:
                        results.append(
                            self.evaluate_place(index, arm, target, pose, yaw_offset=yaw)
                        )
                    except (ValueError, RuntimeError):
                        continue
                    if len(results) >= 3:
                        return sorted(results, key=lambda row: row["cost"])
        return sorted(results, key=lambda row: row["cost"])

    def posture_path(
        self,
        index: int,
        arm: Arm,
        positions: list[float],
        *,
        allow_target_contact: bool = True,
    ) -> list[list[float]]:
        """Use the SDK RRT with actual apartment/cargo collisions during its search."""
        self.initialize_local_probe(index, arm)
        self.allow_target_contact = allow_target_contact
        kin = self.kinematics
        start = kin.seed(self.probe)
        mapping = dict(zip(R1PRO_PICK_PLACE_JOINTS, positions, strict=True))
        goal = JointState(name=start.name, position=[mapping[name] for name in start.name])
        selection = PlanningGroupSelection.from_groups(
            tuple(kin.groups[name] for name in ("torso", "left_arm", "right_arm"))
        )
        checked_world = _ApartmentCollisionWorld(self, index, arm)
        result = RRTConnectPlanner().plan_selected_joint_path(
            cast("WorldSpec", checked_world),
            selection,
            start,
            goal,
            timeout=20.0,
            max_iterations=5000,
        )
        if result.path is None:
            raise RuntimeError(
                f"DimOS apartment posture planning failed: {result.status}: {result.message}"
            )
        grippers = [
            float(self.scene.data.actuator(f"r1pro/{side}_gripper").ctrl[0])
            for side in ("left", "right")
        ]
        points = []
        for state in result.path:
            values = dict(zip(state.name, state.position, strict=True))
            points.append([values[name] for name in R1PRO_PICK_PLACE_JOINTS[:18]] + grippers)
        return points

    def carry_posture(self) -> list[list[float]]:
        """Retract hands for travel while preserving measured grasp orientations."""
        held = [
            (i, cast("Arm", row["held_by"])) for i, row in enumerate(self.rows) if row["held_by"]
        ]
        index: int
        arm: Arm
        if held:
            index, arm = held[0]
        else:
            index, arm = 0, "right"
        hands = list(ARMS)
        failures = []
        for forward, lateral in ((0.28, 0.28), (0.32, 0.30), (0.36, 0.32)):
            self.initialize_local_probe(index, arm)
            self.allow_target_contact = bool(held)
            base = self.probe.body("base_link")
            rotation, origin = base.xmat.reshape(3, 3).copy(), base.xpos.copy()
            try:
                for side in hands:
                    site = self.probe.site(f"{side}_tcp")
                    tcp = np.eye(4)
                    tcp[:3, :3] = site.xmat.reshape(3, 3)
                    tcp[:3, 3] = origin + rotation @ np.array(
                        [forward, lateral if side == "left" else -lateral, 0]
                    )
                    tcp[2, 3] = max(0.95, float(site.xpos[2]))
                    joints = self.solve_pose(side, tcp, torso=True, preserve_other=True)
                    self.probe.qpos[self.qids] = joints
                    self._forward()
                if self._collisions(selected=index, arm=arm):
                    raise RuntimeError("Carrying posture is obstructed")
                return self.posture_path(
                    index,
                    arm,
                    self.probe.qpos[self.qids].tolist(),
                    allow_target_contact=bool(held),
                )
            except RuntimeError as exc:
                failures.append(str(exc))
        raise RuntimeError("No clear compact carrying posture: " + "; ".join(failures))

    def rank(
        self,
        index: int,
        poses: NDArray[Any],
        scores: NDArray[Any],
        *,
        arm: str = "auto",
        seconds_per_arm: float = 25,
        max_results: int = 3,
    ) -> list[ClassicalGraspPlan]:
        if arm not in (*ARMS, "auto"):
            raise ValueError("Choose auto, left or right")
        hands = [
            side
            for side in ARMS
            if arm in ("auto", side) and not any(r["held_by"] == side for r in self.rows)
        ]
        results = []
        target = np.asarray(self.rows[index]["position"])
        for side in hands:
            deadline = time.monotonic() + seconds_per_arm
            found = 0
            current_rotation = self.initial.site(f"{side}_tcp").xmat.reshape(3, 3)
            proposals = []
            for matrix, score in zip(poses, scores, strict=True):
                for symmetry in (np.eye(3), np.diag([-1.0, -1.0, 1.0])):
                    tcp = matrix.copy()
                    tcp[:3, :3] = matrix[:3, :3] @ symmetry
                    angle = float(
                        np.arccos(
                            np.clip((np.trace(current_rotation.T @ tcp[:3, :3]) - 1) / 2, -1, 1)
                        )
                    )
                    # Retain the generated approach/orientation, and also test
                    # a central body grasp for these symmetric demo objects.
                    # This changes the executed TCP goal, never the live object.
                    centered = tcp.copy()
                    offset = target - centered[:3, 3]
                    centered[:3, 3] += sum(
                        centered[:3, column] * np.dot(offset, centered[:3, column])
                        for column in (1, 2)
                    )
                    for pose, refinement in ((centered, "body_centered"), (tcp, "graspgenx")):
                        depth = abs(float(np.dot(target - pose[:3, 3], pose[:3, 2])))
                        proposals.append(
                            (
                                1 - float(score) + 0.3 * angle + 2 * depth,
                                pose,
                                float(score),
                                refinement,
                            )
                        )
            proposals.sort(key=lambda item: item[0])
            diverse: list[tuple[float, NDArray[Any], float, str]] = []
            for proposal in proposals:
                matrix = proposal[1]
                if any(
                    np.linalg.norm(matrix[:3, :3] - existing[1][:3, :3]) < 0.3
                    and np.linalg.norm(matrix[:3, 3] - existing[1][:3, 3]) < 0.02
                    for existing in diverse
                ):
                    continue
                diverse.append(proposal)
                if len(diverse) >= 24:
                    break
            for base in self._body_poses(target, side):
                # Bound each body pose's allocation so a bad dock does not
                # exhaust the entire reachability search on hundreds of grasps.
                for _, tcp, score, refinement in diverse[:12]:
                    if time.monotonic() > deadline:
                        break
                    try:
                        result = self.evaluate_grasp(index, side, tcp, base, score)
                    except (RuntimeError, ValueError):
                        continue
                    results.append(replace(result, refinement=refinement))
                    found += 1
                    if found >= max_results:
                        break
                if time.monotonic() > deadline or found >= max_results:
                    break
        return sorted(results, key=lambda result: result.cost)


class _ApartmentCollisionWorld:
    """Delegate SDK world operations, adding frozen simulator collision checks.

    This adapter is local to one RRT request. It never mutates the shared SDK
    world or live simulator, and both robot self-collisions and scene contacts
    remain enabled while the planner searches (not only after it returns).
    """

    def __init__(self, planner: ClassicalGraspPlanner, index: int, arm: Arm) -> None:
        self.planner, self.index, self.arm = planner, index, arm

    def __getattr__(self, name: str) -> Any:
        return getattr(self.planner.kinematics.world, name)

    def check_config_collision_free(self, state: JointState) -> bool:
        planner = self.planner
        for name, value in zip(state.name, state.position, strict=True):
            planner.probe.joint(name).qpos[0] = value
        planner._forward()
        if planner._collisions(selected=self.index, arm=self.arm):
            return False
        return all(
            preserves_cargo_tilt(
                planner.initial.body(planner.scene.layout.objects[i].name).xmat[8],
                planner.probe.body(planner.scene.layout.objects[i].name).xmat[8],
            )
            for i in planner.attachments
        )
