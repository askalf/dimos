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

"""Measured arm reach and whole-body staging for local ACT manipulation.

A reachable end point is only one prerequisite. Candidates also require a clear
base pose, an approach and lift/retreat with a fixed torso, and room for the other
hand's cargo. These are kinematic checks, not a prediction of ACT success.
"""

from dataclasses import dataclass
from functools import partial
import math
import time
from typing import Any, Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.robot.galaxea.r1pro.grasping_sim import VIRTUAL_BASE_JOINTS
from dimos.robot.galaxea.r1pro.home_kinematics import HomeKinematics
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, Arm, Primitive, active_indices
from dimos.robot.galaxea.r1pro.primitive_workspace import PrimitiveWorkspace


@dataclass(frozen=True)
class ReachableStance:
    arm: Arm
    base_pose: tuple[float, float, float]
    target: tuple[float, float, float]
    ready_joints: tuple[float, ...]
    clearance_z: float
    score: float
    torso_changed: bool
    policy_coverage: Literal["covered", "outside", "unknown"]


def _outside_demonstrated_xy(
    pose: NDArray[Any], *, target: NDArray[Any], workspace: PrimitiveWorkspace
) -> bool:
    c, s = np.cos(pose[2]), np.sin(pose[2])
    xy = np.array([[c, s], [-s, c]]) @ (target[:2] - pose[:2])
    return bool(
        np.any(xy < np.asarray(workspace.target_min)[:2] - 0.005)
        or np.any(xy > np.asarray(workspace.target_max)[:2] + 0.005)
    )


def stance_candidates(
    target: NDArray[Any], current: NDArray[Any], arm: Arm
) -> list[NDArray[np.float64]]:
    """Seed a bounded IK search around the object; geometry decides the usable side."""
    if target.shape != (3,) or current.shape != (3,) or not np.isfinite([target, current]).all():
        raise ValueError("Expected finite target XYZ and current base XY/yaw")
    if arm not in ARMS:
        raise ValueError("Use left or right")
    side = -1 if arm == "right" else 1
    candidates = [current.copy()]
    for yaw in (current[2], *np.arange(8) * np.pi / 4):
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        for forward, lateral in (
            (0.42, 0.32),
            (0.50, 0.32),
            (0.36, 0.28),
            (0.60, 0.32),
            (0.70, 0.32),
            (0.45, 0.45),
            (0.55, 0.45),
            (0.55, 0.55),
            (0.60, 0.50),
        ):
            xy = target[:2] - rotation @ np.array([forward, side * lateral])
            angle = current[2] + math.atan2(math.sin(yaw - current[2]), math.cos(yaw - current[2]))
            pose = np.array([*xy, angle])
            if not any(np.linalg.norm(pose - p) < 1e-6 for p in candidates):
                candidates.append(pose)
    return sorted(
        candidates,
        key=lambda p: float(np.linalg.norm(p[:2] - current[:2]) + 0.15 * abs(p[2] - current[2])),
    )


class ObjectReachability:
    """Use DimOS IK and copied MuJoCo contacts; never move the live scene."""

    def __init__(
        self, scene: PrimitiveSceneState, workspaces: dict[str, PrimitiveWorkspace] | None = None
    ) -> None:
        self.scene = scene
        self.workspaces = workspaces or {}
        self.model = scene.model
        self.initial = mujoco.MjData(self.model)
        self.initial.qpos[:] = scene.data.qpos
        mujoco.mj_forward(self.model, self.initial)
        self.probe = mujoco.MjData(self.model)
        self.probe.qpos[:] = self.initial.qpos
        mujoco.mj_forward(self.model, self.probe)
        self.rows = scene.inventory()
        self.base_ids = np.array([self.model.joint(n).qposadr[0] for n in VIRTUAL_BASE_JOINTS])
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in R1PRO_PICK_PLACE_JOINTS])
        self.grippers = [
            (
                int(self.model.joint(f"r1pro/{arm}_gripper").qposadr[0]),
                int(self.model.joint(f"{arm}_gripper_follower").qposadr[0]),
            )
            for arm in ARMS
        ]
        self.finger_offsets = [
            float(self.initial.qpos[follower] - self.initial.qpos[driver])
            for driver, follower in self.grippers
        ]
        self.kinematics = HomeKinematics(self.model, self.initial)
        self.transport = scene.transport_planner()
        self.robot = (
            self.transport.robot_bodies - self.transport.cargo_ids - {self.transport.tray_id}
        )
        self.reference_position = self.initial.body("base_link").xpos.copy()
        self.reference_rotation = self.initial.body("base_link").xmat.reshape(3, 3).copy()
        self.attachments: dict[int, tuple[Arm, NDArray[Any], NDArray[Any]]] = {}

    def _attach(self, index: int, arm: Arm) -> None:
        body = self.probe.body(self.scene.layout.objects[index].name)
        tcp = self.probe.site(f"{arm}_tcp")
        rotation = tcp.xmat.reshape(3, 3)
        self.attachments[index] = (
            arm,
            rotation.T @ (body.xpos - tcp.xpos),
            rotation.T @ body.xmat.reshape(3, 3),
        )

    def _forward(self) -> None:
        """Move attached cargo only in the planning copy, using its measured grip transform."""
        # mj_forward computes contacts without solving joint equalities. Carry
        # the coupled finger with its driver in this kinematic copy, preserving
        # the measured compliance offset while held and opening both on release.
        for (driver, follower), offset in zip(self.grippers, self.finger_offsets, strict=True):
            self.probe.qpos[follower] = np.clip(self.probe.qpos[driver] + offset, 0.0, 0.05)
        mujoco.mj_forward(self.model, self.probe)
        for index, (arm, position, orientation) in self.attachments.items():
            tcp = self.probe.site(f"{arm}_tcp")
            rotation = tcp.xmat.reshape(3, 3)
            qpos = self.probe.joint(self.scene.layout.objects[index].joint).qpos
            qpos[:3] = tcp.xpos + rotation @ position
            quaternion = Quaternion.from_rotation_matrix(rotation @ orientation)
            qpos[3:] = [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
        if self.attachments:
            mujoco.mj_forward(self.model, self.probe)

    def _solve(self, targets: dict[str, NDArray[Any]], *, torso: bool) -> NDArray[np.float64]:
        """Transform the query into the fixed SDK model frame, keeping measured joints."""
        base = self.probe.body("base_link")
        transform = self.reference_rotation @ base.xmat.reshape(3, 3).T
        return self.kinematics.solve(
            self.probe,
            {
                arm: self.reference_position + transform @ (xyz - base.xpos)
                for arm, xyz in targets.items()
            },
            allow_torso=torso,
            position_tolerance=0.002,
            orientation_tolerance=0.003,
            max_attempts=1,
        )

    def _collisions(
        self, *, selected: int, arm: Arm, allow_selected_contact: bool = False
    ) -> list[str]:
        obj = self.model.body(self.scene.layout.objects[selected].name).id
        held = {
            self.model.body(self.scene.layout.objects[i].name).id: attachment[0]
            for i, attachment in self.attachments.items()
        }
        if allow_selected_contact:
            held[obj] = arm
        obstacles = set()
        if not self.kinematics.world.check_config_collision_free(self.kinematics.seed(self.probe)):
            obstacles.add("robot_self_collision")
        for contact in self.probe.contact:
            if contact.dist > 0 or contact.pos[2] < 0.06:
                continue
            a, b = map(int, self.model.geom_bodyid[contact.geom])
            if (a in self.robot) != (b in self.robot):
                other = b if a in self.robot else a
                robot_geom = int(contact.geom[0] if a in self.robot else contact.geom[1])
                owner = held.get(other)
                if owner and robot_geom in self.scene.arms[owner].pad_ids:
                    continue
                obstacles.add(self.model.body(other).name or f"body:{other}")
            elif (a in held) != (b in held):
                other = b if a in held else a
                # Permit contact on a horizontal support at the corridor's
                # lower end, but never penetration or brushing a side obstacle.
                normal = contact.frame[:3] * (1 if b in held else -1)
                if contact.dist >= -0.002 and normal[2] > 0.7:
                    continue
                obstacles.add(self.model.body(other).name or f"body:{other}")
        return sorted(obstacles)

    def _sweep(self, goal: NDArray[Any], selected: int, arm: Arm) -> bool:
        start = self.probe.qpos[self.qids].copy()
        count = max(2, int(np.ceil(np.max(np.abs(goal - start)) / 0.025)))
        for t in np.linspace(0, 1, count + 1):
            self.probe.qpos[self.qids] = start + t * (goal - start)
            self._forward()
            if self._collisions(selected=selected, arm=arm):
                return False
            if any(
                self.probe.body(self.scene.layout.objects[i].name).xmat[8] < np.cos(np.deg2rad(5))
                for i in self.attachments
            ):
                return False
        return True

    def validate_posture(self, trajectory: JointTrajectory, index: int, arm: Arm) -> None:
        """Check the actual SDK posture trajectory, preserving grip and cargo orientation."""
        names = list(trajectory.joint_names)
        if (
            not trajectory.points
            or not names
            or not set(names) <= set(R1PRO_PICK_PLACE_JOINTS[:18])
            or len(names) != len(set(names))
        ):
            raise ValueError("Posture may move only named torso/arm joints; grippers stay held")
        columns = [R1PRO_PICK_PLACE_JOINTS.index(name) for name in names]
        self.probe.qpos[:] = self.initial.qpos
        self.attachments.clear()
        mujoco.mj_forward(self.model, self.probe)
        for i, row in enumerate(self.rows):
            if row["held_by"]:
                self._attach(i, row["held_by"])
        for point in trajectory.points:
            positions = np.asarray(point.positions)
            if positions.shape != (len(names),) or not np.isfinite(positions).all():
                raise ValueError("Invalid posture trajectory positions")
            goal = self.probe.qpos[self.qids].copy()
            goal[columns] = positions
            if not self._sweep(goal, index, arm):
                raise RuntimeError("Posture trajectory contacts the scene or tips held cargo")

    def _cartesian_corridor(self, target: NDArray[Any], selected: int, arm: Arm) -> bool:
        """Solve the vertical path continuously, rather than just its end points."""
        start = self.probe.site(f"{arm}_tcp").xpos.copy()
        steps = max(1, int(np.ceil(np.linalg.norm(target - start) / 0.005)))
        for t in np.linspace(0, 1, steps + 1)[1:]:
            previous = self.probe.qpos[self.qids].copy()
            goal = self._solve({arm: start + t * (target - start)}, torso=False)
            goal[-2:] = previous[-2:]
            if np.max(np.abs(goal[:-2] - previous[:-2])) > 0.25 or not self._sweep(
                goal, selected, arm
            ):
                return False
        return True

    def evaluate(
        self, primitive: Primitive, arm: Arm, index: int, target: NDArray[Any], pose: NDArray[Any]
    ) -> ReachableStance:
        """Check a stance and an arm-only vertical manipulation corridor from it."""
        if primitive not in ("pick", "place") or arm not in ARMS or not 0 <= index < len(self.rows):
            raise ValueError("Unknown primitive, hand or object")
        row = self.rows[index]
        if primitive == "pick" and (row["held_by"] or any(r["held_by"] == arm for r in self.rows)):
            raise ValueError("Pick requires a free hand and an unheld object")
        if primitive == "place" and row["held_by"] != arm:
            raise ValueError("Placement must use the actual holding hand")
        if np.linalg.norm(target[:2] - pose[:2]) > 0.8:
            raise RuntimeError("Target is outside the arm search bound")
        workspace = self.workspaces.get(f"{primitive}-{arm}")
        c, s = np.cos(pose[2]), np.sin(pose[2])
        relative_target = np.r_[np.array([[c, s], [-s, c]]) @ (target[:2] - pose[:2]), target[2]]
        if not self.transport.clear_pose_segment(pose, pose):
            raise RuntimeError("The robot or carried objects obstruct this stance")
        self.probe.qpos[:] = self.transport.probe.qpos
        mujoco.mj_forward(self.model, self.probe)
        self.attachments.clear()
        for i, before in enumerate(self.rows):
            if before["held_by"]:
                self._attach(i, before["held_by"])
        obj = self.scene.layout.objects[index]
        offset = min(0.03, obj.half_size[2] * 0.45)
        tcp_offset = np.array([0, 0, offset])
        if primitive == "place":
            tcp_offset = (
                self.probe.body("base_link").xmat.reshape(3, 3)
                @ self.reference_rotation.T
                @ np.asarray(row["tcp_offset"])
            )
            offset = float(tcp_offset[2])
        grasp = target + tcp_offset
        # A pick needs a ten-centimetre physical lift plus tracking margin. A
        # placement needs a clear transfer and open-hand retreat, not another
        # full lift above the destination. Requiring that extra height can push
        # an otherwise valid release past a wrist limit. Nearby objects still
        # raise either corridor to clear their complete geometry below.
        clearance = max(0.94, grasp[2] + (0.12 if primitive == "pick" else 0.08))
        for i, neighbor in enumerate(self.rows):
            if (
                i != index
                and np.linalg.norm(np.asarray(neighbor["position"])[:2] - target[:2]) < 0.20
            ):
                height = self.scene.layout.objects[i].half_size[2]
                clearance = max(
                    clearance, neighbor["position"][2] + height + obj.half_size[2] + offset + 0.015
                )
        above = np.r_[grasp[:2], clearance]
        other: Arm = "left" if arm == "right" else "right"
        original = self.probe.qpos[self.qids].copy()
        initial_probe = self.probe.qpos.copy()
        initial_attachments = dict(self.attachments)
        other_tcp = self.probe.site(f"{other}_tcp").xpos.copy()
        # Solve torso and ready posture together, then lock that torso while
        # proving the grasp/placement and retreat are inside the arm workspace.
        torso_seeds = workspace.preferred_torsos(relative_target) if workspace is not None else []
        choices = [(False, seed) for seed in torso_seeds]
        choices.extend([(False, original[:4]), (True, original[:4])])
        for use_torso, torso_seed in choices:
            self.probe.qpos[:] = initial_probe
            self.probe.qpos[self.qids[:4]] = torso_seed
            self.attachments = dict(initial_attachments)
            mujoco.mj_forward(self.model, self.probe)
            targets: dict[str, NDArray[Any]] = {arm: above}
            if (use_torso or np.max(np.abs(torso_seed - original[:4])) > 0.001) and any(
                r["held_by"] == other for r in self.rows
            ):
                targets[other] = other_tcp
            try:
                ready = self._solve(targets, torso=use_torso)
                ready[-2:] = original[-2:]
                self.probe.qpos[self.qids] = ready
                self._forward()
                if self._collisions(selected=index, arm=arm):
                    continue
                if not self._cartesian_corridor(grasp, index, arm):
                    continue
                if primitive == "pick":
                    self._attach(index, arm)
                else:
                    # The released object remains at the lower end while the
                    # opened hand retreats. This is a feasibility probe only.
                    self.attachments.pop(index)
                    self.probe.qpos[self.qids[active_indices(arm)[-1]]] = 0.05
                    self._forward()
                if not self._cartesian_corridor(above, index, arm):
                    continue
                distance = float(np.linalg.norm(pose[:2] - self.transport.start[:2]))
                rotation = abs(float(pose[2] - self.transport.start[2]))
                posture = float(np.linalg.norm(ready[:4] - original[:4]))
                arm_motion = float(
                    np.linalg.norm(
                        ready[list(active_indices(arm))[:-1]]
                        - original[list(active_indices(arm))[:-1]]
                    )
                )
                return ReachableStance(
                    arm,
                    (float(pose[0]), float(pose[1]), float(pose[2])),
                    (float(target[0]), float(target[1]), float(target[2])),
                    tuple(map(float, ready)),
                    float(clearance),
                    distance + 0.15 * rotation + 0.15 * posture + 0.02 * arm_motion,
                    bool(np.max(np.abs(ready[:4] - original[:4])) > 0.01),
                    "unknown"
                    if workspace is None
                    else "covered"
                    if workspace.covers(relative_target, ready[:4])
                    else "outside",
                )
            except RuntimeError:
                continue
        raise RuntimeError("No fixed-torso approach and retreat corridor from this stance")

    def candidates(
        self,
        primitive: Primitive,
        index: int,
        targets: list[NDArray[Any]],
        *,
        arm: str = "auto",
        limit: int = 3,
    ) -> list[ReachableStance]:
        if arm not in (*ARMS, "auto"):
            raise ValueError("Use auto, left or right")
        hands = [a for a in ARMS if arm in ("auto", a)]
        if primitive == "place":
            hands = [a for a in hands if self.rows[index]["held_by"] == a]
        else:
            hands = [a for a in hands if not any(r["held_by"] == a for r in self.rows)]
        results = []
        # Each eligible hand gets the same bounded search before ranking.
        for side in hands:
            deadline = time.monotonic() + 20
            found = 0
            workspace = self.workspaces.get(f"{primitive}-{side}")
            for target in targets:
                poses = stance_candidates(target, self.transport.start, side)
                if workspace is not None:
                    # Prefer demonstrated XY before searching new physical
                    # reach. Height and torso coverage are reported separately;
                    # absence of demonstrations must not masquerade as an IK limit.
                    poses.sort(
                        key=partial(_outside_demonstrated_xy, target=target, workspace=workspace)
                    )
                for pose in poses:
                    if time.monotonic() >= deadline:
                        break
                    try:
                        result = self.evaluate(primitive, side, index, target, pose)
                    except (RuntimeError, ValueError):
                        continue
                    results.append(result)
                    found += 1
                    if found >= limit:
                        break
                if found >= limit or time.monotonic() >= deadline:
                    break
        return sorted(results, key=lambda r: (r.policy_coverage != "covered", r.score))
