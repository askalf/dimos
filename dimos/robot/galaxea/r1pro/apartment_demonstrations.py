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

"""Offline initial-state sampling and local support goals for apartment ACT data."""

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.grasping_sim import VIRTUAL_BASE_JOINTS
from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_FPS
from dimos.robot.galaxea.r1pro.object_bimanual_task import BimanualPrimitiveTask
from dimos.robot.galaxea.r1pro.object_reachability import ObjectReachability, ReachableStance
from dimos.robot.galaxea.r1pro.placement_regions import PlacementRegion
from dimos.robot.galaxea.r1pro.primitive_scene import placement_options


def initialize_reachable_task(task: BimanualPrimitiveTask) -> ReachableStance:
    """Sample the robot's initial pose, then settle real physics before recording.

    This is training episode initialization, not a runtime positioning shortcut.
    Props retain their supported positions; grasps and releases are demonstrated
    through arm commands. No held object may enter this initialization path.
    """
    initial = task.inventory()
    if any(not r["released"] or not r["support_geoms"] for r in initial):
        raise ValueError("Reach sampling requires all objects initially supported and released")
    target = task.data.body(task.bottle_id).xpos.copy()
    candidates = ObjectReachability(task.scene_state).candidates(
        "pick", task.selected, [target], arm=task.arm, limit=3
    )
    if not candidates:
        raise RuntimeError("No physical local manipulation stance for the sampled source")
    stance = candidates[0]
    task.data.qpos[task.qids] = stance.ready_joints
    task.data.ctrl[task.aids] = stance.ready_joints
    for name, value in zip(VIRTUAL_BASE_JOINTS, stance.base_pose, strict=True):
        task.data.joint(name).qpos[0] = value
    task.data.ctrl[task.base_aids] = stance.base_pose
    task.data.qvel[:] = 0
    mujoco.mj_forward(task.model, task.data)
    for _ in range(40):
        task.step(task.data.ctrl[task.aids].copy())
        task.scene_state.validate(initial, arm=task.arm, selected=-1)
    # Refresh the SDK model frame and probe after the sampled base/torso pose.
    task.switch_arm(task.arm)
    task.select(task.selected)
    return stance


def choose_local_placement(
    task: BimanualPrimitiveTask, region: PlacementRegion, seed: int
) -> tuple[NDArray[np.float64], ReachableStance]:
    """Physically align a held object with a free local spot before its ACT labels.

    This is unlabelled positioning, just as deployment positions through DimOS.
    The selected object must remain physically grasped throughout the base move.
    """
    points, _ = placement_options(task, region, seed, check_gripper=False)
    rng = np.random.default_rng(seed)
    planner = ObjectReachability(task.scene_state)
    start = np.array([task.data.joint(n).qpos[0] for n in VIRTUAL_BASE_JOINTS])
    source = task.data.body(task.bottle_id).xpos.copy()
    # A local sample is bounded; a failed layout never causes an unbounded IK search.
    ordered = sorted(
        rng.permutation(len(points)),
        key=lambda i: float(np.linalg.norm(np.asarray(points[int(i)])[:2] - source[:2])),
    )
    for i in ordered[:40]:
        target: NDArray[Any] = np.asarray(points[int(i)])
        pose = start.copy()
        pose[:2] += target[:2] - source[:2]
        if not planner.transport.clear_pose_segment(start, pose):
            continue
        try:
            stance = planner.evaluate("place", task.arm, task.selected, target, pose)
        except (RuntimeError, ValueError):
            continue
        if not stance.torso_changed:
            initial = task.inventory()
            command = task.data.ctrl[task.aids].copy()
            for step in PlanarTransport.targets(
                [start.tolist(), pose.tolist()], R1PRO_PICK_PLACE_FPS, speed=0.08
            ):
                task.step(command, base_target=step)
                task.validate(initial)
                if not task.state.holding():
                    raise RuntimeError("Local base positioning lost the physical grasp")
            for _ in range(R1PRO_PICK_PLACE_FPS):
                task.step(command)
                task.validate(initial)
                if not task.state.holding():
                    raise RuntimeError("The held object slipped after local positioning")
            task.switch_arm(task.arm)
            return target, stance
    raise RuntimeError("No empty local placement has a fixed-torso manipulation corridor")
