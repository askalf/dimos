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

"""Arm-local ACT contracts: pick has no destination; place starts with a held object."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from dimos.imitation.profile import (
    ImageSource,
    JointPositionAction,
    JointPositionSource,
    PolicyIOProfile,
    VectorSource,
)
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_IO, R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.object_packing import OBJECT_GOAL_FEATURES

Arm = Literal["left", "right"]
Primitive = Literal["pick", "place"]
ARMS: tuple[Arm, ...] = ("right", "left")
PRIMITIVES: tuple[Primitive, ...] = ("pick", "place")
MIRROR_ARM_SIGNS = np.array([1, -1, -1, 1, -1, 1, -1, 1], dtype=np.float64)


def active_indices(arm: Arm) -> tuple[int, ...]:
    if arm == "right":
        return (*range(11, 18), 19)
    if arm == "left":
        return (*range(4, 11), 18)
    raise ValueError("Arm must be left or right")


def context_indices(arm: Arm) -> tuple[int, ...]:
    return tuple(i for i in range(20) if i not in active_indices(arm))


def goal_indices(primitive: Primitive) -> tuple[int, ...]:
    if primitive not in PRIMITIVES:
        raise ValueError("Primitive must be pick or place")
    return tuple(
        i for i in range(len(OBJECT_GOAL_FEATURES)) if primitive == "place" or i not in (3, 4, 5)
    )


def primitive_profile(primitive: Primitive, arm: Arm) -> PolicyIOProfile:
    joints = tuple(R1PRO_PICK_PLACE_JOINTS[i] for i in active_indices(arm))
    features = tuple(OBJECT_GOAL_FEATURES[i] for i in goal_indices(primitive)) + tuple(
        f"context_{R1PRO_PICK_PLACE_JOINTS[i]}" for i in context_indices(arm)
    )
    return PolicyIOProfile(
        name=f"r1pro-sim-object-{primitive}-{arm}-v1",
        robot_type=f"r1pro_sim_object_{primitive}_{arm}",
        observations={
            "observation.images.head": ImageSource(stream="color_image", shape=(160, 160, 3)),
            "observation.images.wrist": ImageSource(stream=f"{arm}_wrist", shape=(160, 160, 3)),
            "observation.state": JointPositionSource(
                stream="coordinator_joint_state", joints=joints
            ),
            "observation.environment_state": VectorSource(
                stream=f"{primitive}_{arm}_goal", features=features
            ),
        },
        action=JointPositionAction(
            key="action",
            demonstration=JointPositionSource(
                stream="applied_joint_position_command", joints=joints
            ),
        ),
        sync=R1PRO_PICK_PLACE_IO.sync,
        quality=R1PRO_PICK_PLACE_IO.quality,
    )


def primitive_observation(
    primitive: Primitive, arm: Arm, joints: NDArray[Any], geometry: NDArray[Any]
) -> dict[str, NDArray[np.float32]]:
    """Project measured state and geometry expressed relative to the selected arm TCP.

    Geometry uses the legacy feature order for reuse, but pick deliberately drops
    the three destination entries. Images are supplied separately from the real
    active wrist; no rendered camera or joint mirroring is assumed at runtime.
    """
    if (
        joints.shape != (20,)
        or geometry.shape != (52,)
        or not np.isfinite(joints).all()
        or not np.isfinite(geometry).all()
    ):
        raise ValueError("Expected finite measured joints (20) and selected-TCP geometry (52)")
    return {
        "observation.state": joints[list(active_indices(arm))].astype(np.float32),
        "observation.environment_state": np.concatenate(
            (geometry[list(goal_indices(primitive))], joints[list(context_indices(arm))])
        ).astype(np.float32),
    }


def apply_primitive_action(
    held_command: NDArray[Any], action: NDArray[Any], arm: Arm
) -> NDArray[np.float64]:
    """Offline executor projection; preserve every unowned actuator exactly."""
    if (
        held_command.shape != (20,)
        or action.shape != (8,)
        or not np.isfinite(action).all()
        or not np.isfinite(held_command).all()
    ):
        raise ValueError("Expected finite full command (20) and arm action (8)")
    result = np.asarray(held_command, dtype=np.float64).copy()
    result[list(active_indices(arm))] = action
    return result
