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

"""Primitive commands preserve the unrequested hand and pick has no destination input."""

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.object_primitives import (
    active_indices,
    apply_primitive_action,
    context_indices,
    primitive_observation,
    primitive_profile,
)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_primitive_cannot_command_other_hand_or_torso(arm):
    held = np.linspace(-0.5, 0.5, 20)
    held[18:] = [0.021, 0.013]
    action = np.linspace(0.2, 0.8, 8)
    result = apply_primitive_action(held, action, arm)
    np.testing.assert_array_equal(
        result[list(context_indices(arm))], held[list(context_indices(arm))]
    )
    np.testing.assert_array_equal(result[list(active_indices(arm))], action)
    assert held[18:].tolist() == [0.021, 0.013]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_pick_input_is_independent_of_destination_but_place_is_conditioned(arm):
    joints = np.arange(20, dtype=float)
    geometry = np.arange(52, dtype=float)
    changed = geometry.copy()
    changed[3:6] = [-100, 700, 80]
    first = primitive_observation("pick", arm, joints, geometry)
    second = primitive_observation("pick", arm, joints, changed)
    np.testing.assert_array_equal(
        first["observation.environment_state"], second["observation.environment_state"]
    )
    place = primitive_observation("place", arm, joints, changed)
    np.testing.assert_array_equal(place["observation.environment_state"][3:6], [-100, 700, 80])
    np.testing.assert_array_equal(
        first["observation.environment_state"][-12:], joints[list(context_indices(arm))]
    )


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("primitive", ["pick", "place"])
def test_runtime_profile_owns_only_requested_arm_and_matching_measured_joints(primitive, arm):
    profile = primitive_profile(primitive, arm)
    assert profile.action_state_key == "observation.state"
    assert profile.action.demonstration.joints == (
        *(f"r1pro/{arm}_arm_joint{i}" for i in range(1, 8)),
        f"r1pro/{arm}_gripper",
    )
    assert profile.observations["observation.images.wrist"].stream == f"{arm}_wrist"
    features = profile.observations["observation.environment_state"].features
    assert any("destination" in name for name in features) == (primitive == "place")


@pytest.mark.parametrize("action", [np.zeros(20), np.full(8, np.nan)])
def test_invalid_policy_actions_fail_before_mutating_commands(action):
    held = np.zeros(20)
    with pytest.raises(ValueError):
        apply_primitive_action(held, action, "right")
    np.testing.assert_array_equal(held, np.zeros(20))
