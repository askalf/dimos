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

"""ACT migration retains learned selected-arm mappings in physical coordinates."""

from dimos_lerobot.prepare_primitive_act import migrate_weights
import pytest
import torch

from dimos.robot.galaxea.r1pro.object_primitives import (
    MIRROR_ARM_SIGNS,
    active_indices,
    goal_indices,
)


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("primitive", ["pick", "place"])
def test_migrated_action_head_preserves_physical_selected_arm_mapping(arm, primitive):
    source = {
        "model.action_head.weight": torch.arange(60, dtype=torch.float32).reshape(20, 3),
        "model.action_head.bias": torch.arange(20, dtype=torch.float32),
    }
    target = {
        "model.action_head.weight": torch.zeros(8, 3),
        "model.action_head.bias": torch.zeros(8),
    }
    copied = {k: v.clone() for k, v in source.items()}
    weights = migrate_weights(source, target, primitive, arm)
    latent = torch.tensor([0.4, 0.3, -0.1])
    indices = list(active_indices("right"))
    sign = (
        torch.as_tensor(MIRROR_ARM_SIGNS, dtype=torch.float32) if arm == "left" else torch.ones(8)
    )
    mean, scale = (
        torch.arange(8, dtype=torch.float32) * 0.1,
        torch.arange(1, 9, dtype=torch.float32),
    )
    original = (source["model.action_head.weight"] @ latent + source["model.action_head.bias"])[
        indices
    ] * scale + mean
    migrated = (
        weights["model.action_head.weight"] @ latent + weights["model.action_head.bias"]
    ) * scale + sign * mean
    torch.testing.assert_close(migrated, original * sign)
    for key in source:
        torch.testing.assert_close(source[key], copied[key])


def test_pick_projection_excludes_destination_and_new_context_starts_neutral():
    key = "model.encoder_env_state_input_proj.weight"
    source = {key: torch.arange(104, dtype=torch.float32).reshape(2, 52)}
    target = {key: torch.empty(2, 61)}
    migrated = migrate_weights(source, target, "pick", "right")[key]
    torch.testing.assert_close(migrated[:, :49], source[key][:, list(goal_indices("pick"))])
    torch.testing.assert_close(migrated[:, 49:], torch.zeros(2, 12))


def test_unexpected_model_dimension_is_rejected():
    with pytest.raises(ValueError, match="dimension"):
        migrate_weights(
            {"unchanged": torch.zeros(2)}, {"unchanged": torch.zeros(3)}, "pick", "right"
        )
