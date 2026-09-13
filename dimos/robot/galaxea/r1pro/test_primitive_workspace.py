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

"""Policy starting-state coverage is measured, and is separate from IK reach."""

import json

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.primitive_workspace import audit_workspace


@pytest.mark.parametrize("arm,sign", [("left", 1), ("right", -1)])
@pytest.mark.parametrize(
    "primitive,length,home_index,target_index", [("pick", 61, 18, 0), ("place", 64, 21, 3)]
)
def test_audit_recovers_base_relative_targets_and_rejects_unseen_height_and_torso(
    tmp_path, arm, sign, primitive, length, home_index, target_index
):
    environment = np.zeros((1, length), dtype=np.float32)
    # Target is (.42, +/-.32, .77); measured TCP is (.35, +/-.30, .90).
    environment[0, target_index : target_index + 3] = [0.07, sign * 0.02, -0.13]
    environment[0, home_index : home_index + 3] = [-0.02, sign * 0.02, 0.02]
    environment[0, -12:-8] = [0.4, -0.4, -0.2, 0]
    np.savez(tmp_path / "episode.npz", **{"observation.environment_state": environment})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                profile=f"r1pro-sim-object-{primitive}-{arm}-v1",
                primitive=primitive,
                arm=arm,
                episodes=[dict(file="episode.npz", success=True)],
            )
        )
    )

    result = audit_workspace(manifest, primitive, arm)

    np.testing.assert_allclose(result.target_min, [0.42, sign * 0.32, 0.77], atol=1e-7)
    assert result.covers(np.array([0.42, sign * 0.32, 0.77]), np.array([0.4, -0.4, -0.2, 0]))
    assert not result.covers(np.array([0.42, sign * 0.32, 0.85]), np.array([0.4, -0.4, -0.2, 0]))
    assert not result.covers(np.array([0.42, sign * 0.32, 0.77]), np.array([0.4, -0.2, -0.2, 0]))


def test_workspace_does_not_accept_another_hand_profile(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(dict(profile="r1pro-sim-object-pick-left-v1", arm="left", primitive="pick"))
    )
    with pytest.raises(ValueError, match="does not match"):
        audit_workspace(manifest, "pick", "right")
