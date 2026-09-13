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

"""Per-arm contact fusion preserves a supported grasp and the correct holding hand."""

from types import SimpleNamespace

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState


@pytest.fixture
def scene(mocker):
    def observer(model, data, layout, home, *, arm):
        guard = mocker.Mock()
        guard.collisions.return_value = []
        return SimpleNamespace(geometry=lambda index: readings[arm], guard=guard)

    readings = {
        side: dict(
            grasped=False,
            released=True,
            upright=True,
            support_geoms=[],
            object="item",
        )
        for side in ("left", "right")
    }
    mocker.patch(
        "dimos.robot.galaxea.r1pro.object_primitive_state.ObjectPackingState", side_effect=observer
    )
    data = mocker.Mock()
    data.site.return_value.xpos = np.array([0.3, 0.2, 0.9])
    data.site.return_value.xmat = np.eye(3).ravel()
    data.body.return_value.xpos = np.array([0.3, 0.2, 0.8])
    state = PrimitiveSceneState(
        mocker.Mock(), data, SimpleNamespace(objects=[SimpleNamespace(name="item")]), np.zeros(20)
    )
    return state, data, readings


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("supported", [False, True])
def test_inventory_reports_two_pad_contact_for_either_hand(scene, arm, supported):
    state, _, readings = scene
    readings[arm].update(grasped=True, released=False)
    for row in readings.values():
        row["support_geoms"] = ["table"] if supported else []
    row = state.inventory()[0]
    assert row["grasped"]
    assert row["grasping_arms"] == [arm]
    assert row["held_by"] == (None if supported else arm)
    assert not row["released"]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_turning_with_held_cargo_is_not_slip_but_relative_motion_is(scene, arm):
    state, data, readings = scene
    readings[arm].update(grasped=True, released=False)
    data.body.return_value.xpos = data.site.return_value.xpos - [0.02, 0, 0]
    before = state.inventory()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([1.0, -2.0, 0.0])
    for part in (data.site.return_value, data.body.return_value):
        part.xpos = rotation @ part.xpos + translation
    data.site.return_value.xmat = rotation.ravel()

    state.validate(before, arm=arm, selected=-1)

    after = state.inventory()[0]
    np.testing.assert_allclose(after["tcp_offset_local"], before[0]["tcp_offset_local"], atol=1e-12)
    assert np.linalg.norm(np.array(after["tcp_offset"]) - before[0]["tcp_offset"]) > 0.015
    data.body.return_value.xpos += rotation @ [0.02, 0, 0]
    with pytest.raises(RuntimeError, match="Lost or disturbed"):
        state.validate(before, arm=arm, selected=-1)
