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


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("supported", [False, True])
def test_inventory_reports_two_pad_contact_for_either_hand(mocker, arm, supported):
    def observer(model, data, layout, home, *, arm):
        return SimpleNamespace(geometry=lambda index: readings[arm])

    readings = {
        side: dict(
            grasped=side == arm,
            released=side != arm,
            upright=True,
            support_geoms=["table"] if supported else [],
            object="item",
        )
        for side in ("left", "right")
    }
    mocker.patch(
        "dimos.robot.galaxea.r1pro.object_primitive_state.ObjectPackingState", side_effect=observer
    )
    data = mocker.Mock()
    data.site.return_value.xpos = np.array([0.3, 0.2, 0.9])
    data.body.return_value.xpos = np.array([0.3, 0.2, 0.8])
    state = PrimitiveSceneState(
        mocker.Mock(), data, SimpleNamespace(objects=[SimpleNamespace(name="item")]), np.zeros(20)
    )
    row = state.inventory()[0]
    assert row["grasped"]
    assert row["grasping_arms"] == [arm]
    assert row["held_by"] == (None if supported else arm)
    assert not row["released"]
