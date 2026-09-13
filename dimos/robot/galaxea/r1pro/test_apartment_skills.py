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

"""Interactive arm selection preserves explicit intent and never infers a release."""

import json

import pytest

from dimos.manipulation.manipulation_spec import ExecutionResult, ExecutionStatus
from dimos.robot.galaxea.r1pro.apartment_skills import R1ProApartmentSkills


@pytest.fixture
def skills(mocker, tmp_path):
    module = R1ProApartmentSkills()
    module._sim = mocker.Mock()
    module._apartment = mocker.Mock()
    module._control = mocker.Mock()
    module._manipulation = mocker.Mock()
    module._manipulation.cancel.return_value = ExecutionResult(ExecutionStatus.NO_EXECUTION)
    module._sim.primitive_state.return_value = dict(
        objects=[
            dict(
                id="object_1",
                object="task_object_1",
                index=0,
                shape="box",
                kind="toy_block",
                upright=True,
                released=True,
                settled=True,
                support_geoms=["table"],
                distance_m=0.5,
                left_m=0.3,
            )
        ],
        held_objects={"left": None, "right": None},
        regions=["tray", "table", "kitchen"],
    )
    module._sim.prepare_primitive_session.return_value = dict(output=str(tmp_path))
    for primitive in ("pick", "place"):
        for arm in ("left", "right"):
            policy = mocker.Mock()
            policy.stop_rollout.return_value = dict(active=False)
            setattr(module, f"_{primitive}_{arm}", policy)
    mocker.patch.object(module, "_execute")
    yield module
    module.stop()


def test_auto_pick_uses_reachable_hand_and_does_not_place(skills):
    skills._apartment.assess_object_reachability.return_value = dict(candidates=[dict(arm="left")])
    assert json.loads(skills.pick_object("toy_block"))["accepted"]
    assert json.loads(skills.wait_for_action(5))["success"]
    call = skills._execute.call_args
    assert call.args[:4] == ("pick", "left", 0, "")
    assert skills._execute.call_count == 1
    skills._place_left.start_rollout.assert_not_called()
    skills._sim.reset.assert_not_called()


def test_explicit_hand_cannot_be_replaced_even_by_an_inconsistent_assessment(skills):
    skills._apartment.assess_object_reachability.return_value = dict(candidates=[dict(arm="right")])
    assert json.loads(skills.pick_object("object_1", "left"))["accepted"]
    result = json.loads(skills.wait_for_action(5))
    assert result["state"] == "failed"
    assert not result["recovery_required"]
    skills._execute.assert_not_called()


def test_place_auto_requires_one_unambiguous_held_object(skills):
    held = skills._sim.primitive_state.return_value["held_objects"]
    held.update(left="object_1", right="object_2")
    assert not json.loads(skills.place_object("kitchen"))["accepted"]
    skills._execute.assert_not_called()
    held["right"] = None
    assert json.loads(skills.place_object("kitchen"))["accepted"]
    assert json.loads(skills.wait_for_action(5))["success"]
    assert skills._execute.call_args.args[:4] == ("place", "left", -1, "kitchen")
