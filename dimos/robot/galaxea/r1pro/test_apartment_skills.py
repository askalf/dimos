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

from dimos.manipulation.manipulation_spec import (
    ExecutionResult,
    ExecutionStatus,
    PlanResult,
    PlanStatus,
)
from dimos.robot.galaxea.r1pro.apartment_skills import R1ProApartmentSkills
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS


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


@pytest.mark.parametrize(("arm", "index"), [("left", 4), ("right", 11)])
def test_arm_positioning_cannot_be_skipped_when_torso_is_unchanged(skills, arm, index):
    positions = [0.0] * 20
    positions[index] = 0.5
    skills._sim.primitive_state.return_value["joint_positions"] = dict.fromkeys(
        R1PRO_PICK_PLACE_JOINTS, 0.0
    )
    skills._manipulation.plan_to_joints.return_value = PlanResult(
        PlanStatus.FAILED, "blocked approach"
    )
    selection = dict(
        object="object_1",
        reachability=dict(torso_changed=False, ready_joints=positions, arm=arm),
    )

    with pytest.raises(RuntimeError, match="blocked approach"):
        skills._prepare_posture(selection, {})

    goals = skills._manipulation.plan_to_joints.call_args.args[0]
    assert goals[f"{arm}_arm"].position[0] == 0.5
    skills._manipulation.execute.assert_not_called()


def test_measured_ready_posture_needs_no_motion_even_with_stale_torso_flag(skills):
    positions = [0.1] * 20
    skills._sim.primitive_state.return_value["joint_positions"] = dict(
        zip(R1PRO_PICK_PLACE_JOINTS, positions, strict=True)
    )
    selection = dict(reachability=dict(torso_changed=True, ready_joints=positions))
    report = {}

    skills._prepare_posture(selection, report)

    assert report["initial_posture_error_rad"] == 0.0
    skills._manipulation.plan_to_joints.assert_not_called()


def test_trajectory_completion_waits_for_fresh_measured_posture_samples(skills, mocker):
    positions = [0.1] * 20
    parked = dict(joint_positions=dict.fromkeys(R1PRO_PICK_PLACE_JOINTS, 0.0))
    settled = dict(zip(R1PRO_PICK_PLACE_JOINTS, positions, strict=True))
    lagging = {**settled, "r1pro/right_arm_joint7": 0.07}
    skills._sim.primitive_state.side_effect = [
        parked,
        dict(error=None),  # The trajectory timer finishes.
        dict(error=None, sim_time=1.0, joint_positions=lagging),
        dict(error=None, sim_time=1.1, joint_positions=settled),
        dict(error=None, sim_time=1.1, joint_positions=settled),  # Repeated sample.
        dict(error=None, sim_time=1.2, joint_positions=settled),
        dict(error=None, sim_time=1.3, joint_positions=settled),
    ]
    mocker.patch.object(skills, "_pause")
    skills._manipulation.execute.return_value = ExecutionResult(ExecutionStatus.ACCEPTED)
    skills._manipulation.wait_for_execution.return_value = ExecutionResult(
        ExecutionStatus.COMPLETED
    )
    selection = dict(
        object="object_1",
        reachability=dict(torso_changed=False, ready_joints=positions, arm="right"),
    )
    report = {}

    skills._prepare_posture(selection, report)

    assert report["final_posture_error_rad"] == 0.0
    assert skills._sim.primitive_state.call_count == 7
    skills._apartment.validate_apartment_posture.assert_called_once()
    skills._manipulation.execute.assert_called_once()
