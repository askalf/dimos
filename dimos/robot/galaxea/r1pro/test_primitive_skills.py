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

"""Interactive selection and actuator ownership contracts for primitive ACT."""

import json

import pytest

from dimos.manipulation.manipulation_spec import ExecutionResult, ExecutionStatus
from dimos.robot.galaxea.r1pro.primitive_blueprint import (
    POLICIES,
    R1ProPrimitiveCoordinator,
    r1pro_primitives_sim,
)
from dimos.robot.galaxea.r1pro.primitive_skills import (
    R1ProPrimitiveSkills,
    resolve_primitive_object,
)


@pytest.fixture
def objects():
    return [
        dict(
            id=f"object_{i + 1}",
            object=f"task_object_{i + 1}",
            index=i,
            shape=shape,
            upright=True,
            released=True,
            settled=True,
            support_geoms=["bin_floor" if i == 0 else "table"],
            inside=i == 0,
            left_m=y,
            distance_m=d,
        )
        for i, (shape, y, d) in enumerate(
            [("box", -0.4, 0.7), ("cylinder", 0.1, 0.3), ("cylinder", 0.3, 0.5)]
        )
    ]


@pytest.mark.parametrize(
    "selector,expected",
    [
        ("rightmost", 0),
        ("leftmost", 2),
        ("furthest", 0),
        ("nearest", 1),
        ("object_1", 0),
        ("box", 0),
    ],
)
def test_selection_includes_supported_tray_objects(objects, selector, expected):
    assert resolve_primitive_object(objects, selector) == expected


def test_explicit_unavailable_object_never_substitutes_another(objects):
    objects[0]["released"] = False
    with pytest.raises(ValueError, match="not a supported"):
        resolve_primitive_object(objects, "object_1")


def test_ambiguous_shape_requires_an_id(objects):
    with pytest.raises(ValueError, match="object ID"):
        resolve_primitive_object(objects, "cylinder")


def test_arm_policy_tasks_cannot_command_the_other_hand_or_torso():
    control = next(
        a for a in r1pro_primitives_sim.active_blueprints if a.module is R1ProPrimitiveCoordinator
    )
    tasks = {t.name: t for t in control.kwargs["tasks"]}
    claims = {arm: set(tasks[f"primitive_{arm}"].joint_names) for arm in ("left", "right")}
    assert not claims["left"].intersection(claims["right"])
    for (primitive, arm), cls in POLICIES.items():
        atom = next(a for a in r1pro_primitives_sim.active_blueprints if a.module is cls)
        assert atom.kwargs["trajectory_task_name"] == f"primitive_{arm}"
        assert set(cls.profile.action.demonstration.joints) == claims[arm]
        assert all(f"/{arm}_" in name for name in claims[arm])
        assert (
            r1pro_primitives_sim.remapping_map[(R1ProPrimitiveSkills.name, f"_{primitive}_{arm}")]
            == cls
        )


@pytest.fixture(params=list(POLICIES.values()))
def policy(request):
    instance = request.param(artifact="unused", task="No rollout")
    yield instance
    instance.stop()


def test_primitive_policies_resolve_the_installed_lerobot_project(policy):
    assert policy.runtime_project.name == "python"
    assert (policy.runtime_project / "dimos_lerobot/primitive_runtime.py").is_file()


@pytest.fixture
def skills(mocker, tmp_path, objects):
    module = R1ProPrimitiveSkills()
    module._sim = mocker.Mock()
    module._control = mocker.Mock()
    module._manipulation = mocker.Mock()
    module._manipulation.cancel.return_value = ExecutionResult(ExecutionStatus.NO_EXECUTION)
    module._sim.primitive_state.return_value = dict(
        objects=objects, held_objects={"left": None, "right": None}, regions=["tray", "table"]
    )
    module._sim.prepare_primitive_session.return_value = dict(output=str(tmp_path))
    for primitive, arm in POLICIES:
        policy = mocker.Mock()
        policy.stop_rollout.return_value = dict(active=False)
        setattr(module, f"_{primitive}_{arm}", policy)
    yield module
    module.stop()


def test_failed_recovery_keeps_the_recovery_requirement(skills):
    skills._action["recovery_required"] = True
    skills._sim.primitive_recovery.side_effect = RuntimeError("Unsupported contact")
    assert json.loads(skills.recover_action())["accepted"]
    result = json.loads(skills.wait_for_action(5))
    assert result["state"] == "failed"
    assert result["recovery_required"] is True
    skills._control.execute_trajectory.assert_not_called()
    skills._sim.reset.assert_not_called()


def test_recovering_a_confirmed_hold_never_opens_or_resets_either_hand(skills):
    skills._action["recovery_required"] = True
    skills._sim.primitive_recovery.return_value = dict(mode="hold", arm="left", object="object_1")
    skills._sim.finish_primitive_recovery.return_value = dict(
        mode="hold", arm="left", object="object_1"
    )
    assert json.loads(skills.recover_action())["accepted"]
    result = json.loads(skills.wait_for_action(5))
    assert result["success"] is True
    assert result["recovery_required"] is False
    skills._control.execute_trajectory.assert_not_called()
    skills._manipulation.plan_to_joints.assert_not_called()
    skills._sim.reset.assert_not_called()


def test_occupied_requested_hand_is_rejected_without_selecting_the_other(skills):
    skills._sim.primitive_state.return_value["held_objects"]["left"] = "object_2"
    result = json.loads(skills.pick_object("object_1", "left"))
    assert result["accepted"] is False
    skills._sim.prepare_primitive.assert_not_called()
    skills._pick_right.start_rollout.assert_not_called()
    skills._pick_left.start_rollout.assert_not_called()
