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

"""Classical commands preserve grip force and cancellation across state transitions."""

from concurrent.futures import CancelledError
import json

import numpy as np
import pytest

from dimos.control.tasks.trajectory_task.trajectory_task import (
    TrajectoryExecutionResult,
    TrajectoryExecutionStatus,
)
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.msgs.trajectory_msgs.TrajectoryStatus import TrajectoryState
from dimos.robot.galaxea.r1pro.apartment_navigation import CLASSICAL_POSITION_TASK
from dimos.robot.galaxea.r1pro.classical_skills import R1ProClassicalSkills
from dimos.robot.galaxea.r1pro.config import R1PRO_PLANAR_BASE
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS


@pytest.fixture
def skills(mocker):
    module = R1ProClassicalSkills()
    module._sim = mocker.Mock()
    module._control = mocker.Mock()
    module._control.execute_trajectory.return_value = TrajectoryExecutionResult(
        TrajectoryExecutionStatus.ACCEPTED
    )
    module._control.task_invoke.return_value = TrajectoryState.COMPLETED
    yield module
    module.stop()


def test_opening_left_preserves_other_grip_and_torso_preload(skills, mocker):
    positions = dict(zip(R1PRO_PICK_PLACE_JOINTS, [0.0] * 18 + [0.012, 0.017], strict=True))
    commands = dict(zip(R1PRO_PICK_PLACE_JOINTS, [0.01] + [0.0] * 19, strict=True))
    skills._sim.primitive_state.return_value = dict(
        joint_positions=positions, joint_commands=commands
    )
    drive = mocker.patch.object(skills, "_drive")
    mocker.patch.object(skills, "_pause")
    report = {}
    skills._gripper("left", 0.05, report)
    drive.assert_called_once_with([[0.01] + [0.0] * 19, [0.01] + [0.0] * 17 + [0.05, 0.0]], report)


def test_cancelled_motion_never_reaches_the_coordinator(skills):
    skills._cancel.set()
    with pytest.raises(CancelledError):
        skills._drive([[0.0] * 20, [0.01] * 18 + [0.0, 0.0]], {})
    skills._control.execute_trajectory.assert_not_called()


def test_empty_plan_is_rejected_before_execution(skills):
    with pytest.raises(RuntimeError, match="no executable waypoints"):
        skills._drive([], {})
    skills._control.execute_trajectory.assert_not_called()


def test_scene_fault_prevents_motion(skills):
    skills._sim.primitive_state.return_value = dict(error="lost cargo")
    with pytest.raises(RuntimeError, match="lost cargo"):
        skills._drive([[0.0] * 20, [0.01] * 18 + [0.0, 0.0]], {})
    skills._control.execute_trajectory.assert_not_called()


def test_small_joint_error_does_not_skip_a_needed_tcp_correction(skills, mocker):
    joints = dict(zip(R1PRO_PICK_PLACE_JOINTS, [0.0] * 20, strict=True))
    target = np.eye(4)
    displaced = target.copy()
    displaced[0, 3] = 0.01
    skills._sim.primitive_state.side_effect = [
        dict(joint_positions=joints),
        dict(tcp_poses={"left": displaced.tolist()}),
        dict(tcp_poses={"left": target.tolist()}),
        dict(joint_positions=joints),
    ]
    mocker.patch.object(skills, "_drive")
    skills._prepare_posture(
        dict(
            object="object_1",
            reachability=dict(arm="left", ready_joints=[0.0] * 20, pregrasp=target.tolist()),
        ),
        {},
    )
    skills._sim.classical_align.assert_called_once_with(0, "left", target.tolist())


def test_contact_loss_during_transfer_stops_the_segment(skills, mocker):
    positions = dict(zip(R1PRO_PICK_PLACE_JOINTS, [0.0] * 20, strict=True))
    before = dict(
        active=("place", "right"), objects=[dict(index=0, grasped=True, contacting_arms=["right"])]
    )
    during = dict(
        error=None,
        sim_time=2.0,
        joint_positions=positions,
        objects=[dict(index=0, grasped=False, contacting_arms=[])],
    )
    skills._sim.primitive_state.side_effect = [before, during]
    mocker.patch.object(skills, "_pause")
    with pytest.raises(RuntimeError, match="lost two-finger contact"):
        skills._drive([[0.0] * 20, [0.001] * 18 + [0.0, 0.0]], dict(phase="preplace"))


def test_aborted_trajectory_is_not_success_even_at_target(skills, mocker):
    positions = dict(zip(R1PRO_PICK_PLACE_JOINTS, [0.0] * 20, strict=True))
    state = dict(active=None, error=None, sim_time=2.0, joint_positions=positions, objects=[])
    skills._sim.primitive_state.return_value = state
    skills._control.task_invoke.return_value = TrajectoryState.ABORTED
    mocker.patch.object(skills, "_pause")
    with pytest.raises(RuntimeError, match="stopped before completing"):
        skills._drive([[0.0] * 20, [0.001] * 18 + [0.0, 0.0]], {})


@pytest.mark.parametrize("unsettled", ["velocity", "command"])
@pytest.mark.parametrize("point_count", [1, 2])
def test_completed_task_waits_for_delivered_and_settled_endpoint(
    skills, mocker, unsettled, point_count
):
    positions = dict.fromkeys(R1PRO_PICK_PLACE_JOINTS, 0.0)
    states = [dict(active=None, objects=[])]
    for tick in range(1, 7):
        states.append(
            dict(
                error=None,
                sim_time=float(tick),
                joint_positions=positions,
                joint_commands=dict.fromkeys(
                    positions, 0.005 if tick <= 3 and unsettled == "command" else 0.0
                ),
                joint_velocities=dict.fromkeys(
                    positions, 0.01 if tick <= 3 and unsettled == "velocity" else 0.0
                ),
            )
        )
    skills._sim.primitive_state.side_effect = states
    mocker.patch.object(skills, "_pause")

    report = dict(phase="preplace")
    skills._drive([list(np.zeros(20))] * point_count, report)

    assert skills._sim.primitive_state.call_count == 7
    assert json.loads(json.dumps(report))["last_trajectory_tracking"]["commands_delivered"] is True


@pytest.mark.parametrize("supports", [[], ["wrong_table"]])
def test_missing_intended_support_never_opens_hand(skills, mocker, supports):
    skills._sim.primitive_state.return_value = {
        "objects": [{"support_geoms": supports, "upright": True}],
        "tcp_poses": {"left": np.eye(4).tolist()},
    }
    mocker.patch.object(skills, "_pause")
    line = mocker.patch.object(skills, "_line")
    chosen = dict(index=0, arm="left", tcp=np.eye(4).tolist(), region={"support_geoms": ["table"]})
    with pytest.raises(RuntimeError, match="surface|support contact"):
        skills._seek_support(chosen, {})
    skills._control.execute_trajectory.assert_not_called()
    if not supports:
        # An under-tracking arm must not accumulate unexecuted descents into
        # a target that penetrates the support in the geometric planner.
        assert len(line.call_args_list) == 20
        assert all(call.args[2][2][3] == pytest.approx(-0.0005) for call in line.call_args_list)


def test_confirmed_support_finishes_without_further_descent(skills, mocker):
    skills._sim.primitive_state.return_value = {
        "objects": [{"support_geoms": ["table"], "upright": True}],
        "tcp_poses": {"right": np.eye(4).tolist()},
    }
    mocker.patch.object(skills, "_pause")
    line = mocker.patch.object(skills, "_line")
    report = {}
    skills._seek_support(
        dict(index=0, arm="right", tcp=np.eye(4).tolist(), region={"support_geoms": ["table"]}),
        report,
    )
    assert report["support_before_release"] == ["table"]
    line.assert_not_called()


def test_departing_checked_navigation_corridor_fails_before_route_continues(skills, mocker):
    path = [[0, 0, 0], [1, 0, 0]]
    skills._sim.validate_object_navigation.return_value = path
    skills._sim.primitive_state.side_effect = [
        dict(base_pose=[0, 0, 0], sim_time=1.0),
        dict(base_pose=[0.5, 0.05, 0], sim_time=2.0, error=None),
    ]
    mocker.patch.object(skills, "_pause")
    report = {}

    with pytest.raises(RuntimeError, match="checked tracking allowance"):
        skills._follow(path, report)

    assert report["commanded_paths"] == [path]
    assert report["max_navigation_tracking_error_m"] == pytest.approx(0.05)


def test_local_positioning_checks_sdk_plan_then_follows_measured_progress(skills, mocker):
    names = list(R1PRO_PLANAR_BASE.joint_names)
    trajectory = JointTrajectory(
        joint_names=[names[2], names[0], names[1]],
        points=[
            TrajectoryPoint(positions=[0, 0, 0], time_from_start=0),
            TrajectoryPoint(positions=[0.4, 0.1, 0.2], time_from_start=1),
        ],
    )
    skills._manipulation = mocker.Mock()
    skills._manipulation.plan_to_joints.return_value = mocker.Mock(
        succeeded=True, plan=mocker.Mock(trajectory=trajectory, plan_id="checked-local")
    )
    skills._sim.primitive_state.side_effect = [
        {"base_pose": [0, 0, 0]},
        {"base_pose": [0.1, 0.2, 0.4]},
    ]
    follow = mocker.patch.object(skills, "_execute_base_path")
    mocker.patch.object(skills, "_pause")
    report = {}

    skills._position_base({"base_waypoints": [[0, 0, 0], [0.1, 0.2, 0.4]]}, report)

    skills._sim.validate_primitive_base_plan.assert_called_once_with(trajectory)
    follow.assert_called_once_with(
        [[0, 0, 0], [0.1, 0.2, 0.4]],
        report,
        tracking_limit=0.015,
        task_name=CLASSICAL_POSITION_TASK,
        arrival_tolerance=0.03,
    )
    skills._manipulation.execute.assert_not_called()


def test_local_arrival_accepts_eighteen_mm_and_checks_the_measured_footprint(skills, mocker):
    skills._sim.primitive_state.side_effect = [
        dict(base_pose=[0, 0, 0], sim_time=1.0),
        dict(base_pose=[1.018, 0, 0], sim_time=2.0, error=None),
        dict(base_pose=[1.018, 0, 0], sim_time=2.5, error=None),
    ]
    mocker.patch.object(skills, "_pause")
    report = {}

    skills._execute_base_path(
        [[0, 0, 0], [1, 0, 0]],
        report,
        tracking_limit=0.015,
        task_name=CLASSICAL_POSITION_TASK,
        arrival_tolerance=0.03,
    )

    checked = skills._sim.validate_primitive_base_plan.call_args.args[0]
    assert checked.points[0].positions == [1.018, 0, 0]
    assert report["base_arrivals"][0]["position_error_m"] == pytest.approx(0.018)
    skills._control.task_invoke.assert_any_call(CLASSICAL_POSITION_TASK, "cancel", {})
    skills._sim.stop_primitive_base.assert_called_once()


def test_local_arrival_rejects_excess_drift_after_stopping(skills, mocker):
    skills._sim.primitive_state.side_effect = [
        dict(base_pose=[0, 0, 0], sim_time=1.0),
        dict(base_pose=[1.018, 0, 0], sim_time=2.0, error=None),
        dict(base_pose=[1.045, 0, 0], sim_time=2.5, error=None),
    ]
    mocker.patch.object(skills, "_pause")

    with pytest.raises(RuntimeError, match="outside the docking tolerance"):
        skills._execute_base_path(
            [[0, 0, 0], [1, 0, 0]],
            {},
            tracking_limit=0.015,
            task_name=CLASSICAL_POSITION_TASK,
            arrival_tolerance=0.03,
        )

    skills._sim.validate_primitive_base_plan.assert_not_called()
    skills._sim.stop_primitive_base.assert_called_once()


def test_local_arrival_tolerance_does_not_accept_an_obstructed_footprint(skills, mocker):
    skills._sim.primitive_state.return_value = dict(
        base_pose=[1.018, 0, 0], sim_time=1.0, error=None
    )
    skills._sim.validate_primitive_base_plan.side_effect = RuntimeError("blocked footprint")
    mocker.patch.object(skills, "_pause")
    report = {}

    with pytest.raises(RuntimeError, match="blocked footprint"):
        skills._execute_base_path(
            [[0, 0, 0], [1, 0, 0]],
            report,
            tracking_limit=0.015,
            task_name=CLASSICAL_POSITION_TASK,
            arrival_tolerance=0.03,
        )

    assert "base_arrivals" not in report
    skills._sim.stop_primitive_base.assert_called_once()


def test_transit_stops_at_accepted_endpoint_before_follower_overshoot(skills, mocker):
    path = [[0, 0, 0], [1, 0, 0]]
    skills._sim.validate_object_navigation.return_value = path
    skills._sim.primitive_state.side_effect = [
        dict(base_pose=[0, 0, 0], sim_time=1.0),
        dict(base_pose=[0.98, 0, 0], sim_time=2.0, error=None),
        dict(base_pose=[0.981, 0, 0], sim_time=2.5, error=None),
    ]
    mocker.patch.object(skills, "_pause")
    report = {}

    skills._follow(path, report)

    skills._sim.stop_primitive_base.assert_called_once()
    assert report["base_arrivals"][0]["position_error_m"] == pytest.approx(0.019)
    checked = skills._sim.validate_primitive_base_plan.call_args.args[0]
    assert checked.points[0].positions == [0.981, 0, 0]
