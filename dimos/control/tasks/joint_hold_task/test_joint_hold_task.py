# Copyright 2025-2026 Dimensional Inc.
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

from __future__ import annotations

from dimos.control.task import ControlMode, CoordinatorState, JointStateSnapshot
from dimos.control.tasks.joint_hold_task.joint_hold_task import (
    JointHoldTask,
    JointHoldTaskConfig,
)
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3

JOINTS = ["arm/joint1", "arm/joint2"]


def _task(**overrides: object) -> JointHoldTask:
    config = JointHoldTaskConfig(joint_names=list(JOINTS), **overrides)  # type: ignore[arg-type]
    return JointHoldTask("hold_arms", config)


def _state(positions: dict[str, float], t_now: float) -> CoordinatorState:
    return CoordinatorState(
        joints=JointStateSnapshot(joint_positions=dict(positions)),
        t_now=t_now,
    )


def _drive(vx: float = 0.0, wz: float = 0.0) -> Twist:
    return Twist(linear=Vector3(vx, 0.0, 0.0), angular=Vector3(0.0, 0.0, wz))


def test_idle_until_the_base_moves() -> None:
    task = _task()
    assert not task.is_active()
    assert task.compute(_state({"arm/joint1": 0.1, "arm/joint2": 0.2}, 1.0)) is None


def test_a_slow_crawl_is_not_movement() -> None:
    # Below both thresholds: odometry noise must not pin the arms.
    task = _task()
    task.on_twist_command(_drive(vx=0.001, wz=0.001), t_now=1.0)
    assert not task.is_active()


def test_holds_the_pose_the_arm_was_in_when_the_base_started() -> None:
    task = _task()
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    assert task.is_active()

    out = task.compute(_state({"arm/joint1": 0.1, "arm/joint2": 0.2}, 1.0))
    assert out is not None
    assert out.mode is ControlMode.SERVO_POSITION
    assert out.joint_names == JOINTS
    assert out.positions == [0.1, 0.2]

    # The arm is swinging; the command stays on the latched pose, not the new one.
    out = task.compute(_state({"arm/joint1": 0.4, "arm/joint2": -0.3}, 1.05))
    assert out is not None
    assert out.positions == [0.1, 0.2]


def test_releases_once_the_base_has_been_quiet() -> None:
    task = _task(release_after_s=0.5)
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    assert task.compute(_state({"arm/joint1": 0.1, "arm/joint2": 0.2}, 1.2)) is not None

    # No further twist: the release window is measured on the coordinator clock,
    # so a stream that simply stops still lets go.
    assert task.compute(_state({"arm/joint1": 0.1, "arm/joint2": 0.2}, 1.6)) is None
    assert not task.is_active()


def test_continued_driving_extends_the_hold() -> None:
    task = _task(release_after_s=0.5)
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    task.on_twist_command(_drive(vx=0.3), t_now=1.4)
    assert task.compute(_state({"arm/joint1": 0.1, "arm/joint2": 0.2}, 1.6)) is not None


def test_yaw_alone_counts_as_movement() -> None:
    task = _task()
    task.on_twist_command(_drive(wz=0.4), t_now=1.0)
    assert task.is_active()


def test_preemption_drops_the_latch_and_relatches_where_the_arm_ended_up() -> None:
    task = _task()
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    task.compute(_state({"arm/joint1": 0.1, "arm/joint2": 0.2}, 1.0))

    # A plan takes the joints and moves the arm somewhere else.
    task.on_preempted("joint_trajectory", frozenset({"arm/joint1"}))
    task.on_twist_command(_drive(vx=0.3), t_now=2.0)

    out = task.compute(_state({"arm/joint1": 0.9, "arm/joint2": -0.4}, 2.0))
    assert out is not None
    assert out.positions == [0.9, -0.4]


def test_preemption_of_unrelated_joints_keeps_the_latch() -> None:
    task = _task()
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    task.compute(_state({"arm/joint1": 0.1, "arm/joint2": 0.2}, 1.0))

    task.on_preempted("some_base_task", frozenset({"base/vx"}))

    out = task.compute(_state({"arm/joint1": 0.9, "arm/joint2": -0.4}, 1.1))
    assert out is not None
    assert out.positions == [0.1, 0.2]


def test_a_missing_joint_holds_nothing() -> None:
    # Pinning a subset would leave the rest to swing into the held joints,
    # which is the collision this task exists to prevent.
    task = _task()
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    assert task.compute(_state({"arm/joint1": 0.1}, 1.0)) is None


def test_latches_on_a_later_tick_once_the_state_is_complete() -> None:
    task = _task()
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    assert task.compute(_state({"arm/joint1": 0.1}, 1.0)) is None

    out = task.compute(_state({"arm/joint1": 0.15, "arm/joint2": 0.25}, 1.02))
    assert out is not None
    assert out.positions == [0.15, 0.25]


def test_release_and_disable() -> None:
    task = _task()
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    task.release()
    assert not task.is_active()

    task.set_enabled(False)
    task.on_twist_command(_drive(vx=0.3), t_now=2.0)
    assert not task.is_active()
    assert task.get_status()["enabled"] is False

    task.set_enabled(True)
    task.on_twist_command(_drive(vx=0.3), t_now=3.0)
    assert task.is_active()


def test_hold_without_a_twist() -> None:
    task = _task()
    assert task.hold(t_now=1.0) is True
    out = task.compute(_state({"arm/joint1": 0.1, "arm/joint2": 0.2}, 1.0))
    assert out is not None
    assert out.positions == [0.1, 0.2]


def test_claim_sits_below_the_trajectory_task() -> None:
    from dimos.control.tasks.trajectory_task.trajectory_task import JointTrajectoryTaskConfig

    claim = _task().claim()
    assert claim.mode is ControlMode.SERVO_POSITION
    assert claim.joints == frozenset(JOINTS)
    assert claim.priority < JointTrajectoryTaskConfig(joint_names=list(JOINTS)).priority


def test_requires_joints() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least one joint"):
        JointHoldTask("hold_arms", JointHoldTaskConfig(joint_names=[]))


def test_start_and_stop_drive_the_enable_flag() -> None:
    # auto_start=True makes the coordinator call start(); without it the task
    # is added and then immediately torn down again.
    task = _task()
    task.stop()
    task.on_twist_command(_drive(vx=0.3), t_now=1.0)
    assert not task.is_active()

    task.start()
    task.on_twist_command(_drive(vx=0.3), t_now=2.0)
    assert task.is_active()
