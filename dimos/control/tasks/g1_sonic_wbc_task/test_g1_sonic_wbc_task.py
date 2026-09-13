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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dimos.control.coordinator import ControlCoordinator
from dimos.control.task import CoordinatorState, JointStateSnapshot
from dimos.control.tasks.g1_sonic_wbc_task.g1_sonic_wbc_task import (
    G1SonicWBCTask,
    G1SonicWBCTaskConfig,
    SonicControlState,
    _create_task,
)
from dimos.control.tasks.g1_sonic_wbc_task.sonic_pipeline import DEFAULT_ANGLES_DDS
from dimos.control.tasks.g1_sonic_wbc_task.sonic_safety import damping_commands
from dimos.hardware.whole_body.spec import IMUState

_JOINT_NAMES = [f"joint_{index}" for index in range(29)]


def _state(t_now: float, positions: float = 0.0) -> CoordinatorState:
    return CoordinatorState(
        joints=JointStateSnapshot(
            joint_positions=dict.fromkeys(_JOINT_NAMES, positions),
            joint_velocities=dict.fromkeys(_JOINT_NAMES, 0.0),
        ),
        imu={"g1": IMUState()},
        t_now=t_now,
        dt=0.02,
    )


@pytest.fixture
def make_task(mocker: Any):
    pipeline_class = mocker.patch(
        "dimos.control.tasks.g1_sonic_wbc_task.g1_sonic_wbc_task.SonicPipeline"
    )
    pipeline = pipeline_class.return_value
    pipeline.step.return_value = np.zeros(29, dtype=np.float32)
    pipeline.snapshot.return_value = {"stream_active": False}
    tasks = []

    def factory(
        *,
        auto_arm: bool,
        default_ramp_seconds: float,
        auto_dry_run: bool = False,
    ) -> G1SonicWBCTask:
        config = G1SonicWBCTaskConfig(
            encoder_onnx=Path("encoder.onnx"),
            decoder_onnx=Path("decoder.onnx"),
            planner_onnx=Path("planner.onnx"),
            joint_names=_JOINT_NAMES,
            auto_arm=auto_arm,
            auto_dry_run=auto_dry_run,
            default_ramp_seconds=default_ramp_seconds,
        )
        task = G1SonicWBCTask("sonic", config, mocker.MagicMock())
        tasks.append(task)
        return task

    yield factory, pipeline
    for task in tasks:
        task.stop()


def test_auto_arm_finishes_ramp_before_first_policy_step(make_task: Any) -> None:
    factory, pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)
    task.start()

    initialization_output = task.compute(_state(1.0))

    assert task.control_state is SonicControlState.CONTROL
    assert initialization_output is not None
    assert initialization_output.positions == pytest.approx(DEFAULT_ANGLES_DDS.tolist())
    pipeline.step.assert_not_called()

    task.compute(_state(1.02))

    pipeline.step.assert_called_once()


def test_start_without_auto_arm_holds_measured_pose(make_task: Any) -> None:
    factory, pipeline = make_task
    task = factory(auto_arm=False, default_ramp_seconds=3.0)
    task.start()

    output = task.compute(_state(10.0, positions=0.25))

    assert task.control_state is SonicControlState.UNARMED
    assert output is not None and output.positions == pytest.approx([0.25] * 29)
    snapshot = task.state_snapshot()
    assert snapshot["active"] is True
    assert snapshot["armed"] is False
    assert snapshot["arming"] is False
    assert snapshot["arm_pending"] is False
    assert snapshot["dry_run"] is False
    assert snapshot["arming_duration"] == 3.0
    pipeline.step.assert_not_called()


def test_arm_snapshots_current_pose_then_ramps_to_default(make_task: Any) -> None:
    factory, pipeline = make_task
    task = factory(auto_arm=False, default_ramp_seconds=3.0)
    task.start()
    task.compute(_state(9.0, positions=0.25))
    assert task.arm()
    assert task.state_snapshot()["arm_pending"] is True

    first = task.compute(_state(10.0, positions=0.25))
    halfway = task.compute(_state(11.5, positions=0.25))
    complete = task.compute(_state(13.0, positions=0.25))

    assert first is not None and first.positions == pytest.approx([0.25] * 29)
    assert halfway is not None
    expected_halfway = 0.25 + 0.5 * (DEFAULT_ANGLES_DDS - 0.25)
    assert halfway.positions == pytest.approx(expected_halfway.tolist())
    assert complete is not None and complete.positions == pytest.approx(DEFAULT_ANGLES_DDS.tolist())
    assert task.control_state is SonicControlState.CONTROL
    pipeline.step.assert_not_called()


def test_manual_arm_starts_policy_only_after_ramp(make_task: Any) -> None:
    factory, pipeline = make_task
    task = factory(auto_arm=False, default_ramp_seconds=0.0)
    task.start()
    task.compute(_state(1.0))

    assert task.control_state is SonicControlState.UNARMED
    assert task.arm()

    task.compute(_state(1.02))
    assert task.control_state is SonicControlState.CONTROL
    pipeline.step.assert_not_called()

    task.compute(_state(1.04))
    pipeline.step.assert_called_once()


def test_disarm_returns_to_measured_pose_hold_and_planner(make_task: Any) -> None:
    factory, pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)
    task.start()
    task.compute(_state(1.0))
    task.compute(_state(1.02))
    task._select_stream_reference(True)

    assert task.disarm()
    hold = task.compute(_state(2.0, positions=0.3))

    snapshot = task.state_snapshot()
    assert snapshot["control_state"] == "unarmed"
    assert snapshot["armed"] is False
    assert snapshot["reference_source"] == "planner"
    assert hold is not None and hold.positions == pytest.approx([0.3] * 29)
    pipeline.reset.assert_called()


def test_reset_reactivate_replays_arm_ramp(make_task: Any) -> None:
    factory, _pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)
    task.start()
    task.compute(_state(1.0))

    assert task.reset_runtime_state(reactivate=True)

    snapshot = task.state_snapshot()
    assert snapshot["control_state"] == "unarmed"
    assert snapshot["arm_pending"] is True


def test_dry_run_republishes_fixed_hold_instead_of_learned_targets(make_task: Any) -> None:
    factory, pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0, auto_dry_run=True)
    task.start()

    ramp_output = task.compute(_state(1.0))
    pipeline.step.return_value = np.ones(29, dtype=np.float32)
    policy_output = task.compute(_state(1.02))
    second_output = task.compute(_state(1.04))

    assert ramp_output is not None
    assert policy_output.positions == second_output.positions == ramp_output.positions
    assert policy_output.positions != [1.0] * 29
    assert pipeline.step.call_count == 2


def test_inference_exception_latches_damping_and_requires_restart(make_task):
    factory, pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)
    task.start()
    task.compute(_state(1.0))
    pipeline.step.side_effect = RuntimeError("decoder failed")

    assert task.compute(_state(1.02)) is None
    pipeline.step.side_effect = None
    assert task.compute(_state(1.04)) is None

    assert task.control_state is SonicControlState.FAULT
    assert task.state_snapshot()["fault_reason"] == "decoder failed"
    task._adapter.write_motor_commands.assert_called_with(damping_commands(29))
    assert pipeline.step.call_count == 1
    assert task.reset_runtime_state(reactivate=True) is False
    assert task.disarm() is False
    with pytest.raises(RuntimeError, match="restart"):
        task.arm()
    with pytest.raises(RuntimeError, match="restart"):
        task.set_estop(False)


@pytest.mark.parametrize("velocity", [-36.0, 36.0])
def test_measured_overspeed_trips_before_policy_inference(make_task, velocity):
    factory, pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)
    task.start()
    state = _state(1.0)
    state.joints.joint_velocities[_JOINT_NAMES[0]] = velocity

    assert task.compute(state) is None

    assert "joint overspeed" in task.fault_reason
    pipeline.step.assert_not_called()
    task._adapter.write_motor_commands.assert_called_once_with(damping_commands(29))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_invalid_policy_targets_latch_damping(make_task, value):
    factory, pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)
    task.start()
    task.compute(_state(1.0))
    pipeline.step.return_value = np.full(29, value, dtype=np.float32)

    assert task.compute(_state(1.02)) is None

    assert task.fault_reason == "invalid SONIC motor targets"
    task._adapter.write_motor_commands.assert_called_with(damping_commands(29))


def test_stop_arriving_during_inference_discards_completed_targets(make_task):
    factory, pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)
    task.start()
    task.compute(_state(1.0))

    def stopped_inference(**kwargs):
        task.set_estop(True)
        return np.ones(29, dtype=np.float32)

    pipeline.step.side_effect = stopped_inference

    assert task.compute(_state(1.02)) is None

    assert task.fault_reason == "operator stop"
    task._adapter.write_motor_commands.assert_called_with(damping_commands(29))


def test_hardware_fault_uses_robot_latch_instead_of_triggering_takeover(make_task, mocker):
    factory, _ = make_task
    task = factory(auto_arm=False, default_ramp_seconds=3.0)
    publish = mocker.Mock()
    task.set_fault_publisher(publish)
    task.start()

    task.set_estop(True)

    publish.assert_called_once_with("operator stop")
    task._adapter.write_motor_commands.assert_not_called()


@pytest.fixture
def coordinator():
    coordinator = ControlCoordinator(publish_joint_state=False)
    yield coordinator
    coordinator.stop()


def test_coordinator_estop_reaches_sonic_damping(make_task, coordinator):
    factory, pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)
    coordinator.add_task(task, task_type="g1_sonic_wbc")
    task.start()
    task.compute(_state(1.0))

    assert coordinator.set_estop(True)
    assert task.compute(_state(1.02)) is None

    assert task.fault_reason == "operator stop"
    task._adapter.write_motor_commands.assert_called_with(damping_commands(29))
    pipeline.step.assert_not_called()


def test_policy_timing_is_observational(make_task: Any) -> None:
    factory, _pipeline = make_task
    task = factory(auto_arm=True, default_ramp_seconds=0.0)

    task._record_policy_timing(0.201, 1.0)
    task._record_policy_timing(0.005, 1.02)

    assert task._policy_timing_snapshot() == {
        "step_ms": {"samples": 2, "mean": 103.0, "p95": 191.2, "p99": 199.04, "max": 201.0},
        "start_interval_ms": {
            "samples": 1,
            "mean": 20.0,
            "p95": 20.0,
            "p99": 20.0,
            "max": 20.0,
        },
    }


def test_task_factory_fails_fast_when_selected_model_bundle_is_missing(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        name="sonic",
        joint_names=_JOINT_NAMES,
        priority=50,
        params={
            "encoder_onnx": tmp_path / "low_latency/model_encoder.onnx",
            "decoder_onnx": tmp_path / "low_latency/model_decoder.onnx",
            "planner_onnx": tmp_path / "planner_sonic.onnx",
            "hardware_id": "g1",
            "sonic_pipeline": "sonic-low-latency",
        },
    )

    with pytest.raises(FileNotFoundError, match="setup-sonic-models"):
        _create_task(cfg, {}, G1SonicWBCTask)
