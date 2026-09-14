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

from collections.abc import Iterator
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from dimos.control.tasks.g1_sonic_wbc_task.sonic_pipeline import (
    DEFAULT_ANGLES_DDS,
    DEFAULT_ANGLES_ONNX,
    DEFAULT_HEIGHT,
    NUM_JOINTS,
    SonicPipeline,
)


@pytest.fixture
def velocity_pipeline(mocker: Any) -> Iterator[SonicPipeline]:
    encoder = mocker.MagicMock()
    encoder.get_inputs.return_value = [SimpleNamespace(name="encoder", shape=[1, 1751])]
    encoder.get_providers.return_value = ["CUDAExecutionProvider"]
    encoder.run.return_value = [np.zeros((1, 64), dtype=np.float32)]
    decoder = mocker.MagicMock()
    decoder.get_inputs.return_value = [SimpleNamespace(name="decoder", shape=[1, 994])]
    decoder.get_providers.return_value = ["CUDAExecutionProvider"]
    decoder.run.return_value = [np.zeros((1, NUM_JOINTS), dtype=np.float32)]
    planner = mocker.MagicMock()
    planner.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    qpos = np.tile(
        np.concatenate(([0.0, 0.0, DEFAULT_HEIGHT, 1.0, 0.0, 0.0, 0.0], DEFAULT_ANGLES_ONNX)),
        (1, 44, 1),
    ).astype(np.float32)
    planner.run.return_value = [qpos, np.array([44])]
    mocker.patch.object(ort, "InferenceSession", side_effect=[encoder, decoder, planner])
    mocker.patch.object(ort, "preload_dlls")
    mocker.patch.object(
        ort,
        "get_available_providers",
        return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    pipeline = SonicPipeline("encoder.onnx", "decoder.onnx", "planner.onnx")
    try:
        yield pipeline
    finally:
        pipeline._planner_executor.shutdown(wait=True)


@pytest.fixture
def planner_requests(velocity_pipeline: SonicPipeline, mocker: Any) -> Any:
    completed: Future[list[Any]] = Future()
    completed.set_result(velocity_pipeline._planner.run.return_value)
    return mocker.patch.object(
        velocity_pipeline._planner_executor, "submit", return_value=completed
    )


def _step(pipeline: SonicPipeline) -> None:
    pipeline.step(
        q_dds=DEFAULT_ANGLES_DDS,
        dq_dds=np.zeros(NUM_JOINTS, dtype=np.float32),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        gyro_body=np.zeros(3, dtype=np.float32),
        gravity_body=np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("forward", "yaw_rate"),
    [(0.3, 0.0), (0.0, 0.3), (0.0, -0.3), (0.0, 0.06), (0.0, -0.06), (0.3, 0.3)],
)
def test_held_velocity_keeps_refreshing_planner(
    velocity_pipeline: SonicPipeline, planner_requests: Any, forward: float, yaw_rate: float
) -> None:
    for _ in range(250):
        velocity_pipeline.set_velocity(forward, 0.0, yaw_rate)
        _step(velocity_pipeline)

    assert planner_requests.call_count == 5


@pytest.mark.parametrize("yaw_rate", [0.06, -0.06])
def test_small_turn_starts_and_stops_without_waiting_for_periodic_replan(
    velocity_pipeline: SonicPipeline, planner_requests: Any, yaw_rate: float
) -> None:
    _step(velocity_pipeline)
    planner_requests.reset_mock()

    velocity_pipeline.set_velocity(0.0, 0.0, yaw_rate)
    _step(velocity_pipeline)

    assert planner_requests.call_count == 1
    np.testing.assert_allclose(
        planner_requests.call_args.args[2]["facing_direction"],
        [[np.cos(yaw_rate), np.sin(yaw_rate), 0.0]],
        atol=1e-7,
    )

    velocity_pipeline.set_velocity(0.0, 0.0, 0.0)
    _step(velocity_pipeline)

    assert planner_requests.call_count == 2
    np.testing.assert_array_equal(
        planner_requests.call_args.args[2]["facing_direction"], [[1.0, 0.0, 0.0]]
    )
    for _ in range(100):
        _step(velocity_pipeline)
    assert planner_requests.call_count == 2


def test_stop_received_during_inference_is_planned_when_worker_is_available(
    velocity_pipeline: SonicPipeline, planner_requests: Any
) -> None:
    pending: Future[list[Any]] = Future()
    completed = planner_requests.return_value
    planner_requests.side_effect = [pending, completed]
    velocity_pipeline.set_velocity(0.0, 0.0, 0.3)
    _step(velocity_pipeline)

    velocity_pipeline.set_velocity(0.0, 0.0, 0.0)
    _step(velocity_pipeline)
    assert planner_requests.call_count == 1

    pending.set_result(completed.result())
    _step(velocity_pipeline)

    assert planner_requests.call_count == 2
    np.testing.assert_array_equal(
        planner_requests.call_args.args[2]["facing_direction"], [[1.0, 0.0, 0.0]]
    )
