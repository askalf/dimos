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

import time

import pytest

from dimos.e2e_tests.dimos_cli_call import DimosCliCall
from dimos.evals.runner import EvalRunner
from dimos.evals.sim2 import contained, fresh_states, inside_region, touching
from dimos.evals.types import InteractiveEval
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.porcelain.dimos import Dimos
from dimos.sim2.interaction import reset_scene
from dimos.sim2.scene_types import EntityState, RegionState, SceneState


@pytest.fixture
def initial():
    return SceneState(
        world_id="owned",
        scene_id="kitchen",
        generation=2,
        tick=0,
        sim_time=0,
        ts=time.time() - 1,
        entities={},
        robots={},
        joints={},
        regions={},
        contacts=(),
    )


@pytest.fixture
def store(tmp_path):
    store = SqliteStore(path=str(tmp_path / "recording.db"))
    try:
        yield store
    finally:
        store.stop()


def test_physical_recording_roundtrip_and_episode_freshness(store, initial):
    stream = store.stream("sim_truth", SceneState, codec="pickle")
    for index in range(11):
        state = initial.model_copy(
            update={"tick": index + 1, "sim_time": index * 0.1, "ts": initial.ts + index * 0.1}
        )
        stream.append(state, ts=state.ts)
    assert len(fresh_states(store, initial, dwell_s=0.5)) >= 6
    with pytest.raises(RuntimeError, match="different world"):
        fresh_states(store, initial.model_copy(update={"world_id": "old"}))
    with pytest.raises(RuntimeError, match="different world"):
        fresh_states(store, initial.model_copy(update={"generation": 1}))


def test_empty_or_stale_truth_cannot_pass(store, initial):
    stream = store.stream("sim_truth", SceneState, codec="pickle")
    with pytest.raises(LookupError):
        fresh_states(store, initial)
    stale = initial.model_copy(update={"ts": time.time() - 10})
    stream.append(stale, ts=stale.ts)
    with pytest.raises(RuntimeError, match="stale"):
        fresh_states(store, initial)


def test_containment_requires_whole_object_not_only_center(initial):
    region = RegionState(pose=Pose(1, 0, 0), size=(1, 1, 1))
    entity = EntityState(
        pose=Pose(1, 0, 0),
        velocity=(0, 0, 0),
        angular_velocity=(0, 0, 0),
        bounds_min=(0.8, -0.2, -0.2),
        bounds_max=(1.2, 0.2, 0.2),
    )
    assert inside_region((1, 0, 0), region)
    assert contained(entity, region)
    assert not contained(entity.model_copy(update={"bounds_max": (1.8, 0.2, 0.2)}), region)
    contact = initial.model_copy(update={"contacts": (("block", "arm/left_finger"),)})
    assert touching(contact, "block", "arm/left_finger")
    assert not touching(contact, "block", "arm/right_finger")


def test_scripted_eval_uses_application_and_sim2_launch_not_mcp(mocker, tmp_path, initial):
    process = mocker.patch.object(DimosCliCall, "start")
    stop = mocker.patch.object(DimosCliCall, "stop")
    app = mocker.Mock()
    app.get_module.return_value.scene_state.return_value = initial
    app.get_module.return_value.status.return_value = {"running": True, "error": None}
    app.get_module.return_value.recording_ready.return_value = True
    mocker.patch.object(Dimos, "connect", return_value=app)
    mcp = mocker.patch.object(EvalRunner, "_wait_mcp")
    setup, action = mocker.Mock(), mocker.Mock()
    case = InteractiveEval(
        id="scripted",
        inputs="test",
        simulator="mujoco",
        scene="kitchen",
        blueprint="xarm7-planner-coordinator",
        setup=setup,
        action=action,
        score=lambda _: 1.0,
    )
    runner = EvalRunner(out_dir=tmp_path)
    sample = mocker.patch.object(runner, "sample", return_value=[(0.0, 1.0)])
    try:
        result = runner.run([case])[0]
    finally:
        runner.stop()
    assert result.passed
    setup.assert_called_once_with(app)
    action.assert_called_once_with(app)
    mcp.assert_not_called()
    process.assert_called_once()
    stop.assert_called_once()
    sample.assert_called_once()
    assert list(tmp_path.glob("*/scripted.setup.json"))


def test_reset_failure_leaves_physics_paused(mocker, initial):
    app = mocker.Mock()
    sim, coordinator = mocker.Mock(), mocker.Mock()
    coordinator.get_tick_count.return_value = 1
    sim.reset.return_value = initial
    coordinator.reset_runtime_state.return_value = {"balance": False}
    app.get_module.side_effect = lambda name: sim if name == "SimulationModule" else coordinator
    with pytest.raises(RuntimeError, match="world remains paused"):
        reset_scene(app)
    sim.set_paused.assert_called_once_with(True)
    coordinator.set_activated.assert_called_once_with(False)


def test_action_and_skill_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        InteractiveEval(id="bad", inputs="", skill="move", action=lambda _: None, score=lambda _: 1)


def test_cleanup_failure_preserves_case_error_and_closes_every_resource(mocker, tmp_path):
    runner = EvalRunner(out_dir=tmp_path)
    runner._run_dir = tmp_path
    app, sim, process = mocker.Mock(), mocker.Mock(), mocker.Mock()
    runner._app, runner._sim, runner._proc = app, sim, process
    app.stop.side_effect = RuntimeError("disconnect failed")
    case = mocker.Mock(id="failed-action")
    case.evaluate.side_effect = ValueError("setup failed")
    result = runner._guarded(case)
    assert not result.passed
    assert "setup failed" in result.error and "disconnect failed" in result.error
    app.stop.assert_called_once()
    sim.stop.assert_called_once()
    process.stop.assert_called_once()
    runner.stop()
    process.stop.assert_called_once()
