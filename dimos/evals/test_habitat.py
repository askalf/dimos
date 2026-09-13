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

# Copyright 2026 Dimensional Inc.
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dimos.evals.environments.habitat_spec import HabitatEvalBackend, HabitatEvalConfig
from dimos.evals.environments.sim import Sim
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry


@pytest.mark.parametrize("missing", ["color_image", "habitat_scan", "odometry", None])
def test_smoke_checks_samples_without_health_artifact(mocker, missing):
    from dimos.evals.suites.habitat_smoke import sensor_score

    store = mocker.patch(
        "dimos.evals.suites.habitat_smoke.recording"
    ).return_value.__enter__.return_value
    store.streams.__contains__.side_effect = lambda name: name != missing
    assert sensor_score(SimpleNamespace(artifacts={})) == (1.0 if missing is None else 0.0)
    if missing is None:
        store.streams.habitat_scan.last.assert_called_once()


def test_scene_config_reaches_blueprint_parser(tmp_path):
    from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
    from dimos.simulation.habitat.blueprints import habitat_nav

    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}")
    backend = HabitatEvalBackend(
        HabitatEvalConfig(
            scene_dataset_config=str(dataset),
            scene_id="apt_1",
            seed=42,
            start_position_ros=(1, 2, 3),
            start_yaw_deg=15,
        )
    )
    backend.preflight()
    launch = backend.launch_spec()
    parsed = BlueprintConfigParser(habitat_nav).parse(environ=dict(launch.environment))
    config = parsed.module_kwargs("habitatconnection")
    assert config["scene_id"] == "apt_1"
    assert config["scene_dataset_config"] == str(dataset)
    assert config["start_position_ros"] == (1, 2, 3)
    assert config["seed"] == 42
    assert config["publish_semantic"] is False
    assert parsed.global_config["transport"] == "zenoh"
    assert launch.simulation_flag is None


def test_invalid_spawn_and_missing_scene(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        HabitatEvalConfig(start_position_ros=(float("nan"), 0, 0))
    with pytest.raises(FileNotFoundError):
        HabitatEvalBackend(
            HabitatEvalConfig(scene_dataset_config=str(tmp_path / "missing"))
        ).preflight()


def test_backend_does_not_inject_skill_or_navigation_modules():
    launch = HabitatEvalBackend(HabitatEvalConfig()).launch_spec()
    assert launch.blueprint == ()
    assert "habitat_planned_path" not in launch.global_args[1]


def test_habitat_rejects_dimsim_setup_and_attach():
    with pytest.raises(ValueError, match="fresh launches"):
        Sim(simulator="habitat", attach=True)
    with pytest.raises(ValueError, match="fresh launches"):
        Sim(simulator="habitat", setup=lambda client: None)
    with pytest.raises(ValueError, match="requires"):
        Sim(habitat=HabitatEvalConfig())
    env = Sim(simulator="habitat", scene="apt_1")
    assert env._backend.config.scene_id == "apt_1"
    with pytest.raises(ValueError, match="must agree"):
        Sim(simulator="habitat", scene="apt_1", habitat=HabitatEvalConfig(scene_id="apt_2"))


def test_habitat_launch_and_cleanup(tmp_path, mocker):
    proc = mocker.patch("dimos.evals.environments.sim.DimosCliCall").return_value
    proc.extra_env = {}
    mocker.patch(
        "dimos.evals.environments.sim.McpAdapter"
    ).return_value.wait_for_ready.return_value = True
    store = mocker.patch("dimos.memory.store.sqlite.SqliteStore").return_value
    client = mocker.patch("dimos.evals.environments.sim.DimSimClient")
    env = Sim(
        simulator="habitat",
        habitat=HabitatEvalConfig(scene_id="apt_1"),
        blueprint=["habitat-nav", "mcp-server", "observe-skill"],
    )
    mocker.patch.object(env, "_wait_recording", return_value=tmp_path / "memory.db")
    ready = mocker.patch.object(env._backend, "wait_ready")
    try:
        result = env.start(("speak-skill",))
        assert proc.simulator is None
        assert proc.global_args[0] == "--record-topics"
        assert proc.global_args[-1] == "--record"
        from dimos.memory.tap import matching

        selected = matching(
            proc.global_args[1],
            (
                "depth_image",
                "semantic_image",
                "node_edges",
                "color_image",
                "habitat_scan",
                "odometry",
                "tf",
                "path",
                "cmd_vel",
            ),
        )
        assert selected == {"color_image", "habitat_scan", "odometry", "tf", "path", "cmd_vel"}
        assert proc.demo_args == [
            "run",
            "habitat-nav",
            "mcp-server",
            "observe-skill",
            "speak-skill",
        ]
        assert proc.extra_env["DIMOS_TRANSPORT"] == "zenoh"
        ready.assert_called_once()
        assert result.artifacts["episode"].is_file()
        client.assert_not_called()
    finally:
        env.stop()
    proc.stop.assert_called_once()
    store.stop.assert_called_once()


def test_readiness_failure_releases_resources(tmp_path, mocker):
    proc = mocker.patch("dimos.evals.environments.sim.DimosCliCall").return_value
    mocker.patch(
        "dimos.evals.environments.sim.McpAdapter"
    ).return_value.wait_for_ready.return_value = True
    store = mocker.patch("dimos.memory.store.sqlite.SqliteStore").return_value
    env = Sim(simulator="habitat")
    mocker.patch.object(env, "_wait_recording", return_value=tmp_path / "memory.db")
    mocker.patch.object(env._backend, "wait_ready", side_effect=TimeoutError("no RGB"))
    try:
        with pytest.raises(TimeoutError, match="no RGB"):
            env.start(())
    finally:
        env.stop()
        env.stop()
    proc.stop.assert_called_once()
    store.stop.assert_called_once()


def test_recorded_odometry_normalization():
    odom = Odometry(ts=123, frame_id="world", pose=Pose(position=Vector3(1, 2, 3)))
    stream = Mock()
    stream.last.return_value = SimpleNamespace(data=odom)
    streams = Mock()
    streams.__contains__ = Mock(return_value=True)
    streams.odometry = stream
    pose = HabitatEvalBackend(HabitatEvalConfig()).latest_pose(SimpleNamespace(streams=streams))
    assert pose.ts == 123
    assert pose.frame_id == "world"
    assert tuple(pose.position) == (1, 2, 3)


def test_readiness_requires_fresh_frames_without_prescribing_tools(mocker):
    import numpy as np

    from dimos.memory.store.memory import MemoryStore
    from dimos.msgs.sensor_msgs.Image import Image

    adapter = mocker.patch("dimos.agents.mcp.mcp_adapter.McpAdapter").return_value
    backend = HabitatEvalBackend(HabitatEvalConfig())
    with MemoryStore() as store:
        with pytest.raises(TimeoutError):
            backend.wait_ready(store, deadline=time.monotonic())
        store.stream("odometry", Odometry).append(Odometry(frame_id="world"))
        store.stream("color_image", Image).append(Image(np.zeros((2, 2, 3), dtype=np.uint8)))
        backend.wait_ready(store, deadline=time.monotonic() + 1)
        assert backend.episode_metadata()["initial_observed_position_ros"] == [0, 0, 0]
        adapter.list_tools.assert_not_called()


def test_explicit_spawn_converts_ros_and_rejects_non_navigable():
    import numpy as np

    from dimos.simulation.habitat.server import HabitatHost

    host = object.__new__(HabitatHost)
    host.cfg = {"seed": 4, "start_position_ros": (1, 2, 3)}
    host._sim = Mock()
    host._agent = Mock()
    host.hs = Mock()
    host._sim.pathfinder.is_navigable.return_value = True
    host.reset_pose()
    state = host._agent.set_state.call_args.args[0]
    np.testing.assert_allclose(state.position, [-2, 3, -1])
    host._sim.pathfinder.get_random_navigable_point.assert_not_called()
    host._sim.pathfinder.is_navigable.return_value = False
    with pytest.raises(ValueError, match="not navigable"):
        host.reset_pose()
