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

import subprocess
import sys

import pytest


@pytest.mark.self_hosted
@pytest.mark.parametrize("simulation", ["mujoco", ""])
def test_pico_blueprint_routes_native_inputs_to_existing_sonic_task(simulation: str) -> None:
    # Global config is consumed on blueprint import. Isolate each backend choice.
    code = """
import sys
from dimos.core.global_config import global_config
global_config.update(simulation=sys.argv[1], viewer="none")
from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.robot.get_all_blueprints import get_blueprint_by_name
from dimos.msgs.std_msgs.String import String
from dimos.teleop.webxr.controller_types import Buttons

blueprint = get_blueprint_by_name("unitree-g1-sonic-pico-teleop")
source = next(atom for atom in blueprint.blueprints if atom.module.__name__ == "PicoTeleopModule")
assert source.kwargs["linear_scale"] == 0.6
assert source.kwargs["linear_min_speed"] == 0.1
assert source.kwargs["yaw_scale"] == 1.5
assert source.kwargs["deadzone"] == 0.15
coordinator = next(atom for atom in blueprint.blueprints if atom.name == "ControlCoordinator")
outputs = {(stream.name, stream.type) for stream in source.streams if stream.direction == "out"}
inputs = {
    (blueprint.remapping_map.get((coordinator.name, stream.name), stream.name), stream.type)
    for stream in coordinator.streams if stream.direction == "in"
}
assert {name for name, _ in outputs} == {"body_tracking", "teleop_buttons", "cmd_vel"}
assert outputs <= inputs, (outputs, inputs)
assert not any("WebXR" in atom.module.__name__ or "ArmTeleop" in atom.module.__name__ for atom in blueprint.blueprints)
task = coordinator.kwargs["tasks"][0]
assert task.type == "g1_sonic_teleop"
assert task.name == "sonic_teleop"
assert task.params["auto_arm"] is bool(sys.argv[1])
assert task.params["auto_dry_run"] is (not bool(sys.argv[1]))
assert blueprint.global_config_overrides["transport"] == "zenoh"
backend = "MujocoSimModule" if sys.argv[1] else "G1WholeBodyConnection"
assert any(atom.module.__name__ == backend for atom in blueprint.blueprints)
if sys.argv[1]:
    camera = next(atom for atom in blueprint.blueprints if atom.module.__name__ == backend)
    video = next(atom for atom in blueprint.blueprints if atom.module.__name__ == "PicoVideoModule")
    assert camera.kwargs["enable_color"] is True
    assert camera.kwargs["camera_name"] == "head_color"
    camera_outputs = {(stream.name, stream.type) for stream in camera.streams if stream.direction == "out"}
    video_inputs = {(stream.name, stream.type) for stream in video.streams if stream.direction == "in"}
    assert video_inputs <= camera_outputs
else:
    assert not any(atom.module.__name__ == "PicoVideoModule" for atom in blueprint.blueprints)
    connection = next(atom for atom in blueprint.blueprints if atom.module.__name__ == backend)
    assert connection.kwargs["command_timeout_seconds"] == 0.1
    connection_inputs = {(stream.name, stream.type) for stream in connection.streams if stream.direction == "in"}
    connection_outputs = {(stream.name, stream.type) for stream in connection.streams if stream.direction == "out"}
    coordinator_outputs = {(stream.name, stream.type) for stream in coordinator.streams if stream.direction == "out"}
    assert ("teleop_buttons", Buttons) in outputs & connection_inputs
    assert ("sonic_fault", String) in coordinator_outputs & connection_inputs
    assert ("g1_fault", String) in connection_outputs & inputs
parsed = BlueprintConfigParser(blueprint).parse(cli_tokens=[
    "--device-id", "chosen-pico", "--sonic-pipeline", "sonic-low-latency",
    "--manage-pc-service", "false", "--pc-service-dir", "/tmp/pico-service",
])
assert parsed.module_kwargs(source.name)["device_id"] == "chosen-pico"
assert parsed.module_kwargs(source.name)["manage_pc_service"] is False
assert str(parsed.module_kwargs(source.name)["pc_service_dir"]) == "/tmp/pico-service"
assert parsed.module_kwargs(coordinator.name)["sonic_pipeline"] == "sonic-low-latency"
"""
    subprocess.run([sys.executable, "-c", code, simulation], check=True, timeout=30)
