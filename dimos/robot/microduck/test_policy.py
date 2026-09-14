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
from pathlib import Path
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
from pytest_mock import MockerFixture

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.protocol.rpc.zenohrpc import ZenohRPC
from dimos.robot.microduck.blueprints import microduck_sim
from dimos.robot.microduck.policy import MicroduckPolicy
from dimos.robot.microduck.simulation import MicroduckSim
from dimos.simulation.engines.mujoco_engine import MujocoEngine
from dimos.simulation.engines.robot_sim_binding import resolve_robot_sim_binding
from dimos.simulation.utils.xml_parser import build_joint_mappings


@pytest.fixture
def session(mocker: MockerFixture) -> MagicMock:
    session = mocker.patch("dimos.robot.microduck.policy.ort.InferenceSession").return_value
    session.get_modelmeta.return_value.custom_metadata_map = {
        "joint_names": ",".join(f"joint{i}" for i in range(14)),
        "default_joint_pos": ",".join(["0.1"] * 14),
        "action_scale": "0.5",
        "observation_names": "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command",
    }
    session.get_inputs.return_value = [mocker.Mock(name="input", shape=[1, 61])]
    session.get_inputs.return_value[0].name = "obs"
    session.get_outputs.return_value = [mocker.Mock(name="output", shape=[1, 14])]
    session.get_outputs.return_value[0].name = "actions"
    session.run.return_value = [np.full((1, 14), 0.2, dtype=np.float32)]
    return session


@pytest.fixture
def model() -> mujoco.MjModel:
    # An unrelated free body precedes the robot; actuators reverse joint order.
    # This catches indexing by global qpos offsets or actuator order.
    joints = "".join(
        f'<body name="link{i}" pos="{i * 0.1} 0 0">'
        f'<joint name="joint{i}" type="hinge"/>'
        '<geom type="sphere" size="0.01" mass="0.1"/></body>'
        for i in range(14)
    )
    actuators = "".join(
        f'<position name="joint{i}" joint="joint{i}" kp="1"/>' for i in reversed(range(14))
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="prop"><freejoint/>'
        '<geom type="sphere" size="0.1" mass="1"/></body>'
        '<body name="trunk_base"><freejoint name="trunk_base_freejoint"/>'
        '<geom type="sphere" size="0.1" mass="1"/><site name="imu"/>'
        f"{joints}</body></worldbody><actuator>{actuators}</actuator>"
        '<sensor><gyro name="imu_ang_vel" site="imu"/>'
        '<accelerometer name="imu_accel" site="imu"/></sensor></mujoco>'
    )


@pytest.fixture
def sim_module(mocker: MockerFixture) -> Iterator[MicroduckSim]:
    mocker.patch.object(ZenohRPC, "start")
    mocker.patch.object(ZenohRPC, "serve_module_rpc")
    module = MicroduckSim(rpc_transport=ZenohRPC, headless=True)
    yield module
    module.stop()


@pytest.fixture
def engine(session: MagicMock, model: mujoco.MjModel, tmp_path: Path) -> Iterator[MujocoEngine]:
    path = tmp_path / "scene.xml"
    path.write_text("<mujoco/>")
    policy = MicroduckPolicy(Path("unused.onnx"))
    engine = MujocoEngine(
        config_path=path, headless=True, model=model, robot_sim_spec=policy.sim_spec()
    )
    yield engine
    engine.disconnect()


def test_policy_maps_named_joints_and_carries_action_history(
    session: MagicMock, model: mujoco.MjModel
) -> None:
    policy = MicroduckPolicy(Path("unused.onnx"))
    binding = resolve_robot_sim_binding(model, policy.sim_spec(), build_joint_mappings(None, model))
    data = mujoco.MjData(model)
    data.qpos[list(binding.joint_qpos_adrs)] = 0.3
    data.qvel[list(binding.joint_qvel_adrs)] = np.arange(14)
    mujoco.mj_forward(model, data)
    data.sensordata[binding.imu_gyro_slice] = [1, 2, 3]

    targets = policy.targets(data, binding, (0.3, -0.1, 0.5))

    np.testing.assert_allclose(targets, np.full(14, 0.2))
    expected = np.concatenate(
        (
            [1, 2, 3, 0, 0, -1],
            np.full(14, 0.2),
            np.arange(14),
            np.zeros(14),
            [0.3, -0.1, 0.5],
            np.zeros(10),
        )
    )[None, :]
    np.testing.assert_allclose(session.run.call_args.args[1]["obs"], expected, atol=1e-7)

    policy.targets(data, binding, (0, 0, 0))
    np.testing.assert_allclose(session.run.call_args.args[1]["obs"][0, 34:48], np.full(14, 0.2))
    policy.reset()
    policy.targets(data, binding, (0, 0, 0))
    np.testing.assert_array_equal(session.run.call_args.args[1]["obs"][0, 34:48], np.zeros(14))


def test_policy_rejects_nonfinite_actions(session: MagicMock, model: mujoco.MjModel) -> None:
    policy = MicroduckPolicy(Path("unused.onnx"))
    binding = resolve_robot_sim_binding(model, policy.sim_spec(), build_joint_mappings(None, model))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    session.run.return_value = [np.full((1, 14), np.nan)]

    with pytest.raises(ValueError, match="invalid joint actions"):
        policy.targets(data, binding, (0, 0, 0))


@pytest.mark.parametrize("shape", [[1, 60], [1, 62], [None, 61]])
def test_policy_rejects_incompatible_observations(
    session: MagicMock, shape: list[int | None]
) -> None:
    session.get_inputs.return_value[0].shape = shape

    with pytest.raises(ValueError, match="Expected Microduck policy input"):
        MicroduckPolicy(Path("unused.onnx"))


def test_blueprint_parses_local_assets_without_downloading(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    download = mocker.patch("dimos.robot.assets.source.RobotDescriptionSource.checkout_path")
    parsed = BlueprintConfigParser(microduck_sim).parse(
        [
            f"--microducksim.scene-path={tmp_path / 'scene.xml'}",
            f"--microducksim.policy-path={tmp_path / 'walk.onnx'}",
            "--microducksim.headless=true",
        ],
        environ={},
    )

    config = parsed.module_kwargs("microducksim")
    assert config["scene_path"] == tmp_path / "scene.xml"
    assert config["policy_path"] == tmp_path / "walk.onnx"
    assert config["headless"] is True
    assert parsed.global_config["simulation"] == "mujoco"
    download.assert_not_called()


def test_odometry_preserves_engine_quaternion_order(
    sim_module: MicroduckSim, engine: MujocoEngine, mocker: MockerFixture
) -> None:
    publish = mocker.patch.object(sim_module.odom, "publish")
    binding = engine.robot_binding
    assert binding is not None and binding.root_qpos_adr is not None
    # 90 degrees about z; MuJoCo qpos uses wxyz, engine pose uses xyzw.
    quaternion = np.array([1, 0, 0, 1]) / np.sqrt(2)
    address = binding.root_qpos_adr + 3
    engine.data.qpos[address : address + 4] = quaternion

    sim_module._publish_state(engine)

    orientation = publish.call_args.args[0].orientation
    np.testing.assert_allclose(
        [orientation.w, orientation.x, orientation.y, orientation.z], quaternion
    )
    np.testing.assert_allclose(sim_module.get_state()["orientation_wxyz"], quaternion)


@pytest.mark.parametrize("duration, expires_at", [(0.0, 100.5), (2.0, 102.0)])
def test_velocity_command_expires_without_further_messages(
    sim_module: MicroduckSim,
    engine: MujocoEngine,
    session: MagicMock,
    mocker: MockerFixture,
    duration: float,
    expires_at: float,
) -> None:
    clock = mocker.patch("dimos.robot.microduck.simulation.time.monotonic", return_value=100.0)
    policy = MicroduckPolicy(Path("unused.onnx"))
    sim_module.move(Twist(linear=[0.3, -0.1, 0], angular=[0, 0, 0.5]), duration=duration)

    sim_module._control(policy, engine)
    np.testing.assert_allclose(session.run.call_args.args[1]["obs"][0, 48:51], [0.3, -0.1, 0.5])

    clock.return_value = expires_at
    engine.data.time = 0.02
    sim_module._control(policy, engine)
    np.testing.assert_array_equal(session.run.call_args.args[1]["obs"][0, 48:51], np.zeros(3))
