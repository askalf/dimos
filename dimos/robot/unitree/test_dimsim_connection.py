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

from contextlib import ExitStack
from functools import partial
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.transport import LCMTransport
from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.robot.unitree import dimsim_connection
from dimos.robot.unitree.dimsim_connection import DimSimConnection


@pytest.fixture
def simulator(mocker):
    return mocker.patch.object(dimsim_connection, "DimSimProcess").return_value


@pytest.mark.parametrize("backend", ["lcm", "zenoh"])
def test_wire_sensors_commands_and_restart(backend, simulator, monkeypatch, unused_udp_port):
    """Exercise actual LCM/Zenoh serialization; only browser startup is stubbed."""
    config = GlobalConfig(transport=backend, robot_ip=None, robot_ips=None, zenoh_connect="")
    for field in ("transport", "robot_ip", "robot_ips", "zenoh_connect"):
        monkeypatch.setattr(global_config, field, getattr(config, field))
    wire = partial(LCMTransport, url=f"udpm://239.255.77.55:{unused_udp_port}?ttl=0")
    monkeypatch.setattr(dimsim_connection, "LCMTransport", wire)

    def output(topic, msg_type, *, g):
        return (
            wire(topic, msg_type) if g.transport == "lcm" else make_transport(topic, msg_type, g=g)
        )

    monkeypatch.setattr(dimsim_connection, "make_transport", output)
    connection = DimSimConnection(config)
    with ExitStack() as stack:
        stack.callback(connection.stop)
        topics = {
            "/color_image": Image,
            "/lidar": PointCloud2,
            "/camera_info": CameraInfo,
            "/odom": PoseStamped,
            "/tf": TFMessage,
        }
        received = {topic: [] for topic in topics}
        events = {topic: threading.Event() for topic in topics}
        sources = {}
        for topic, msg_type in topics.items():
            source = wire(topic, msg_type)
            target = output(topic, msg_type, g=config)
            sources[topic] = source
            for transport in (source, target):
                transport.start()
                stack.callback(transport.stop)

            def collect(msg, topic=topic):
                received[topic].append(msg)
                events[topic].set()

            stack.callback(target.subscribe(collect))

        # Same wiring GO2Connection uses: active cmd_vel -> connection.move.
        commands = output("/cmd_vel", Twist, g=config)
        commands.start()
        stack.callback(commands.stop)
        stack.callback(commands.subscribe(connection.move))
        wire_commands = wire("/cmd_vel", Twist)
        wire_commands.start()
        stack.callback(wire_commands.stop)
        command_event = threading.Event()
        command_values = []

        def simulate_motion(msg):
            command_values.append(msg.linear.x)
            sources["/odom"].publish(
                PoseStamped(ts=42.0, frame_id="world", position=[msg.linear.x, 2, 0.5])
            )
            command_event.set()

        stack.callback(wire_commands.subscribe(simulate_motion))
        for cycle in range(2):
            connection.start()
            connection.start()  # idempotent: don't install a second relay
            assert simulator.start.call_count == cycle + 1
            for event in events.values():
                event.clear()
            command_event.clear()
            messages = {
                "/color_image": Image.from_numpy(
                    np.full((8, 8, 3), 30 + cycle, dtype=np.uint8),
                    ts=42.0,
                    frame_id="camera_optical",
                ),
                "/lidar": PointCloud2.from_numpy(
                    np.array([[1, 2, 3]], dtype=np.float32), timestamp=42.0, frame_id="world"
                ),
                "/camera_info": connection.camera_info_static,
            }
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not all(e.is_set() for e in events.values()):
                for topic, msg in messages.items():
                    sources[topic].publish(msg)
                commands.publish(Twist(linear=[0.3 + cycle, 0, 0]))
                command_event.wait(0.05)
                events["/tf"].wait(0.05)
            assert all(e.is_set() for e in events.values()), {
                k: len(v) for k, v in received.items()
            }
            assert received["/odom"][-1].position.x == pytest.approx(0.3 + cycle)
            assert received["/odom"][-1].frame_id == "world"
            assert received["/odom"][-1].ts == 42.0
            np.testing.assert_array_equal(received["/lidar"][-1].points_f32(), [[1, 2, 3]])
            assert received["/lidar"][-1].frame_id == "world"
            assert received["/color_image"][-1].frame_id == "camera_optical"
            np.testing.assert_array_equal(
                received["/color_image"][-1].as_numpy(), messages["/color_image"].as_numpy()
            )
            assert received["/camera_info"][-1].width == connection.camera_info_static.width
            assert {t.child_frame_id for t in received["/tf"][-1].transforms} == {
                "base_link",
                "camera_link",
                "camera_optical",
                "lidar_link",
            }
            connection.stop()
            connection.stop()
            assert simulator.stop.call_count == cycle + 1
            if backend == "zenoh":
                assert connection.move(Twist(linear=[1, 0, 0])) is False
                events["/lidar"].clear()
                sources["/lidar"].publish(messages["/lidar"])
                assert not events["/lidar"].wait(0.1)


def test_lcm_does_not_republish_commands(simulator, mocker):
    wire = mocker.patch.object(dimsim_connection, "LCMTransport")
    mocker.patch.object(dimsim_connection, "make_transport")
    connection = DimSimConnection(GlobalConfig(transport="lcm"))
    connection.start()
    try:
        connection.move(Twist(linear=[0.5, 0, 0]))
        connection.stop_movement()
        assert connection._relays == []
        wire.return_value.publish.assert_not_called()
    finally:
        connection.stop()


def test_failed_start_releases_transports_and_subscriptions(simulator, mocker):
    transports = []

    def transport(*args, **kwargs):
        instance = MagicMock()
        transports.append(instance)
        return instance

    mocker.patch.object(dimsim_connection, "LCMTransport", side_effect=transport)
    mocker.patch.object(dimsim_connection, "make_transport", side_effect=transport)
    simulator.start.side_effect = RuntimeError("browser startup failed")
    connection = DimSimConnection(GlobalConfig(transport="zenoh"))
    with pytest.raises(RuntimeError, match="browser startup failed"):
        connection.start()
    for instance in transports:
        instance.stop.assert_called_once()
        if instance.subscribe.called:
            instance.subscribe.return_value.assert_called_once_with()
    assert not connection.move(Twist())
    simulator.stop.assert_called_once()
    connection.stop()
    simulator.stop.assert_called_once()
