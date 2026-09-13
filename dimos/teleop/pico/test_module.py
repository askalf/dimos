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

from collections.abc import Callable, Iterator
from queue import Queue
import time
from typing import TypeVar

import pytest

from dimos.core.global_config import GlobalConfig
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.teleop.pico.module import PicoTeleopModule
from dimos.teleop.pico.proto.tracking_pb2 import DeviceStateJson, ServerFeedback
from dimos.teleop.webxr.body_tracking import BodyTrackingSnapshot
from dimos.teleop.webxr.controller_types import Buttons

T = TypeVar("T")


def take(queue: Queue[T], predicate: Callable[[T], bool]) -> T:
    deadline = time.monotonic() + 3.0
    while True:
        item = queue.get(timeout=max(0.001, deadline - time.monotonic()))
        if predicate(item):
            return item
        assert time.monotonic() < deadline, "expected output did not arrive"


@pytest.fixture
def source(
    pc_service, mocker
) -> Iterator[tuple[PicoTeleopModule, Queue[BodyTrackingSnapshot], Queue[Buttons], Queue[Twist]]]:
    module = PicoTeleopModule(
        g=GlobalConfig(xrobotoolkit_host="127.0.0.1", xrobotoolkit_port=pc_service.port),
        stale_timeout=0.2,
        reconnect_interval=0.02,
    )
    body: Queue[BodyTrackingSnapshot] = Queue()
    buttons: Queue[Buttons] = Queue()
    velocity: Queue[Twist] = Queue()
    mocker.patch.object(module.body_tracking, "publish", side_effect=body.put)
    mocker.patch.object(module.teleop_buttons, "publish", side_effect=buttons.put)
    mocker.patch.object(module.cmd_vel, "publish", side_effect=velocity.put)
    try:
        module.start()
        yield module, body, buttons, velocity
    finally:
        module.stop()


def test_live_packets_publish_then_timeout_clears_all_outputs(source, pc_service, feedback_factory):
    module, body, buttons, velocity = source
    _, outgoing = pc_service.subscriptions.get(timeout=2)
    outgoing.put(feedback_factory(0))
    outgoing.put(feedback_factory(1))
    snapshot = take(body, lambda msg: msg.joints is not None)
    assert len(snapshot.joints) == 24
    assert take(velocity, lambda msg: msg.linear.x > 0).linear.x == 0.3
    assert module.tracking_status()["pc_service_connected"] is True
    take(body, lambda msg: msg.joints is None)
    assert module.tracking_status()["body_available"] is False
    assert take(velocity, lambda msg: msg.linear.x == 0).angular.z == 0
    assert buttons.get(timeout=2).data == 0


def test_disconnect_reconnect_changes_frame_and_requires_ax_release(
    source, pc_service, feedback_factory
):
    module, body, _, _ = source
    _, outgoing = pc_service.subscriptions.get(timeout=2)
    outgoing.put(feedback_factory(0))
    outgoing.put(feedback_factory(1))
    first = take(body, lambda msg: msg.joints is not None)
    outgoing.put(None)
    take(body, lambda msg: msg.joints is None)
    _, reconnected = pc_service.subscriptions.get(timeout=2)
    reconnected.put(feedback_factory(2, pressed=True))
    reconnected.put(feedback_factory(3, pressed=True))
    second = take(body, lambda msg: msg.joints is not None)
    assert first.frame_id != second.frame_id
    assert module.tracking_status()["release_required"] is True


@pytest.mark.parametrize(
    "event",
    [
        ServerFeedback(name="deviceMissing", devid="pico-1"),
        ServerFeedback(
            name="deviceStateJson",
            devicestatejson=DeviceStateJson(devid="pico-1", statejson="bad-json"),
        ),
    ],
)
def test_missing_headset_and_bad_packets_clear_tracking(
    source, pc_service, feedback_factory, event
):
    module, body, _, velocity = source
    _, outgoing = pc_service.subscriptions.get(timeout=2)
    outgoing.put(feedback_factory(0))
    outgoing.put(feedback_factory(1))
    take(body, lambda msg: msg.joints is not None)
    take(velocity, lambda msg: msg.linear.x > 0)
    outgoing.put(event)
    assert take(body, lambda msg: msg.joints is None).joints is None
    assert take(velocity, lambda msg: msg.linear.x == 0).angular.z == 0
    assert module.tracking_status()["body_available"] is False


def test_stopping_module_cancels_idle_subscription_and_stops_commands(
    source, pc_service, feedback_factory
):
    module, body, _, velocity = source
    pid, outgoing = pc_service.subscriptions.get(timeout=2)
    outgoing.put(feedback_factory(0))
    outgoing.put(feedback_factory(1))
    take(body, lambda msg: msg.joints is not None)
    take(velocity, lambda msg: msg.linear.x > 0)
    module.stop()
    assert pc_service.cancellations.get(timeout=2) == pid
    assert take(body, lambda msg: msg.joints is None).joints is None
    assert take(velocity, lambda msg: msg.linear.x == 0).angular.z == 0
    assert module.tracking_status()["pc_service_connected"] is False


@pytest.fixture
def module_factory():
    modules = []

    def create(**kwargs):
        module = PicoTeleopModule(**kwargs)
        modules.append(module)
        return module

    try:
        yield create
    finally:
        for module in modules:
            module.stop()


def test_build_prepares_service_without_starting_process(module_factory, mocker, tmp_path):
    prepare = mocker.patch("dimos.teleop.pico.module.ensure_pc_service")
    spawn = mocker.patch("dimos.teleop.pico.service.asyncio.create_subprocess_exec")
    module = module_factory(g=GlobalConfig(), pc_service_dir=tmp_path)

    module.build()

    prepare.assert_called_once_with(tmp_path)
    spawn.assert_not_called()
    assert module.tracking_status()["pc_service_connected"] is False


@pytest.mark.parametrize(
    ("manage", "host", "port"),
    [(False, "127.0.0.1", 60061), (True, "192.0.2.1", 60061), (True, "127.0.0.1", 60062)],
)
def test_build_skips_setup_for_external_services(module_factory, mocker, manage, host, port):
    prepare = mocker.patch("dimos.teleop.pico.module.ensure_pc_service")
    module = module_factory(
        g=GlobalConfig(xrobotoolkit_host=host, xrobotoolkit_port=port), manage_pc_service=manage
    )

    module.build()

    prepare.assert_not_called()
