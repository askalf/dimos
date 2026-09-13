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
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from queue import Queue
import threading
from typing import Any

from google.protobuf.empty_pb2 import Empty
import grpc
import pytest

from dimos.teleop.pico.proto.tracking_pb2 import DeviceStateJson, ServerFeedback, VRPid


@pytest.fixture
def packet_factory() -> Callable[[int], dict[str, Any]]:
    def make(frame: int) -> dict[str, Any]:
        controller = {
            "pose": "0,0,0,0,0,0,1",
            "axisX": 0.0,
            "axisY": 0.0,
            "axisClick": False,
            "grip": 0.0,
            "trigger": 0.0,
            "primaryButton": False,
            "secondaryButton": False,
            "menuButton": False,
        }
        return {
            "timeStampNs": 1_700_000_000_000_000_000 + frame * 20_000_000,
            "appState": {"focus": True},
            "Controller": {"left": dict(controller), "right": dict(controller)},
            "Body": {"joints": [{"p": "0,1,0,0,-1,0,0", "t": 1000 + frame} for _ in range(24)]},
        }

    return make


@pytest.fixture
def feedback_factory(
    packet_factory: Callable[[int], dict[str, Any]],
) -> Callable[..., ServerFeedback]:
    def make(frame: int, *, device_id: str = "pico-1", pressed: bool = False) -> ServerFeedback:
        payload = packet_factory(frame)
        payload["Controller"]["left"]["primaryButton"] = pressed
        payload["Controller"]["right"]["primaryButton"] = pressed
        payload["Controller"]["left"]["axisY"] = 1.0
        return ServerFeedback(
            name="deviceStateJson",
            devicestatejson=DeviceStateJson(
                devid=device_id,
                statejson=json.dumps({"value": json.dumps(payload)}),
            ),
        )

    return make


class LocalPCService:
    """Real loopback gRPC transport with controllable feedback and heartbeat."""

    def __init__(self) -> None:
        self.subscriptions: Queue[tuple[int, Queue[ServerFeedback | None]]] = Queue()
        self.cancellations: Queue[int] = Queue()
        self.fail_heartbeat = threading.Event()
        self.port = 0

    def watch(self, request: VRPid, context: grpc.ServicerContext) -> Iterator[ServerFeedback]:
        events: Queue[ServerFeedback | None] = Queue()
        context.add_callback(lambda: events.put(None))
        self.subscriptions.put((request.pid, events))
        while True:
            feedback = events.get(timeout=10)
            if feedback is None:
                return
            yield feedback

    def beat(self, request: Empty, context: grpc.ServicerContext) -> Empty:
        if self.fail_heartbeat.is_set():
            context.abort(grpc.StatusCode.UNAVAILABLE, "heartbeat failed")
        return Empty()

    def cancel(self, request: VRPid, context: grpc.ServicerContext) -> Empty:
        self.cancellations.put(request.pid)
        return Empty()


@pytest.fixture
def pc_service() -> Iterator[LocalPCService]:
    service = LocalPCService()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "PXREAService.EAService",
                    {
                        "WatchServerFeedback": grpc.unary_stream_rpc_method_handler(
                            service.watch,
                            request_deserializer=VRPid.FromString,
                            response_serializer=ServerFeedback.SerializeToString,
                        ),
                        "SendBeat": grpc.unary_unary_rpc_method_handler(
                            service.beat,
                            request_deserializer=Empty.FromString,
                            response_serializer=Empty.SerializeToString,
                        ),
                        "CancelServerFeedback": grpc.unary_unary_rpc_method_handler(
                            service.cancel,
                            request_deserializer=VRPid.FromString,
                            response_serializer=Empty.SerializeToString,
                        ),
                    },
                ),
            )
        )
        service.port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            yield service
        finally:
            server.stop(0).wait(timeout=5)
    # Closed aio channels retained by exception tracebacks release gRPC's shared
    # completion-queue thread when their reference cycles are collected.
    gc.collect()
