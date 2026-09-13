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

"""Read atomic tracking events directly from the public XRoboToolkit PC service."""

import asyncio
from collections.abc import AsyncGenerator
import os
from typing import Any

from google.protobuf.empty_pb2 import Empty
import grpc

from dimos.teleop.pico.proto.tracking_pb2 import ServerFeedback, VRPid


class PCServiceClient:
    def __init__(self, target: str, *, rpc_timeout: float = 0.5) -> None:
        self.target = target
        self.rpc_timeout = rpc_timeout

    async def is_ready(self) -> bool:
        """Probe the service protocol without registering a tracking subscriber."""
        async with grpc.aio.insecure_channel(self.target) as channel:
            beat = channel.unary_unary(
                "/PXREAService.EAService/SendBeat",
                request_serializer=Empty.SerializeToString,
                response_deserializer=Empty.FromString,
            )
            try:
                await beat(Empty(), timeout=self.rpc_timeout)
            except grpc.aio.AioRpcError:
                return False
            return True

    async def events(self) -> AsyncGenerator[ServerFeedback | None, None]:
        """Yield None on connection, then feedback; raise on stream/heartbeat loss.

        The caller owns reconnect policy. Closing this generator cancels its
        subscription, heartbeat task and gRPC channel, including an idle stream.
        """
        async with grpc.aio.insecure_channel(self.target) as channel:
            beat = channel.unary_unary(
                "/PXREAService.EAService/SendBeat",
                request_serializer=Empty.SerializeToString,
                response_deserializer=Empty.FromString,
            )
            subscribe = channel.unary_stream(
                "/PXREAService.EAService/WatchServerFeedback",
                request_serializer=VRPid.SerializeToString,
                response_deserializer=ServerFeedback.FromString,
            )
            cancel = channel.unary_unary(
                "/PXREAService.EAService/CancelServerFeedback",
                request_serializer=VRPid.SerializeToString,
                response_deserializer=Empty.FromString,
            )
            await beat(Empty(), timeout=self.rpc_timeout)
            pid = VRPid(pid=os.getpid())
            stream = subscribe(pid)

            async def heartbeat() -> None:
                while True:
                    await asyncio.sleep(0.5)
                    await beat(Empty(), timeout=self.rpc_timeout)

            heartbeat_task = asyncio.create_task(heartbeat(), name="pico-heartbeat")
            read_task: asyncio.Task[Any] | None = None
            try:
                yield None
                while True:
                    read_task = asyncio.create_task(stream.read(), name="pico-feedback")
                    await asyncio.wait(
                        (read_task, heartbeat_task), return_when=asyncio.FIRST_COMPLETED
                    )
                    if heartbeat_task.done():
                        await heartbeat_task
                    feedback = await read_task
                    if not isinstance(feedback, ServerFeedback):
                        raise ConnectionError("XRoboToolkit feedback stream ended")
                    yield feedback
            finally:
                stream.cancel()
                heartbeat_task.cancel()
                tasks: list[asyncio.Task[Any]] = [heartbeat_task]
                if read_task is not None:
                    read_task.cancel()
                    tasks.append(read_task)
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await cancel(pid, timeout=self.rpc_timeout)
                except grpc.aio.AioRpcError:
                    # A disconnected service cannot acknowledge cancellation.
                    pass
