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

import asyncio
from contextlib import aclosing
import os

import grpc
import pytest

from dimos.teleop.pico.client import PCServiceClient


@pytest.mark.asyncio
async def test_streams_native_feedback_and_cancels_subscription(pc_service, feedback_factory):
    client = PCServiceClient(f"127.0.0.1:{pc_service.port}")
    async with aclosing(client.events()) as events:
        assert await asyncio.wait_for(events.__anext__(), 2) is None
        pid, outgoing = await asyncio.to_thread(pc_service.subscriptions.get, timeout=2)
        assert pid == os.getpid()
        feedback = feedback_factory(0)
        outgoing.put(feedback)
        assert await asyncio.wait_for(events.__anext__(), 2) == feedback
    assert pc_service.cancellations.get(timeout=2) == pid


@pytest.mark.asyncio
async def test_heartbeat_failure_interrupts_an_idle_stream(pc_service):
    client = PCServiceClient(f"127.0.0.1:{pc_service.port}")
    async with aclosing(client.events()) as events:
        assert await asyncio.wait_for(events.__anext__(), 2) is None
        pc_service.fail_heartbeat.set()
        with pytest.raises(grpc.aio.AioRpcError, match="heartbeat failed"):
            await asyncio.wait_for(events.__anext__(), 2)


@pytest.mark.asyncio
async def test_end_of_stream_is_a_disconnect(pc_service):
    client = PCServiceClient(f"127.0.0.1:{pc_service.port}")
    async with aclosing(client.events()) as events:
        assert await asyncio.wait_for(events.__anext__(), 2) is None
        _, outgoing = await asyncio.to_thread(pc_service.subscriptions.get, timeout=2)
        outgoing.put(None)
        with pytest.raises(ConnectionError, match="stream ended"):
            await asyncio.wait_for(events.__anext__(), 2)


@pytest.mark.asyncio
async def test_cancelling_an_idle_reader_releases_the_subscription(pc_service):
    client = PCServiceClient(f"127.0.0.1:{pc_service.port}")
    async with aclosing(client.events()) as events:
        assert await asyncio.wait_for(events.__anext__(), 2) is None
        pid, _ = await asyncio.to_thread(pc_service.subscriptions.get, timeout=2)
        reader = asyncio.create_task(events.__anext__())
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
    assert pc_service.cancellations.get(timeout=2) == pid
