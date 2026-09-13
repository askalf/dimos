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
import struct

import av
import numpy as np
import pytest

from dimos.core.global_config import GlobalConfig
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.teleop.pico.video import CameraEncoder, CameraRequest, PicoVideoModule

# CameraRequestSerializer.cs v1.1.1, PICO4U profile: 2160x810, 60 Hz,
# 20 Mibit/s bitrate field, H.264, stereo, port 12345, camera="VR", IP=127.0.0.1.
_PICO4U_REQUEST = bytes.fromhex(
    "cafe01 70080000 2a030000 3c000000 00004001 00000000 02000000 39300000"
    "025652 093132372e302e302e31"
)


def camera_command(command: str, payload: bytes = b"") -> bytes:
    name = command.encode("ascii")
    return struct.pack("<i", len(name)) + name + struct.pack("<i", len(payload)) + payload


def request_for_port(port: int) -> bytes:
    # Small test image; preserve the released APK's packet layout.
    data = bytearray(_PICO4U_REQUEST)
    struct.pack_into("<ii", data, 3, 128, 64)
    struct.pack_into("<i", data, 27, port)
    return bytes(data)


@pytest.fixture
async def video_main():
    module = PicoVideoModule(g=GlobalConfig(listen_host="127.0.0.1", xrobotoolkit_video_port=0))
    try:
        async with aclosing(module.main()) as running:
            await anext(running)
            yield module, running
    finally:
        module.stop()


@pytest.fixture
def video_module(video_main):
    return video_main[0]


@pytest.fixture
async def control(video_module):
    reader, writer = await asyncio.open_connection("127.0.0.1", video_module.video_status()["port"])
    try:
        yield reader, writer
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.fixture
async def headset():
    connections: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = asyncio.Queue()
    writers: list[asyncio.StreamWriter] = []

    def connected(reader, writer):
        writers.append(writer)
        connections.put_nowait((reader, writer))

    server = await asyncio.start_server(connected, "127.0.0.1", 0)
    async with server:
        try:
            yield server.sockets[0].getsockname()[1], connections
        finally:
            for writer in writers:
                writer.close()
            await asyncio.gather(*(writer.wait_closed() for writer in writers))


async def receive_image(reader: asyncio.StreamReader) -> np.ndarray:
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    packet = await reader.readexactly(size)
    # A newly opened stream must decode immediately, without earlier packets.
    decoder = av.CodecContext.create("h264", "r")
    frames = decoder.decode(av.Packet(packet))
    assert len(frames) == 1
    return frames[0].to_ndarray(format="rgb24")


def test_released_profile_encodes_a_decodable_image_for_both_eyes():
    request = CameraRequest.decode(_PICO4U_REQUEST)
    encoder = CameraEncoder(request, fps=20, bitrate=4_000_000)
    image = Image(np.full((48, 64, 3), (20, 40, 220), dtype=np.uint8), format=ImageFormat.BGR)
    packets = encoder.encode(image)
    decoder = av.CodecContext.create("h264", "r")
    frames = [frame for packet in packets for frame in decoder.decode(av.Packet(packet))]
    assert len(frames) == 1
    rgb = frames[0].to_ndarray(format="rgb24")
    assert rgb.shape == (810, 2160, 3)
    np.testing.assert_allclose(rgb[405, 540], [220, 40, 20], atol=5)
    np.testing.assert_allclose(rgb[405, 1620], [220, 40, 20], atol=5)


async def test_fragmented_request_streams_latest_image_and_close_releases_video(
    video_module, control, headset
):
    _, control_writer = control
    port, connections = headset
    # Pre-Listen images must not be shown when a session starts.
    await video_module.handle_color_image(Image(np.zeros((32, 64, 3), dtype=np.uint8)))
    command = camera_command("OPEN_CAMERA", request_for_port(port))
    for part in (command[:2], command[2:13], command[13:]):
        control_writer.write(part)
        await control_writer.drain()
    reader, _ = await asyncio.wait_for(connections.get(), timeout=2)
    # Two arrivals before the sender runs: the newer one must win.
    await video_module.handle_color_image(Image(np.zeros((32, 64, 3), dtype=np.uint8)))
    await video_module.handle_color_image(
        Image(np.full((32, 64, 3), (20, 220, 40), dtype=np.uint8), format=ImageFormat.RGB)
    )
    rgb = await asyncio.wait_for(receive_image(reader), timeout=3)
    np.testing.assert_allclose(rgb[32, 32], [20, 220, 40], atol=5)
    np.testing.assert_allclose(rgb[32, 96], [20, 220, 40], atol=5)
    # Letterboxing preserves the source's 2:1 aspect ratio in each square eye.
    np.testing.assert_allclose(rgb[0, 32], [0, 0, 0], atol=5)
    control_writer.write(camera_command("CLOSE_CAMERA"))
    await control_writer.drain()
    assert await asyncio.wait_for(reader.read(), timeout=2) == b""
    assert video_module.video_status()["frames_sent"] == 1


async def test_reopening_video_uses_a_new_independently_decodable_stream(
    video_module, control, headset
):
    _, control_writer = control
    port, connections = headset
    command = camera_command("OPEN_CAMERA", request_for_port(port))
    control_writer.write(command)
    await control_writer.drain()
    first_reader, _ = await asyncio.wait_for(connections.get(), timeout=2)
    await video_module.handle_color_image(Image(np.full((64, 64, 3), 80, dtype=np.uint8)))
    first = await asyncio.wait_for(receive_image(first_reader), timeout=3)
    np.testing.assert_allclose(first[32, 32], [80, 80, 80], atol=5)

    control_writer.write(command)
    await control_writer.drain()
    second_reader, _ = await asyncio.wait_for(connections.get(), timeout=2)
    assert await asyncio.wait_for(first_reader.read(), timeout=2) == b""
    await video_module.handle_color_image(Image(np.full((64, 64, 3), 160, dtype=np.uint8)))
    second = await asyncio.wait_for(receive_image(second_reader), timeout=3)
    np.testing.assert_allclose(second[32, 32], [160, 160, 160], atol=5)


@pytest.mark.parametrize("length", [-1, 65, 2**30])
async def test_invalid_control_length_closes_the_connection(video_module, control, length):
    reader, writer = control
    writer.write(struct.pack("<i", length))
    await writer.drain()
    assert await asyncio.wait_for(reader.read(), timeout=2) == b""
    assert video_module.video_status()["last_error"] == "Invalid camera command length"


async def test_control_disconnect_ends_video_even_without_camera_frames(control, headset):
    _, control_writer = control
    port, connections = headset
    control_writer.write(camera_command("OPEN_CAMERA", request_for_port(port)))
    await control_writer.drain()
    reader, _ = await asyncio.wait_for(connections.get(), timeout=2)
    control_writer.close()
    await control_writer.wait_closed()
    assert await asyncio.wait_for(reader.read(), timeout=2) == b""


async def test_module_shutdown_closes_active_control_and_video(video_main, control, headset):
    module, running = video_main
    control_reader, control_writer = control
    port, connections = headset
    control_writer.write(camera_command("OPEN_CAMERA", request_for_port(port)))
    await control_writer.drain()
    reader, _ = await asyncio.wait_for(connections.get(), timeout=2)

    await asyncio.wait_for(running.aclose(), timeout=2)

    assert await asyncio.wait_for(reader.read(), timeout=2) == b""
    assert await asyncio.wait_for(control_reader.read(), timeout=2) == b""
    assert module.video_status()["listening"] is False
