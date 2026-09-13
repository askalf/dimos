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

"""Send DimOS camera images to the stock XRoboToolkit APK's Remote Vision view.

Implements XRoboToolkit v1.1.1's OPEN_CAMERA/CLOSE_CAMERA control messages and
its TCP video framing: a big-endian uint32 size followed by an Annex B H.264
access unit. The APK's built-in PICO4U and ZEDMINI profiles need no modification.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from fractions import Fraction
import struct
import threading
import time
from typing import Any

import av
from av.video.codeccontext import VideoCodecContext
import cv2
import numpy as np
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_COMMAND_LENGTH = struct.Struct("<i")
_VIDEO_LENGTH = struct.Struct("!I")
_CAMERA_HEADER = struct.Struct("<7i")
_CAMERA_MAGIC = b"\xca\xfe\x01"
_MAX_COMMAND_BYTES = 64
_MAX_REQUEST_BYTES = 1024


@dataclass(frozen=True)
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    port: int

    @classmethod
    def decode(cls, data: bytes) -> "CameraRequest":
        if not data.startswith(_CAMERA_MAGIC) or len(data) < 33:
            raise ValueError("Invalid XRoboToolkit camera request header")
        width, height, fps, bitrate, hevc, mode, port = _CAMERA_HEADER.unpack_from(data, 3)
        # Consume both length-prefixed strings (camera kind and advertised IP).
        # Video goes back to the control connection's peer, not an arbitrary IP.
        offset = 31
        for _ in range(2):
            if offset >= len(data):
                raise ValueError("Truncated XRoboToolkit camera request")
            offset += 1 + data[offset]
        if offset != len(data):
            raise ValueError("Invalid XRoboToolkit camera request strings")
        if not (
            64 <= width <= 4096 and width % 4 == 0 and 64 <= height <= 2160 and height % 2 == 0
        ):
            raise ValueError("Unsupported camera dimensions for side-by-side H.264")
        if not (1 <= fps <= 120 and 1 <= bitrate <= 100_000_000 and 1 <= port <= 65535):
            raise ValueError("Invalid camera frame rate, bitrate or receiver port")
        if hevc != 0 or mode != 2:
            raise ValueError("Only the stock APK's H.264 stereo layout is supported")
        return cls(width, height, fps, bitrate, port)


async def read_camera_command(reader: asyncio.StreamReader) -> tuple[str, bytes]:
    command_size = _COMMAND_LENGTH.unpack(await reader.readexactly(4))[0]
    if not 0 < command_size <= _MAX_COMMAND_BYTES:
        raise ValueError("Invalid camera command length")
    command = (await reader.readexactly(command_size)).decode("ascii")
    data_size = _COMMAND_LENGTH.unpack(await reader.readexactly(4))[0]
    if not 0 <= data_size <= _MAX_REQUEST_BYTES:
        raise ValueError("Invalid camera command payload length")
    return command, await reader.readexactly(data_size)


async def _close_socket(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except OSError as exc:
        logger.warning("PICO video socket closed after connection loss", error=str(exc))


class CameraEncoder:
    """Encode one monocular camera into both eyes of the APK's stereo layout."""

    def __init__(self, request: CameraRequest, fps: int, bitrate: int) -> None:
        self._request = request
        self._pts = 0
        self._codec: VideoCodecContext = av.CodecContext.create("libx264", "w")
        self._codec.width = request.width
        self._codec.height = request.height
        self._codec.pix_fmt = "yuv420p"
        self._codec.time_base = Fraction(1, fps)
        self._codec.framerate = Fraction(fps, 1)
        self._codec.bit_rate = bitrate
        self._codec.thread_count = 2
        self._codec.gop_size = fps
        self._codec.max_b_frames = 0
        self._codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "x264-params": "repeat-headers=1:annexb=1",
        }

    def encode(self, image: Image) -> list[bytes]:
        rgb = image.to_rgb().data
        height, width = self._request.height, self._request.width // 2
        scale = min(width / rgb.shape[1], height / rgb.shape[0])
        resized = cv2.resize(
            rgb, (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        )
        eye = np.zeros((height, width, 3), dtype=np.uint8)
        y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
        eye[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        frame = av.VideoFrame.from_ndarray(np.concatenate((eye, eye), axis=1), format="rgb24")
        frame.pts = self._pts
        self._pts += 1
        return [bytes(packet) for packet in self._codec.encode(frame)]


class PicoVideoConfig(ModuleConfig):
    max_fps: int = Field(default=20, ge=1, le=60)
    max_bitrate: int = Field(default=4_000_000, ge=100_000, le=20_000_000)
    socket_timeout: float = Field(default=2.0, gt=0)


class PicoVideoModule(Module):
    """Offer a camera stream when the stock APK requests Remote Vision."""

    config: PicoVideoConfig
    color_image: In[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frames: asyncio.Queue[Image] = asyncio.Queue(maxsize=1)
        self._client_task: asyncio.Task[Any] | None = None
        self._status_lock = threading.Lock()
        self._status: dict[str, Any] = {
            "listening": False,
            "streaming": False,
            "peer": None,
            "frames_sent": 0,
            "bytes_sent": 0,
            "last_error": None,
        }

    @rpc
    def video_status(self) -> dict[str, Any]:
        """Report the request listener and delivery to the headset's video socket."""
        with self._status_lock:
            return dict(self._status)

    def _set_status(self, **values: Any) -> None:
        with self._status_lock:
            self._status.update(values)

    async def handle_color_image(self, image: Image) -> None:
        if self._frames.full():
            self._frames.get_nowait()
        self._frames.put_nowait(image)

    async def main(self) -> AsyncIterator[None]:
        g = self.config.g
        server = await asyncio.start_server(
            self._serve_client, g.listen_host, g.xrobotoolkit_video_port
        )
        port = server.sockets[0].getsockname()[1]
        self._set_status(listening=True, host=g.listen_host, port=port)
        logger.info("PICO Remote Vision listening", host=g.listen_host, port=port)
        try:
            yield
        finally:
            server.close()
            if self._client_task is not None:
                self._client_task.cancel()
                await asyncio.gather(self._client_task, return_exceptions=True)
            await server.wait_closed()
            self._set_status(listening=False)

    async def _serve_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._client_task is not None:
            await _close_socket(writer)
            return
        self._client_task = asyncio.current_task()
        peer = str(writer.get_extra_info("peername")[0])
        video_task: asyncio.Task[None] | None = None
        self._set_status(peer=peer, last_error=None)
        try:
            while True:
                command, data = await read_camera_command(reader)
                if command not in {"OPEN_CAMERA", "CLOSE_CAMERA"}:
                    raise ValueError(f"Unsupported camera command: {command}")
                request = CameraRequest.decode(data) if command == "OPEN_CAMERA" else None
                if video_task is not None:
                    video_task.cancel()
                    await asyncio.gather(video_task, return_exceptions=True)
                    video_task = None
                if request is not None:
                    # Start from a new image, not the frame buffered before Listen.
                    while not self._frames.empty():
                        self._frames.get_nowait()
                    video_task = asyncio.create_task(self._send_video(peer, request))
        except asyncio.IncompleteReadError:
            pass  # The APK closed its control connection.
        except (OSError, ValueError) as exc:
            self._set_status(last_error=str(exc))
            logger.warning("PICO video request failed", peer=peer, error=str(exc))
        finally:
            if video_task is not None:
                video_task.cancel()
                await asyncio.gather(video_task, return_exceptions=True)
            await _close_socket(writer)
            self._client_task = None
            self._set_status(peer=None)

    async def _send_video(self, peer: str, request: CameraRequest) -> None:
        writer: asyncio.StreamWriter | None = None
        frames_sent = bytes_sent = 0
        fps = min(request.fps, self.config.max_fps)
        bitrate = min(request.bitrate, self.config.max_bitrate)
        try:
            _, connection = await asyncio.wait_for(
                asyncio.open_connection(peer, request.port),
                timeout=self.config.socket_timeout,
            )
            writer = connection
            encoder = CameraEncoder(request, fps, bitrate)
            self._set_status(
                streaming=True,
                frames_sent=0,
                bytes_sent=0,
                last_error=None,
                width=request.width,
                height=request.height,
                fps=fps,
                bitrate=bitrate,
            )
            logger.info(
                "PICO video connected",
                peer=peer,
                width=request.width,
                height=request.height,
                fps=fps,
            )
            next_frame = 0.0
            while True:
                image = await self._frames.get()
                await asyncio.sleep(max(0.0, next_frame - time.monotonic()))
                while not self._frames.empty():
                    image = self._frames.get_nowait()
                frame_started_at = time.monotonic()
                packets = await asyncio.to_thread(encoder.encode, image)
                for packet in packets:
                    connection.write(_VIDEO_LENGTH.pack(len(packet)) + packet)
                    bytes_sent += len(packet)
                await asyncio.wait_for(connection.drain(), timeout=self.config.socket_timeout)
                frames_sent += 1
                self._set_status(
                    frames_sent=frames_sent, bytes_sent=bytes_sent, last_frame_time=time.time()
                )
                next_frame = frame_started_at + 1 / fps
        except (OSError, TimeoutError, ValueError, av.FFmpegError, cv2.error) as exc:
            self._set_status(last_error=str(exc))
            logger.warning("PICO video stream ended", peer=peer, error=str(exc))
        finally:
            if writer is not None:
                await _close_socket(writer)
            self._set_status(streaming=False)
