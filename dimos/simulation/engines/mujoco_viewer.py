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

"""Process-isolated, latest-frame MuJoCo display for interactive robot stacks."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
from types import TracebackType
from typing import Any

import mujoco
import mujoco.viewer as viewer  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ViewerConfig:
    fps: float = 30.0
    track_body: str | None = None
    lookat: tuple[float, float, float] | None = None
    distance: float | None = None
    azimuth: float | None = None
    elevation: float | None = None


class SnapshotViewer:
    """Send bounded, nonblocking snapshots to a separate native viewer process.

    Full queues drop frames. A slow or covered window cannot apply backpressure
    to robot physics, and native GUI calls cannot hold its Python interpreter.
    """

    def __init__(self, model_path: Path, config: ViewerConfig) -> None:
        self._model_path = model_path
        self._config = config

    def __enter__(self) -> SnapshotViewer:
        self._socket, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._camera = np.zeros(4)
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "dimos.simulation.engines.mujoco_viewer",
                    "--model",
                    str(self._model_path),
                    "--fd",
                    str(child.fileno()),
                    "--parent-pid",
                    str(os.getpid()),
                    "--config",
                    json.dumps(asdict(self._config)),
                ],
                pass_fds=(child.fileno(),),
            )
        except BaseException:
            self._socket.close()
            raise
        finally:
            child.close()
        return self

    def publish(
        self, state: NDArray[np.float64], camera: tuple[float, float, float] | None
    ) -> bool:
        if self._process.poll() is not None:
            return False
        if camera is not None:
            self._camera[0] += 1
            self._camera[1:] = camera
        packet = np.concatenate((self._camera, state)).tobytes()
        try:
            self._socket.send(packet)
        except BlockingIOError:
            pass
        except (BrokenPipeError, ConnectionRefusedError):
            return False
        return True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._socket.close()
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


def run_viewer(model_path: Path, channel: socket.socket, config: ViewerConfig) -> None:
    model = mujoco.MjModel.from_binary_path(str(model_path))  # type: ignore[attr-defined]
    data = mujoco.MjData(model)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    packet_size = (4 + mujoco.mj_stateSize(model, spec)) * 8
    channel.setblocking(False)
    camera_sequence = 0.0
    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as handle:
        with handle.lock():
            if config.track_body is not None:
                handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                handle.cam.trackbodyid = model.body(config.track_body).id
            if config.lookat is not None:
                handle.cam.lookat[:] = config.lookat
            for name in ("distance", "azimuth", "elevation"):
                value = getattr(config, name)
                if value is not None:
                    setattr(handle.cam, name, value)
        while handle.is_running():
            started = time.monotonic()
            # Drain queued frames and display the freshest complete state.
            select.select([channel], [], [], 1.0 / config.fps)
            packet = None
            while True:
                try:
                    packet = channel.recv(packet_size)
                except BlockingIOError:
                    break
            if packet is None:
                continue
            if len(packet) != packet_size:
                raise ValueError("Incomplete MuJoCo viewer snapshot")
            values = np.frombuffer(packet, dtype=np.float64)
            with handle.lock():
                mujoco.mj_setState(model, data, values[4:], spec)
                mujoco.mj_forward(model, data)
                if values[0] != camera_sequence:
                    camera_sequence = values[0]
                    handle.cam.azimuth, handle.cam.elevation, handle.cam.distance = values[1:4]
            handle.sync()
            time.sleep(max(0.0, 1.0 / config.fps - (time.monotonic() - started)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if sys.platform.startswith("linux"):
        # Set this in the child, not preexec_fn in the multithreaded simulator.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "Could not tie viewer lifetime to simulator")
    if os.getppid() != args.parent_pid:
        return
    settings: dict[str, Any] = json.loads(args.config)
    with socket.socket(fileno=args.fd) as channel:
        run_viewer(args.model, channel, ViewerConfig(**settings))


if __name__ == "__main__":
    main()
