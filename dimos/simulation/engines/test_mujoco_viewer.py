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

"""Viewer IPC drops old frames and cannot backpressure physics."""

import os
import socket
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dimos.simulation.engines.mujoco_viewer import SnapshotViewer, ViewerConfig, run_viewer

pytestmark = pytest.mark.mujoco


def test_unread_viewer_queue_drops_frames_and_retains_latest_camera_request(tmp_path, mocker):
    peers = []
    process = mocker.Mock()
    process.poll.return_value = None

    def spawn(*args, **kwargs):
        peers.append(socket.socket(fileno=os.dup(kwargs["pass_fds"][0])))
        return process

    mocker.patch("dimos.simulation.engines.mujoco_viewer.subprocess.Popen", side_effect=spawn)
    try:
        with SnapshotViewer(tmp_path / "scene.mjb", ViewerConfig()) as display:
            display._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)
            for _ in range(100):
                assert display.publish(np.zeros(256), None)
            assert display.publish(np.ones(256), (90, -30, 3))
            # Drain the stale queue. The pending camera change survives a dropped frame.
            peer = peers[0]
            peer.setblocking(False)
            frames = 0
            while True:
                try:
                    peer.recv(4096)
                    frames += 1
                except BlockingIOError:
                    break
            assert 0 < frames < 100
            assert display.publish(np.full(256, 2.0), None)
            packet = np.frombuffer(peer.recv(4096), dtype=np.float64)
            np.testing.assert_array_equal(packet[:4], [1, 90, -30, 3])
            np.testing.assert_array_equal(packet[4:], np.full(256, 2.0))
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)
    finally:
        for peer in peers:
            peer.close()


def test_viewer_displays_newest_snapshot_and_updates_camera(tmp_path, mocker):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
    <joint name="joint"/><geom type="sphere" size=".1"/>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    path = tmp_path / "scene.mjb"
    mujoco.mj_saveModel(model, str(path), None)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    handle = mocker.MagicMock()
    handle.__enter__.return_value = handle
    handle.is_running.side_effect = [True, False]
    handle.cam = SimpleNamespace(lookat=np.zeros(3))
    launch = mocker.patch(
        "dimos.simulation.engines.mujoco_viewer.viewer.launch_passive", return_value=handle
    )
    mocker.patch("dimos.simulation.engines.mujoco_viewer.time.sleep")
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    with sender, receiver:
        for position in (0.1, 0.4):
            data.qpos[0] = position
            mujoco.mj_getState(model, data, state, spec)
            sender.send(np.concatenate(([1, 90, -30, 3], state)).tobytes())
        run_viewer(path, receiver, ViewerConfig())
    displayed_model, displayed_data = launch.call_args.args
    assert displayed_model is not model
    assert displayed_data.qpos[0] == pytest.approx(0.4)
    assert (handle.cam.azimuth, handle.cam.elevation, handle.cam.distance) == (90, -30, 3)
    handle.sync.assert_called_once_with()
