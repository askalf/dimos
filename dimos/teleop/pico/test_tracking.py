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

from collections.abc import Callable
import json
from typing import Any

import numpy as np
import pytest

from dimos.control.tasks.g1_sonic_wbc_task.webxr_retargeting import (
    SMPL_WEBXR_JOINTS,
    WebXRSonicRetargeter,
)
from dimos.teleop.pico.tracking import (
    PICO_JOINT_NAMES,
    PicoPacket,
    PicoTrackingSession,
    parse_packet,
)


def decode(payload: dict[str, Any]) -> PicoPacket:
    return parse_packet(json.dumps({"value": json.dumps(payload)}))


def test_native_pose_enters_existing_nvidia_retargeter(packet_factory):
    session = PicoTrackingSession()
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    update = session.receive("pico", decode(packet_factory(1)), now=0.03, wall_time=100.03)
    assert update is not None
    assert update.body is not None
    assert update.body.capture_time_s == pytest.approx(100.02)
    assert tuple(update.body.joints) == PICO_JOINT_NAMES == tuple(SMPL_WEBXR_JOINTS)
    pose = WebXRSonicRetargeter().retarget(update.body, frame_index=0).fields
    np.testing.assert_allclose(pose["smpl_pose"], 0.0, atol=1e-6)
    np.testing.assert_allclose(pose["joint_pos"], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        pose["body_quat_w"], [[0.70710677, 0.0, 0.0, -0.70710677]], atol=1e-6
    )


def test_native_buttons_axes_and_right_stick_stop(packet_factory):
    session = PicoTrackingSession()
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    session.receive("pico", decode(packet_factory(1)), now=0.02, wall_time=100.02)
    payload = packet_factory(2)
    payload["Controller"]["left"].update(axisY=1.0, primaryButton=True, trigger=0.6, grip=1.0)
    payload["Controller"]["right"].update(axisX=0.5, primaryButton=True, trigger=0.25)
    update = session.receive("pico", decode(payload), now=0.04, wall_time=100.04)
    assert update is not None
    assert update.buttons.left_primary and update.buttons.right_primary
    assert update.buttons.left_grip and update.buttons.left_trigger
    assert update.buttons.left_trigger_analog == pytest.approx(0.6, abs=1 / 127)
    assert update.buttons.right_trigger_analog == pytest.approx(0.25, abs=1 / 127)
    assert update.cmd_vel.linear.x == 0.3
    assert update.cmd_vel.angular.z == -0.15
    payload["timeStampNs"] += 20_000_000
    payload["Controller"]["right"]["axisClick"] = True
    update = session.receive("pico", decode(payload), now=0.06, wall_time=100.06)
    assert update is not None
    assert update.cmd_vel.linear.x == update.cmd_vel.angular.z == 0.0


@pytest.mark.parametrize(
    ("x", "y", "expected_x", "expected_y"),
    [
        (0.0, 1.0, 0.6, 0.0),
        (0.0, -1.0, -0.6, 0.0),
        (-1.0, 0.0, 0.0, 0.6),
        (1.0, 0.0, 0.0, -0.6),
        (1.0, 1.0, 0.6 / np.sqrt(2), -0.6 / np.sqrt(2)),
        (0.0, 0.575, 0.35, 0.0),
        (0.1, 0.1, 0.0, 0.0),
        (0.0, 0.15, 0.0, 0.0),
        (0.0, 0.151, 0.100588235, 0.0),
    ],
)
def test_planar_stick_direction_deadzone_and_speed_range(
    packet_factory: Callable[[int], dict[str, Any]],
    x: float,
    y: float,
    expected_x: float,
    expected_y: float,
) -> None:
    session = PicoTrackingSession(
        linear_scale=0.6, linear_min_speed=0.1, yaw_scale=1.5, deadzone=0.15
    )
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    session.receive("pico", decode(packet_factory(1)), now=0.02, wall_time=100.02)
    payload = packet_factory(2)
    payload["Controller"]["left"].update(axisX=x, axisY=y)
    update = session.receive("pico", decode(payload), now=0.04, wall_time=100.04)
    assert update is not None
    assert update.cmd_vel.linear.x == pytest.approx(expected_x)
    assert update.cmd_vel.linear.y == pytest.approx(expected_y)
    assert update.cmd_vel.linear.z == update.cmd_vel.angular.z == 0.0


@pytest.mark.parametrize(
    ("stick", "yaw_rate"), [(-1.0, 1.5), (1.0, -1.5), (0.5, -0.75), (0.14, 0.0)]
)
def test_pico_reference_turn_rate(
    packet_factory: Callable[[int], dict[str, Any]], stick: float, yaw_rate: float
) -> None:
    session = PicoTrackingSession(yaw_scale=1.5, deadzone=0.15)
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    session.receive("pico", decode(packet_factory(1)), now=0.02, wall_time=100.02)
    payload = packet_factory(2)
    payload["Controller"]["right"]["axisX"] = stick
    update = session.receive("pico", decode(payload), now=0.04, wall_time=100.04)
    assert update is not None
    assert update.cmd_vel.angular.z == yaw_rate


@pytest.mark.parametrize("stop", ["center", "stick_click", "focus_lost", "abxy"])
def test_stop_clears_sideways_motion_and_turning(
    packet_factory: Callable[[int], dict[str, Any]], stop: str
) -> None:
    session = PicoTrackingSession(linear_scale=0.6, linear_min_speed=0.1)
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    session.receive("pico", decode(packet_factory(1)), now=0.02, wall_time=100.02)
    payload = packet_factory(2)
    payload["Controller"]["left"].update(axisX=1.0, axisY=1.0)
    payload["Controller"]["right"].update(axisX=-1.0)
    moving = session.receive("pico", decode(payload), now=0.04, wall_time=100.04)
    assert moving is not None
    assert moving.cmd_vel.linear.y < 0.0 < moving.cmd_vel.angular.z

    payload["timeStampNs"] += 20_000_000
    if stop == "center":
        payload["Controller"]["left"].update(axisX=0.0, axisY=0.0)
        payload["Controller"]["right"].update(axisX=0.0)
    elif stop == "stick_click":
        payload["Controller"]["right"]["axisClick"] = True
    elif stop == "focus_lost":
        payload["appState"]["focus"] = False
    else:
        for controller in payload["Controller"].values():
            controller.update(primaryButton=True, secondaryButton=True)
    update = session.receive("pico", decode(payload), now=0.06, wall_time=100.06)
    assert update is not None
    assert update.cmd_vel.linear.x == update.cmd_vel.linear.y == update.cmd_vel.angular.z == 0.0


def test_repeated_body_cannot_be_kept_alive_by_new_packets(packet_factory):
    session = PicoTrackingSession(stale_timeout=0.1)
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    session.receive("pico", decode(packet_factory(1)), now=0.02, wall_time=100.02)
    payload = packet_factory(1)
    payload["timeStampNs"] += 20_000_000
    unchanged = session.receive("pico", decode(payload), now=0.04, wall_time=100.04)
    assert unchanged is not None
    assert unchanged.body is None
    expired = session.expire(now=0.13, wall_time=100.13)
    assert expired is not None
    assert expired.body.joints is None
    assert expired.buttons.data == 0
    assert expired.cmd_vel.linear.x == expired.cmd_vel.angular.z == 0.0
    payload["timeStampNs"] += 100_000_000
    assert session.receive("pico", decode(payload), now=0.14, wall_time=100.14) is None
    assert session.status(now=0.14)["body_available"] is False


def test_stationary_pose_with_advancing_joint_clocks_stays_live(packet_factory):
    session = PicoTrackingSession(stale_timeout=0.1)
    for frame in range(20):
        session.receive(
            "pico", decode(packet_factory(frame)), now=frame * 0.02, wall_time=100 + frame * 0.02
        )
    assert session.status(now=0.4)["body_frames"] == 19
    assert session.status(now=0.4)["body_available"] is True


def test_pico_ultra_head_clock_accepts_complete_body_and_expires_cached_data(packet_factory):
    session = PicoTrackingSession(stale_timeout=0.1)
    for frame in range(3):
        payload = packet_factory(frame)
        for i, joint in enumerate(payload["Body"]["joints"]):
            joint["t"] = 1000 + frame if i == 15 else 0
        update = session.receive(
            "pico", decode(payload), now=frame * 0.02, wall_time=100 + frame * 0.02
        )
    assert update is not None
    assert len(update.body.joints) == 24
    assert session.status(now=0.04)["body_clock_joints"] == ["head"]
    assert session.status(now=0.04)["body_timestamp"] == 1002
    assert session.status(now=0.04)["body_frames"] == 2
    # The APK can refresh the outer packet while its entire Body stays cached.
    payload["timeStampNs"] += 20_000_000
    cached = session.receive("pico", decode(payload), now=0.06, wall_time=100.06)
    assert cached.body is None
    assert session.expire(now=0.15, wall_time=100.15).body.joints is None
    assert session.status(now=0.15)["body_available"] is False


def test_all_zero_body_clocks_do_not_invent_freshness_from_pose_changes(packet_factory):
    session = PicoTrackingSession()
    for frame in range(3):
        payload = packet_factory(frame)
        for joint in payload["Body"]["joints"]:
            joint.update(t=0, p=f"{frame},1,0,0,-1,0,0")
        update = session.receive(
            "pico", decode(payload), now=frame * 0.02, wall_time=100 + frame * 0.02
        )
        assert update.body.joints is None
        assert update.buttons.data == 0
    assert session.status(now=0.04)["state"] == "body_clock_missing"
    assert session.status(now=0.04)["body_frames"] == 0


def test_changing_populated_body_clocks_restarts_tracking_generation(packet_factory):
    session = PicoTrackingSession()
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    session.receive("pico", decode(packet_factory(1)), now=0.02, wall_time=100.02)
    old_frame_id = session.frame_id
    payload = packet_factory(2)
    for i, joint in enumerate(payload["Body"]["joints"]):
        joint["t"] = 1002 if i == 15 else 0
    update = session.receive("pico", decode(payload), now=0.04, wall_time=100.04)
    assert update.body.joints is None
    assert update.buttons.data == 0
    assert session.frame_id != old_frame_id
    assert session.status(now=0.04)["state"] == "body_clock_source_changed"
    assert session.status(now=0.04)["release_required"] is True


def test_held_ax_does_not_engage_on_start_or_after_disconnect(packet_factory):
    session = PicoTrackingSession()
    for frame in range(3):
        payload = packet_factory(frame)
        payload["Controller"]["left"].update(primaryButton=True, axisY=1.0)
        payload["Controller"]["right"]["primaryButton"] = True
        update = session.receive(
            "pico", decode(payload), now=frame * 0.02, wall_time=100 + frame * 0.02
        )
        assert update is not None
        assert not update.buttons.left_primary and not update.buttons.right_primary
        assert update.cmd_vel.linear.x == 0.0
    old_frame_id = session.frame_id
    session.invalidate("disconnected", wall_time=100.1, reset_clock=True)
    assert session.frame_id != old_frame_id
    session.receive("pico", decode(payload), now=0.1, wall_time=100.1)
    payload["timeStampNs"] += 20_000_000
    for joint in payload["Body"]["joints"]:
        joint["t"] += 1
    update = session.receive("pico", decode(payload), now=0.12, wall_time=100.12)
    assert update is not None
    assert not update.buttons.left_primary and not update.buttons.right_primary
    released = packet_factory(10)
    session.receive("pico", decode(released), now=0.26, wall_time=100.26)
    pressed = packet_factory(11)
    pressed["Controller"]["left"]["primaryButton"] = True
    pressed["Controller"]["right"]["primaryButton"] = True
    update = session.receive("pico", decode(pressed), now=0.28, wall_time=100.28)
    assert update is not None
    assert update.buttons.left_primary and update.buttons.right_primary


@pytest.mark.parametrize("body_available", [False, True])
def test_stop_buttons_survive_tracking_and_focus_gates(packet_factory, body_available):
    session = PicoTrackingSession()
    payload = packet_factory(0)
    payload["appState"]["focus"] = False
    if not body_available:
        payload.pop("Body")
    payload["Controller"]["left"].update(primaryButton=True, secondaryButton=True)
    payload["Controller"]["right"].update(primaryButton=True, secondaryButton=True)

    update = session.receive("pico", decode(payload), now=0.0, wall_time=100.0)

    assert update.buttons.left_primary and update.buttons.left_secondary
    assert update.buttons.right_primary and update.buttons.right_secondary
    assert update.cmd_vel.linear.x == update.cmd_vel.angular.z == 0.0
    assert session.status(now=0.0)["release_required"] is True


@pytest.mark.parametrize(
    "loss",
    [
        "body_missing",
        "app_not_focused",
        "body_clock_reset",
        "packet_clock_reset",
        "packet_clock_jump",
        "delayed_packets",
    ],
)
def test_loss_clears_body_and_commands(packet_factory, loss):
    session = PicoTrackingSession()
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    session.receive("pico", decode(packet_factory(1)), now=0.02, wall_time=100.02)
    payload = packet_factory(2)
    now = 0.04
    if loss == "body_missing":
        del payload["Body"]
    elif loss == "app_not_focused":
        payload["appState"]["focus"] = False
    elif loss == "body_clock_reset":
        payload["Body"]["joints"][0]["t"] = 1
    elif loss == "packet_clock_reset":
        payload["timeStampNs"] -= 1_000_000_000
    elif loss == "packet_clock_jump":
        payload["timeStampNs"] += 10_000_000_000
    else:
        # Keep the body recently received while packet time falls behind real time.
        session.receive("pico", decode(payload), now=0.9, wall_time=100.9)
        payload = packet_factory(3)
        now = 1.1
    update = session.receive("pico", decode(payload), now=now, wall_time=100 + now)
    assert update is not None
    assert update.body.joints is None
    assert update.buttons.data == 0
    assert update.cmd_vel.linear.x == update.cmd_vel.angular.z == 0.0
    assert session.reason == loss


def test_packet_duplicates_and_other_headsets_do_not_refresh_tracking(packet_factory):
    session = PicoTrackingSession(device_id="wanted", stale_timeout=0.1)
    assert session.receive("other", decode(packet_factory(0)), now=0.0, wall_time=100.0) is None
    session.receive("wanted", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    session.receive("wanted", decode(packet_factory(1)), now=0.02, wall_time=100.02)
    assert session.receive("wanted", decode(packet_factory(1)), now=0.03, wall_time=100.03) is None
    assert session.receive("other", decode(packet_factory(2)), now=0.12, wall_time=100.12) is None
    assert session.expire(now=0.13, wall_time=100.13).body.joints is None


def test_one_stale_joint_blocks_a_partial_body_update(packet_factory):
    session = PicoTrackingSession()
    session.receive("pico", decode(packet_factory(0)), now=0.0, wall_time=100.0)
    payload = packet_factory(1)
    payload["Body"]["joints"][23]["t"] = 1000
    assert session.receive("pico", decode(payload), now=0.02, wall_time=100.02) is None
    assert session.status(now=0.02)["body_available"] is False


@pytest.mark.parametrize("bad_pose", ["0,0", "0,0,0,0,0,0,0", "nan,0,0,0,0,0,1", "0,0,0,inf,0,0,1"])
def test_rejects_invalid_joint_geometry(packet_factory, bad_pose):
    payload = packet_factory(0)
    payload["Body"]["joints"][0]["p"] = bad_pose
    with pytest.raises(ValueError, match="pose|quaternion"):
        PicoTrackingSession().receive("pico", decode(payload), now=0.0, wall_time=100.0)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda p: p["Body"]["joints"].pop(),
        lambda p: p["Body"]["joints"][0].update(t=-1),
        lambda p: p["Controller"]["left"].update(axisX=float("nan")),
        lambda p: p["Controller"].pop("right"),
        lambda p: p["appState"].update(focus="false"),
    ],
)
def test_rejects_incomplete_or_malformed_packets(
    packet_factory, corrupt: Callable[[dict[str, Any]], Any]
):
    payload = packet_factory(0)
    corrupt(payload)
    with pytest.raises(ValueError):
        decode(payload)


@pytest.mark.parametrize("raw", ["[]", "{}", '{"value": {}}', '{"value": "invalid"}'])
def test_rejects_bad_envelope(raw):
    with pytest.raises(ValueError):
        parse_packet(raw)
