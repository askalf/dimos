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

"""Decode native PICO packets and gate the full-body source on fresh tracking."""

from dataclasses import dataclass
import json
import math
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, Field, StrictBool

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.teleop.webxr.body_tracking import BodyJointPose, BodyTrackingSnapshot
from dimos.teleop.webxr.controller_types import (
    Buttons,
    ThumbstickState,
    WebXRControllerState,
)

# PICO BodyTrackerRole order; collar and shoulder have different native names.
PICO_JOINT_NAMES = (
    "hips",
    "left-upper-leg",
    "right-upper-leg",
    "spine-lower",
    "left-lower-leg",
    "right-lower-leg",
    "spine-middle",
    "left-foot-ankle",
    "right-foot-ankle",
    "spine-upper",
    "left-foot-ball",
    "right-foot-ball",
    "neck",
    "left-shoulder",
    "right-shoulder",
    "head",
    "left-arm-upper",
    "right-arm-upper",
    "left-arm-lower",
    "right-arm-lower",
    "left-hand-wrist",
    "right-hand-wrist",
    "left-hand-palm",
    "right-hand-palm",
)

Axis = Annotated[float, Field(strict=True, ge=-1, le=1, allow_inf_nan=False)]
Analog = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Timestamp = Annotated[int, Field(strict=True, gt=0)]


class PicoController(BaseModel):
    axis_x: Axis = Field(alias="axisX")
    axis_y: Axis = Field(alias="axisY")
    axis_click: StrictBool = Field(alias="axisClick")
    trigger: Analog
    grip: Analog
    primary: StrictBool = Field(alias="primaryButton")
    secondary: StrictBool = Field(alias="secondaryButton")
    menu: StrictBool = Field(alias="menuButton")

    def controller_state(self, *, is_left: bool) -> WebXRControllerState:
        return WebXRControllerState(
            is_left=is_left,
            trigger=self.trigger,
            grip=self.grip,
            primary=self.primary,
            secondary=self.secondary,
            menu=self.menu,
            thumbstick_press=self.axis_click,
            thumbstick=ThumbstickState(x=self.axis_x, y=self.axis_y),
        )


class PicoControllers(BaseModel):
    left: PicoController
    right: PicoController


class PicoJoint(BaseModel):
    p: str
    # PICO 4 Ultra reports an IMU timestamp on the head joint only. Zero means
    # this joint has no separate clock, not that its inferred pose is invalid.
    t: Annotated[int, Field(strict=True, ge=0)]

    def pose(self) -> BodyJointPose:
        values = [float(part) for part in self.p.split(",")]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError("body pose must contain seven finite values")
        x, y, z, qx, qy, qz, qw = values
        norm = math.hypot(qx, qy, qz, qw)
        if not math.isfinite(norm) or norm < 1e-6:
            raise ValueError("body quaternion has invalid length")
        # The APK already converts Unity coordinates to its outgoing PICO frame.
        # SONIC's NVIDIA retargeter expects this frame; do not flip Z again.
        return BodyJointPose(
            position=(x, y, z),
            orientation=(qx / norm, qy / norm, qz / norm, qw / norm),
        )


class PicoBody(BaseModel):
    joints: list[PicoJoint] = Field(min_length=24, max_length=24)


class PicoAppState(BaseModel):
    focus: StrictBool


class PicoPacket(BaseModel):
    timestamp_ns: Timestamp = Field(alias="timeStampNs")
    app_state: PicoAppState = Field(alias="appState")
    controllers: PicoControllers = Field(alias="Controller")
    body: PicoBody | None = Field(default=None, alias="Body")


def parse_packet(state_json: str) -> PicoPacket:
    """Decode the PC service envelope and its stringified tracking JSON value."""
    envelope = json.loads(state_json)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("value"), str):
        raise ValueError("expected PC service JSON with a string 'value'")
    return PicoPacket.model_validate_json(envelope["value"])


@dataclass(frozen=True)
class TrackingUpdate:
    body: BodyTrackingSnapshot | None
    buttons: Buttons
    cmd_vel: Twist


class PicoTrackingSession:
    """One headset, with explicit clock epochs and no cached-pose restamping.

    Packet time is UTC nanoseconds. Joint ``t`` is an opaque sensor clock: only
    ordering is assumed. Two complete samples with their populated joint clocks
    advancing are required before publishing. Capture time preserves packet intervals, aligned
    to the PC on the first packet; it is not an absolute latency measurement.
    """

    def __init__(
        self,
        *,
        device_id: str | None = None,
        stale_timeout: float = 1.0,
        linear_scale: float = 0.3,
        yaw_scale: float = 0.3,
        deadzone: float = 0.18,
    ) -> None:
        self.device_id = device_id
        self.stale_timeout = stale_timeout
        self.linear_scale = linear_scale
        self.yaw_scale = yaw_scale
        self.deadzone = deadzone
        self.reason = "waiting_for_headset"
        self.frames = 0
        self.packets = 0
        self._session_id = uuid4().hex[:12]
        self._generation = 0
        self._available = False
        self._release_required = True
        self._joint_times: tuple[int, ...] | None = None
        self._last_packet_ns: int | None = None
        self._anchor: tuple[int, float, float] | None = None
        self._last_body_at: float | None = None
        self._last_packet_at: float | None = None

    @property
    def frame_id(self) -> str:
        return f"pico/{self.device_id or 'unknown'}/{self._session_id}/{self._generation}"

    def accepts(self, device_id: str) -> bool:
        if not device_id:
            return False
        if self.device_id is None:
            self.device_id = device_id
        return self.device_id == device_id

    def invalidate(
        self, reason: str, *, wall_time: float, reset_clock: bool = False
    ) -> TrackingUpdate:
        if self._available:
            self._generation += 1
        self._available = False
        self._release_required = True
        self.reason = reason
        self._last_body_at = None
        if reset_clock:
            self._anchor = None
            self._last_packet_ns = None
            self._joint_times = None
        return TrackingUpdate(
            BodyTrackingSnapshot(
                type="body_tracking_snapshot",
                capture_time_s=wall_time,
                frame_id=self.frame_id,
                joints=None,
            ),
            Buttons(),
            Twist.zero(),
        )

    def receive(
        self,
        device_id: str,
        packet: PicoPacket,
        *,
        now: float,
        wall_time: float,
    ) -> TrackingUpdate | None:
        if not self.accepts(device_id):
            return None
        left, right = packet.controllers.left, packet.controllers.right
        if left.primary and left.secondary and right.primary and right.secondary:
            # Stop buttons must survive focus/body loss and the reconnect
            # release gate. Consumers own the stop semantics and latch.
            return TrackingUpdate(
                None,
                Buttons.from_controllers(
                    left.controller_state(is_left=True), right.controller_state(is_left=False)
                ),
                Twist.zero(),
            )
        expired = self.expire(now=now, wall_time=wall_time)
        if expired is not None:
            return expired
        self.packets += 1
        timestamp = packet.timestamp_ns
        if self._last_packet_ns is not None:
            if timestamp < self._last_packet_ns:
                return self.invalidate("packet_clock_reset", wall_time=wall_time, reset_clock=True)
            if timestamp == self._last_packet_ns:
                return self.expire(now=now, wall_time=wall_time)
        self._last_packet_ns = timestamp
        self._last_packet_at = now
        if self._anchor is None:
            self._anchor = (timestamp, now, wall_time)
        source_start, mono_start, wall_start = self._anchor
        elapsed = (timestamp - source_start) / 1e9
        lag = now - mono_start - elapsed
        if lag < -self.stale_timeout:
            return self.invalidate("packet_clock_jump", wall_time=wall_time, reset_clock=True)
        if lag > self.stale_timeout:
            return self.invalidate("delayed_packets", wall_time=wall_time)
        if not packet.app_state.focus:
            return self.invalidate("app_not_focused", wall_time=wall_time)
        if packet.body is None:
            return self.invalidate("body_missing", wall_time=wall_time)

        joints = dict(
            zip(PICO_JOINT_NAMES, (joint.pose() for joint in packet.body.joints), strict=True)
        )
        joint_times = tuple(joint.t for joint in packet.body.joints)
        clock_indices = tuple(i for i, timestamp in enumerate(joint_times) if timestamp > 0)
        if not clock_indices:
            return self.invalidate("body_clock_missing", wall_time=wall_time)
        previous = self._joint_times
        previous_indices = (
            ()
            if previous is None
            else tuple(i for i, timestamp in enumerate(previous) if timestamp > 0)
        )
        if previous is None or clock_indices != previous_indices:
            self._joint_times = joint_times
            return self.invalidate(
                "waiting_for_body_clock" if previous is None else "body_clock_source_changed",
                wall_time=wall_time,
            )
        if any(joint_times[i] < previous[i] for i in clock_indices):
            self._joint_times = joint_times
            return self.invalidate("body_clock_reset", wall_time=wall_time)

        body = None
        if all(joint_times[i] > previous[i] for i in clock_indices):
            self._joint_times = joint_times
            self._last_body_at = now
            self._available = True
            self.frames += 1
            body = BodyTrackingSnapshot(
                type="body_tracking_snapshot",
                capture_time_s=wall_start + elapsed,
                frame_id=self.frame_id,
                joints=joints,
            )
        expired = self.expire(now=now, wall_time=wall_time)
        if expired is not None:
            return expired
        if not self._available:
            return None

        left, right = packet.controllers.left, packet.controllers.right
        if not left.primary and not right.primary:
            self._release_required = False
        buttons = Buttons.from_controllers(
            left.controller_state(is_left=True),
            right.controller_state(is_left=False),
        )
        buttons.pack_analog_triggers(left.trigger, right.trigger)
        if self._release_required:
            buttons.left_primary = False
            buttons.right_primary = False
        self.reason = "release_a_and_x" if self._release_required else "tracking"
        forward = left.axis_y if abs(left.axis_y) >= self.deadzone else 0.0
        turn = right.axis_x if abs(right.axis_x) >= self.deadzone else 0.0
        # Unity's native stick Y is positive forward (WebXR gamepad Y is negative).
        velocity = (
            Twist.zero()
            if right.axis_click or self._release_required
            else Twist(
                linear=Vector3(forward * self.linear_scale, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, -turn * self.yaw_scale),
            )
        )
        return TrackingUpdate(body, buttons, velocity)

    def expire(self, *, now: float, wall_time: float) -> TrackingUpdate | None:
        if self._available and self._last_body_at is not None:
            if now - self._last_body_at >= self.stale_timeout:
                return self.invalidate("body_stale", wall_time=wall_time)
        return None

    def status(self, *, now: float) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "state": self.reason,
            "body_available": self._available,
            "release_required": self._release_required,
            "body_frames": self.frames,
            "packets": self.packets,
            "body_age_ms": None
            if self._last_body_at is None
            else (now - self._last_body_at) * 1000,
            "packet_age_ms": None
            if self._last_packet_at is None
            else (now - self._last_packet_at) * 1000,
            "packet_timestamp_ns": self._last_packet_ns,
            "body_timestamp": next((t for t in self._joint_times or () if t > 0), None),
            "body_clock_joints": [
                PICO_JOINT_NAMES[i]
                for i, timestamp in enumerate(self._joint_times or ())
                if timestamp > 0
            ],
            "frame_id": self.frame_id,
        }
