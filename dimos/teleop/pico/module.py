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

"""Native PICO full-body input for the existing DimOS SONIC teleop task."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path
import threading
import time
from typing import Any

import grpc
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.teleop.pico.client import PCServiceClient
from dimos.teleop.pico.install import ensure_pc_service
from dimos.teleop.pico.proto.tracking_pb2 import ServerFeedback
from dimos.teleop.pico.service import pc_service_lifecycle
from dimos.teleop.pico.tracking import PicoTrackingSession, TrackingUpdate, parse_packet
from dimos.teleop.webxr.body_tracking import BodyTrackingSnapshot
from dimos.teleop.webxr.controller_types import Buttons
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class PicoTeleopConfig(ModuleConfig):
    manage_pc_service: bool = True
    pc_service_dir: Path | None = None
    pc_service_start_timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    device_id: str | None = None
    stale_timeout: float = Field(default=1.0, gt=0, le=1.0)
    reconnect_interval: float = Field(default=1.0, gt=0)
    linear_scale: float = Field(default=0.3, ge=0, allow_inf_nan=False)
    linear_min_speed: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    yaw_scale: float = Field(default=0.3, ge=0, allow_inf_nan=False)
    deadzone: float = Field(default=0.18, ge=0, lt=1)


class PicoTeleopModule(Module):
    """Consume XRoboToolkit's native tracking stream; no browser is involved.

    Own the local XRoboToolkit PC service, then consume Body and Controller
    sending from its PICO APK. See this directory's README for setup.
    """

    config: PicoTeleopConfig
    body_tracking: Out[BodyTrackingSnapshot]
    teleop_buttons: Out[Buttons]
    cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tracking = PicoTrackingSession(
            device_id=self.config.device_id,
            stale_timeout=self.config.stale_timeout,
            linear_scale=self.config.linear_scale,
            linear_min_speed=self.config.linear_min_speed,
            yaw_scale=self.config.yaw_scale,
            deadzone=self.config.deadzone,
        )
        self._lock = threading.Lock()
        self._connected = False
        self._last_logged_state = ""
        self._last_error: str | None = None
        self._service: asyncio.subprocess.Process | None = None

    @property
    def _manages_local_service(self) -> bool:
        g = self.config.g
        return (
            self.config.manage_pc_service
            and g.xrobotoolkit_host in {"localhost", "127.0.0.1"}
            and g.xrobotoolkit_port == GlobalConfig.model_fields["xrobotoolkit_port"].default
        )

    @rpc
    def build(self) -> None:
        super().build()
        if self._manages_local_service:
            ensure_pc_service(self.config.pc_service_dir)

    @rpc
    def tracking_status(self) -> dict[str, Any]:
        """Return service connection, tracking freshness and input readiness."""
        with self._lock:
            return {
                "pc_service_connected": self._connected,
                "pc_service_pid": self._service.pid if self._service is not None else None,
                "pc_service_exit_code": (
                    self._service.returncode if self._service is not None else None
                ),
                "last_error": self._last_error,
                **self._tracking.status(now=time.monotonic()),
            }

    async def main(self) -> AsyncIterator[None]:
        g = self.config.g
        client = PCServiceClient(f"{g.xrobotoolkit_host}:{g.xrobotoolkit_port}")
        async with pc_service_lifecycle(
            client,
            manage=self._manages_local_service,
            directory=self.config.pc_service_dir,
            startup_timeout=self.config.pc_service_start_timeout,
        ) as service:
            with self._lock:
                self._service = service
            receive_task = asyncio.create_task(self._receive(client), name="pico-receive")
            monitor_task = asyncio.create_task(self._monitor(), name="pico-monitor")
            try:
                yield
            finally:
                receive_task.cancel()
                monitor_task.cancel()
                await asyncio.gather(receive_task, monitor_task, return_exceptions=True)
                self._invalidate("stopped", reset_clock=True)

    def _publish(self, update: TrackingUpdate | None) -> None:
        if update is not None:
            if update.body is not None:
                self.body_tracking.publish(update.body)
            self.teleop_buttons.publish(update.buttons)
            self.cmd_vel.publish(update.cmd_vel)

    def _invalidate(self, reason: str, *, reset_clock: bool = False) -> None:
        with self._lock:
            update = self._tracking.invalidate(
                reason,
                wall_time=time.time(),
                reset_clock=reset_clock,
            )
        self._publish(update)

    def _on_feedback(self, feedback: ServerFeedback) -> None:
        if feedback.name == "deviceMissing":
            with self._lock:
                selected = self._tracking.device_id == feedback.devid
            if selected:
                self._invalidate("headset_disconnected", reset_clock=True)
        elif feedback.name == "deviceStateJson":
            data = feedback.devicestatejson
            with self._lock:
                if not self._tracking.accepts(data.devid):
                    return
            try:
                packet = parse_packet(data.statejson)
                with self._lock:
                    update = self._tracking.receive(
                        data.devid,
                        packet,
                        now=time.monotonic(),
                        wall_time=time.time(),
                    )
                    self._last_error = None
            except ValueError as exc:
                with self._lock:
                    self._last_error = str(exc)[:500]
                self._invalidate("invalid_packet")
                return
            self._publish(update)

    async def _receive(self, client: PCServiceClient) -> None:
        while True:
            try:
                async with aclosing(client.events()) as events:
                    async for feedback in events:
                        if feedback is None:
                            with self._lock:
                                self._connected = True
                                self._last_error = None
                            self._invalidate("waiting_for_headset", reset_clock=True)
                        else:
                            self._on_feedback(feedback)
            except (grpc.aio.AioRpcError, ConnectionError) as exc:
                with self._lock:
                    self._last_error = str(exc)[:500]
            finally:
                with self._lock:
                    self._connected = False
                self._invalidate("pc_service_disconnected", reset_clock=True)
            await asyncio.sleep(self.config.reconnect_interval)

    async def _monitor(self) -> None:
        next_log = 0.0
        while True:
            now = time.monotonic()
            with self._lock:
                update = self._tracking.expire(now=now, wall_time=time.time())
            self._publish(update)
            status = self.tracking_status()
            state = str(status["state"])
            if state != self._last_logged_state or now >= next_log:
                log = logger.warning if status["last_error"] is not None else logger.info
                log("PICO tracking", **status)
                self._last_logged_state = state
                next_log = now + 5.0
            await asyncio.sleep(min(0.1, self.config.stale_timeout / 2))


# Source-only diagnostics: inspect tracking without starting a robot or simulator.
teleop_pico_body_tracking = PicoTeleopModule.blueprint()
