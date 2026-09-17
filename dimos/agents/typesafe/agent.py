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
"""Reactive text-only agent: WorldState in, joystick-style cmd_vel out."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from reactivex.disposable import Disposable

from dimos.agents.typesafe.client import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SystemOneClient,
)
from dimos.agents.typesafe.drive import Drive, decode_drive, drive_questions
from dimos.agents.typesafe.world_state import build_world_state
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_STOP_WORDS = {"stop", "halt", "stop.", "halt."}


def typesafe_api_key() -> str | None:
    if os.environ.get(API_KEY_ENV):
        return None
    return f"{API_KEY_ENV} is not set. Create a key at https://console.typesafe.ai/settings/keys"


class TypeSafeAgentConfig(ModuleConfig):
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = API_KEY_ENV
    rate_hz: float = 2.0
    publish_hz: float = 10.0
    timeout_s: float = 5.0
    stale_s: float = 2.0
    # None -> max(1.0, 2.5 / rate_hz): the deadman must outlast one slow inference.
    deadman_s: float | None = None
    linear_speed: float = 0.5
    angular_speed: float = 0.8
    linear_accel: float = 0.8
    angular_accel: float = 1.6
    min_confidence: float = 0.5
    stop_threshold: float = 0.7
    blend: bool = False
    stops_to_clear_goal: int = 3
    max_objects: int = 20
    image_width: int = 1280
    image_height: int = 720
    lidar_z_min: float = -0.2
    lidar_z_max: float = 0.8
    lidar_max_range: float = 5.0
    trace_dir: Path | None = None


class TypeSafeAgent(Module):
    config: TypeSafeAgentConfig

    odom: In[PoseStamped]
    detections_3d: In[Detection3DArray]
    detections_2d: In[Detection2DArray]
    lidar: In[PointCloud2]
    human_input: In[str]

    cmd_vel: Out[Twist]
    agent: Out[BaseMessage]
    agent_idle: Out[bool]
    world_state: Out[str]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[float, Any]] = {}
        self._goal: str | None = None
        self._robot: dict[str, Any] = {"motion": "idle"}
        self._questions = drive_questions()
        self._client: SystemOneClient | None = None
        self._target = (0.0, 0.0, 0.0)
        self._current = (0.0, 0.0, 0.0)
        self._last_decision_ts = 0.0
        self._last_labels: tuple[str, ...] = ()
        self._stop_streak = 0
        self._idle_published: bool | None = None
        self._seq = 0
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def deadman_s(self) -> float:
        return (
            self.config.deadman_s
            if self.config.deadman_s is not None
            else max(1.0, 2.5 / self.config.rate_hz)
        )

    @rpc
    def start(self) -> None:
        super().start()
        self._client = SystemOneClient(
            os.environ.get(self.config.api_key_env),
            model=self.config.model,
            base_url=self.config.base_url,
            timeout_s=self.config.timeout_s,
        )
        for name in ("odom", "detections_3d", "detections_2d", "lidar"):
            stream = getattr(self, name)
            self.register_disposable(Disposable(stream.subscribe(self._store(name))))
        self.register_disposable(Disposable(self.human_input.subscribe(self._on_human_input)))
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._infer_loop, name="TypeSafeAgent-infer", daemon=True),
            threading.Thread(target=self._publish_loop, name="TypeSafeAgent-publish", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._publish_idle(True)

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._threads = []
        self.cmd_vel.publish(Twist.zero())
        if self._client is not None:
            self._client.close()
            self._client = None
        super().stop()

    @rpc
    def set_goal(self, goal: str | None) -> None:
        self._on_human_input(goal or "")

    @rpc
    def goal(self) -> str | None:
        return self._goal

    @rpc
    def robot_state(self) -> dict[str, Any]:
        return dict(self._robot)

    def _store(self, name: str) -> Any:
        def cb(msg: Any) -> None:
            with self._lock:
                self._latest[name] = (time.monotonic(), msg)

        return cb

    def _fresh(self, name: str) -> Any:
        with self._lock:
            item = self._latest.get(name)
        if item is None or time.monotonic() - item[0] > self.config.stale_s:
            return None
        return item[1]

    def _on_human_input(self, text: str) -> None:
        text = (text or "").strip()
        self.agent.publish(HumanMessage(content=text))
        if not text or text.lower() in _STOP_WORDS:
            self._clear_goal("stopped by operator" if text else "goal cleared")
            return
        with self._lock:
            self._goal = text
            self._stop_streak = 0
            self._robot["motion"] = "idle"
        self._publish_idle(False)

    def _clear_goal(self, reason: str) -> None:
        with self._lock:
            self._goal = None
            self._target = (0.0, 0.0, 0.0)
            self._current = (0.0, 0.0, 0.0)
            self._robot["motion"] = "idle"
            self._last_labels = ()
        self.cmd_vel.publish(Twist.zero())
        self.agent.publish(AIMessage(content=reason))
        self._publish_idle(True)

    def _publish_idle(self, idle: bool) -> None:
        if self._idle_published != idle:
            self._idle_published = idle
            self.agent_idle.publish(idle)

    def _infer_loop(self) -> None:
        period = 1.0 / self.config.rate_hz
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception:
                logger.exception("TypeSafeAgent tick failed")
                with self._lock:
                    self._target = (0.0, 0.0, 0.0)
            self._stop_event.wait(max(0.0, period - (time.monotonic() - t0)))

    def _tick(self) -> None:
        goal = self._goal
        if not goal or self._client is None:
            return
        pose = self._fresh("odom")
        state = build_world_state(
            goal=goal,
            pose=pose,
            detections_3d=self._fresh("detections_3d"),
            detections_2d=self._fresh("detections_2d"),
            lidar=self._fresh("lidar"),
            robot=self._robot,
            max_objects=self.config.max_objects,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            lidar_kwargs={
                "z_min": self.config.lidar_z_min,
                "z_max": self.config.lidar_z_max,
                "max_range": self.config.lidar_max_range,
            },
        )
        self.world_state.publish(json.dumps(state))
        answers = self._client.system_one(state, self._questions)
        self._trace(state, answers)
        self._apply(
            decode_drive(
                answers,
                min_confidence=self.config.min_confidence,
                blend=self.config.blend,
                stop_threshold=self.config.stop_threshold,
            )
        )

    def _apply(self, drive: Drive) -> None:
        lin, ang = self.config.linear_speed, self.config.angular_speed
        labels = (*drive.labels, "stop" if drive.stop else "go")
        with self._lock:
            self._target = (drive.x * lin, drive.y * lin, drive.yaw * ang)
            self._last_decision_ts = time.monotonic()
            self._robot["motion"] = "stopped" if drive.is_zero else "driving"
            self._robot["last_drive"] = {
                "x": drive.labels[0],
                "y": drive.labels[1],
                "yaw": drive.labels[2],
            }
            changed = labels != self._last_labels
            self._last_labels = labels
            if drive.stop:
                self._current = (0.0, 0.0, 0.0)
                self._stop_streak += 1
            else:
                self._stop_streak = 0
            streak = self._stop_streak
        if drive.stop:
            self.cmd_vel.publish(Twist.zero())
        if changed:
            logger.info("drive", labels=labels, confidence=round(drive.confidence, 2))
            self.agent.publish(
                AIMessage(
                    content=f"drive x={drive.labels[0]} y={drive.labels[1]} yaw={drive.labels[2]} stop={drive.stop} confidence={drive.confidence:.2f}",
                    additional_kwargs={
                        "typesafe": {
                            "x": drive.x,
                            "y": drive.y,
                            "yaw": drive.yaw,
                            "stop": drive.stop,
                            "confidence": drive.confidence,
                        }
                    },
                )
            )
        if streak >= self.config.stops_to_clear_goal:
            self._clear_goal("goal reached or unreachable; stopped")

    def _publish_loop(self) -> None:
        dt = 1.0 / self.config.publish_hz
        step_lin, step_ang = self.config.linear_accel * dt, self.config.angular_accel * dt
        while not self._stop_event.is_set():
            with self._lock:
                target = self._target
                if self._goal is None:
                    target = (0.0, 0.0, 0.0)
                elif time.monotonic() - self._last_decision_ts > self.deadman_s:
                    target = (0.0, 0.0, 0.0)
                cur = self._current
                nxt = tuple(
                    c + max(-s, min(s, t - c))
                    for c, t, s in zip(cur, target, (step_lin, step_lin, step_ang), strict=True)
                )
                self._current = (nxt[0], nxt[1], nxt[2])
                publish = self._goal is not None or any(cur)
            if publish:
                self.cmd_vel.publish(
                    Twist(linear=(nxt[0], nxt[1], 0.0), angular=(0.0, 0.0, nxt[2]))
                )
            self._stop_event.wait(dt)

    def _trace(self, state: dict[str, Any], answers: dict[str, Any]) -> None:
        if self.config.trace_dir is None:
            return
        d = Path(self.config.trace_dir)
        d.mkdir(parents=True, exist_ok=True)
        self._seq += 1
        (d / f"{self._seq}-request.json").write_text(
            json.dumps({"body": {"state": state, "questions": self._questions}})
        )
        (d / f"{self._seq}-response.json").write_text(json.dumps({"body": {"answers": answers}}))
