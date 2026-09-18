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
"""Reactive text-only agent: world state in, joystick-style cmd_vel out.

Each tick folds the latest pose, detections and scan into one JSON state
(world_state.py), sends it with the drive questions to TypeSafe in one POST,
and the adapter (drive.py) turns the typed answers into a Twist. The model
picks directions; code sets magnitudes from the target's bearing and distance,
ramps the Twist at 10 Hz and zeroes it on a deadman.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Generic, TypeVar

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from reactivex.disposable import Disposable
import requests

from dimos.agents.typesafe.drive import Answers, Drive, Question, decode, questions
from dimos.agents.typesafe.world_state import WorldState, build_world_state
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

API_KEY_ENV = "TYPESAFE_API_KEY"
T = TypeVar("T")
Vec3 = tuple[float, float, float]
Ask = Callable[[WorldState, dict[str, Question]], Answers]
ZERO: Vec3 = (0.0, 0.0, 0.0)
PUBLISH_HZ = 10.0
# ponytail: fixed steering gains; config them if a robot needs different ones.
SLOW_WITHIN_M = 1.5
TURN_FULL_AT_DEG = 45.0


def typesafe_api_key() -> str | None:
    if os.environ.get(API_KEY_ENV):
        return None
    return f"{API_KEY_ENV} is not set. Create a key at https://console.typesafe.ai/settings/keys"


class _Latest(Generic[T]):
    def __init__(self) -> None:
        self._item: tuple[float, T] | None = None

    def put(self, msg: T) -> None:
        self._item = (time.monotonic(), msg)

    def get(self, max_age_s: float) -> T | None:
        item = self._item
        return item[1] if item and time.monotonic() - item[0] <= max_age_s else None


class TypeSafeAgentConfig(ModuleConfig):
    model: str = "jev-latest"
    rate_hz: float = 2.0
    timeout_s: float = 5.0
    stale_s: float = 2.0
    deadman_s: float | None = None  # None: max(1, 2.5 / rate_hz), must outlast one slow inference
    linear_speed: float = 0.5
    angular_speed: float = 0.8
    linear_accel: float = 0.8
    angular_accel: float = 1.6
    min_confidence: float = 0.5
    stop_threshold: float = 0.7
    reached_m: float = 0.5
    give_up_s: float = 5.0  # goal clears after this long without motion
    image_size: tuple[int, int] = (1280, 720)  # for 2D detections
    lidar_band: tuple[float, float, float] = (-0.2, 0.8, 5.0)  # z_min, z_max, max_range
    trace_dir: Path | None = None  # raw request/response pairs, for debugging


class TypeSafeAgent(Module):
    config: TypeSafeAgentConfig

    odom: In[PoseStamped]
    odometry: In[Odometry]  # same pose, nav_msgs flavour (habitat)
    detections_3d: In[Detection3DArray]
    detections_2d: In[Detection2DArray]
    lidar: In[PointCloud2]
    human_input: In[str]

    cmd_vel: Out[Twist]
    goal: Out[PointStamped]  # world-frame XY the goal text resolved to
    agent: Out[BaseMessage]
    agent_idle: Out[bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pose: _Latest[PoseStamped] = _Latest()
        self._det3d: _Latest[Detection3DArray] = _Latest()
        self._det2d: _Latest[Detection2DArray] = _Latest()
        self._lidar: _Latest[PointCloud2] = _Latest()
        self._ask: Ask = self._post
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._goal: str | None = None
        self._goal_xy: tuple[float, float] | None = None
        self._motion = "idle"
        self._target: Vec3 = ZERO
        self._current: Vec3 = ZERO
        self._decided_at = 0.0
        self._zero_since: float | None = None
        self._last_said = ""
        self._seq = 0
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    @rpc
    def start(self) -> None:
        super().start()
        self._session.headers["Authorization"] = f"Bearer {os.environ.get(API_KEY_ENV, '')}"
        self.register_disposable(Disposable(self.odom.subscribe(self._pose.put)))
        self.register_disposable(Disposable(self.odometry.subscribe(self._on_odometry)))
        self.register_disposable(Disposable(self.detections_3d.subscribe(self._det3d.put)))
        self.register_disposable(Disposable(self.detections_2d.subscribe(self._det2d.put)))
        self.register_disposable(Disposable(self.lidar.subscribe(self._lidar.put)))
        self.register_disposable(Disposable(self.human_input.subscribe(self.set_goal)))
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._infer_loop, name="TypeSafeAgent-infer", daemon=True),
            threading.Thread(target=self._publish_loop, name="TypeSafeAgent-publish", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self.agent_idle.publish(True)

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self.cmd_vel.publish(Twist.zero())
        self._session.close()
        super().stop()

    @rpc
    def set_goal(self, goal: str | None) -> None:
        goal = (goal or "").strip() or None
        with self._lock:
            self._goal = goal
            self._goal_xy = None
            self._target = self._current = ZERO
            self._zero_since = None
            self._motion = "idle"
        if goal is None:
            self.cmd_vel.publish(Twist.zero())
        else:
            self.agent.publish(HumanMessage(content=goal))
        self.agent_idle.publish(goal is None)

    @rpc
    def current_goal(self) -> str | None:
        return self._goal

    def _post(self, state: WorldState, qs: dict[str, Question]) -> Answers:
        """TypeSafe System One: `POST /v1/systemone`. A failed tick zeroes the target; the next tick retries."""
        r = self._session.post(
            os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone",
            json={"state": state, "model": self.config.model, "questions": qs},
            timeout=self.config.timeout_s,
        )
        r.raise_for_status()
        answers: Answers = r.json()["answers"]
        return answers

    def _on_odometry(self, o: Odometry) -> None:
        self._pose.put(
            PoseStamped(
                ts=o.ts, frame_id=o.frame_id, position=o.position, orientation=o.orientation
            )
        )

    def _say(self, text: str) -> None:
        if text != self._last_said:
            self._last_said = text
            logger.info(text)
            self.agent.publish(AIMessage(content=text))

    def _infer_loop(self) -> None:
        period = 1.0 / self.config.rate_hz
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception:
                logger.exception("TypeSafeAgent tick failed")
                self._set_target(ZERO)
            self._stop_event.wait(max(0.0, period - (time.monotonic() - t0)))

    def _tick(self) -> None:
        goal = self._goal
        if goal is None:
            return
        stale = self.config.stale_s
        pose, det3d, det2d = self._pose.get(stale), self._det3d.get(stale), self._det2d.get(stale)
        if pose is None or (det3d is None and det2d is None and self._goal_xy is None):
            self._set_target(ZERO)
            self._say("holding: no odom" if pose is None else "holding: no detections")
            return
        state = build_world_state(
            goal,
            pose,
            self._motion,
            detections_3d=det3d,
            detections_2d=det2d,
            lidar=self._lidar.get(stale),
            goal_xy=self._goal_xy,
            image_size=self.config.image_size,
            lidar_band=self.config.lidar_band,
        )
        qs = questions(tuple(dict.fromkeys(o["label"] for o in state["objects"])))
        answers = self._ask(state, qs)
        self._trace(state, qs, answers)
        drive = decode(
            answers,
            min_confidence=self.config.min_confidence,
            stop_threshold=self.config.stop_threshold,
        )
        self._steer(state, drive)

    def _steer(self, state: WorldState, drive: Drive) -> None:
        """The model picked directions; geometry to the goal point sets the magnitudes."""
        target = next((o for o in state["objects"] if o["label"] == drive.target), None)
        if target is not None and "position" in target:
            self._resolve_goal(target["position"]["x"], target["position"]["y"])
            dist, err = target.get("distance_m", 0.0), abs(target.get("bearing_deg", 0.0))
        elif "goal_point" in state:
            dist, err = state["goal_point"]["distance_m"], abs(state["goal_point"]["bearing_deg"])
        else:
            dist, err = None, 0.0
        lin, ang = self.config.linear_speed, self.config.angular_speed
        if dist is not None:
            if dist <= self.config.reached_m:
                drive = Drive(0.0, 0.0, 0.0, True, drive.confidence, drive.labels, drive.target)
            lin *= min(1.0, max(0.3, dist / SLOW_WITHIN_M))
            ang *= min(1.0, max(0.25, err / TURN_FULL_AT_DEG))
        self._set_target((drive.x * lin, drive.y * lin, drive.yaw * ang), immediate=drive.stop)
        with self._lock:
            self._motion = "stopped" if drive.is_zero else "driving"
            gave_up = (
                self._zero_since is not None
                and time.monotonic() - self._zero_since > self.config.give_up_s
            )
        self._say(
            f"drive {'/'.join(drive.labels)} stop={drive.stop} target={drive.target} confidence={drive.confidence:.2f}"
        )
        if gave_up:
            self.set_goal(None)
            self._say("goal reached or unreachable; stopped")

    def _resolve_goal(self, x: float, y: float) -> None:
        """Latch the target's world XY; publish it when it moves more than 10 cm."""
        with self._lock:
            prev = self._goal_xy
            self._goal_xy = (x, y)
        if prev is None or abs(prev[0] - x) > 0.1 or abs(prev[1] - y) > 0.1:
            self.goal.publish(PointStamped(x, y, 0.0, frame_id="world"))

    def _set_target(self, target: Vec3, *, immediate: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            self._target = target
            self._decided_at = now
            if immediate:
                self._current = ZERO
            self._zero_since = (self._zero_since or now) if target == ZERO else None
        if immediate:
            self.cmd_vel.publish(Twist.zero())

    def _publish_loop(self) -> None:
        dt = 1.0 / PUBLISH_HZ
        steps = (
            self.config.linear_accel * dt,
            self.config.linear_accel * dt,
            self.config.angular_accel * dt,
        )
        deadman = self.config.deadman_s or max(1.0, 2.5 / self.config.rate_hz)
        while not self._stop_event.is_set():
            with self._lock:
                stale = time.monotonic() - self._decided_at > deadman
                target = ZERO if self._goal is None or stale else self._target
                cur = self._current
                nxt = tuple(
                    c + max(-s, min(s, t - c)) for c, t, s in zip(cur, target, steps, strict=True)
                )
                self._current = (nxt[0], nxt[1], nxt[2])
                publish = self._goal is not None or any(cur)
            if publish:
                self.cmd_vel.publish(
                    Twist(linear=(nxt[0], nxt[1], 0.0), angular=(0.0, 0.0, nxt[2]))
                )
            self._stop_event.wait(dt)

    def _trace(self, state: WorldState, qs: dict[str, Question], answers: Answers) -> None:
        if self.config.trace_dir is None:
            return
        d = Path(self.config.trace_dir)
        d.mkdir(parents=True, exist_ok=True)
        self._seq += 1
        (d / f"{self._seq}-request.json").write_text(
            json.dumps({"body": {"state": state, "questions": qs}})
        )
        (d / f"{self._seq}-response.json").write_text(json.dumps({"body": {"answers": answers}}))
