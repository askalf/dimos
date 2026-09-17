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

"""TypeSafe (Jev) as a closed-loop navigation policy.

The model calls no skill. Each tick: code assembles the world state, Jev picks
one world-frame unit step, code rotates it into the body frame and publishes a
single Twist. The scene (obstacles, room bounds, goal) is a static JSON file;
the robot pose comes live off ``/odom``.

Two contracts are meant to be edited: :class:`WorldState` (what Jev sees) and
:data:`STEP_CRITERIA` / :func:`build_questions` (what Jev answers). Everything
else is plumbing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any

from dimos.evals.agents.base import Agent, AgentConfig
from dimos.evals.agents.lib.trajectory_builder import TrajectoryBuilder
from dimos.evals.types import EndedBy, Metrics, RunningEnvironment, Trajectory
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist

if TYPE_CHECKING:
    from dimos.evals.environments.base import Environment


# --- input contract: what Jev sees each tick ---------------------------------
# One global world frame (ROS: +x east, +y north, meters). Obstacles carry a
# label and an axis-aligned 2D box; there is no object/obstacle distinction,
# every box is something to avoid.
#
# Note: Jev is documented as unreliable at numeric comparison
# (docs.typesafe.ai/model-jaggedness/jev-1.13). Raw coordinates are a
# deliberate choice; the scripted baselines are what tell us whether it works.


@dataclass(frozen=True, kw_only=True)
class Obstacle:
    label: str
    min_xy: tuple[float, float]
    max_xy: tuple[float, float]


@dataclass(frozen=True, kw_only=True)
class WorldState:
    frame_id: str = "world"  # the global origin everything is expressed in
    robot_xy: tuple[float, float] = (0.0, 0.0)
    robot_yaw_deg: float = 0.0
    goal_label: str = ""
    goal_xy: tuple[float, float] = (0.0, 0.0)
    room_bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    obstacles: list[Obstacle] = field(default_factory=list)
    ticks_elapsed: int = 0

    def encode(self) -> dict[str, Any]:
        """The ``state`` payload sent to system_one."""
        return asdict(self)


# --- output contract: what Jev answers ---------------------------------------
# Options are world-frame unit steps keyed by their literal coordinates. Code
# rotates the pick into the body frame, so the model never reasons about heading.

STEPS: dict[str, tuple[float, float]] = {
    "0,0": (0.0, 0.0),
    "0,1": (0.0, 1.0),
    "1,0": (1.0, 0.0),
    "0,-1": (0.0, -1.0),
    "-1,0": (-1.0, 0.0),
}

STEP_CRITERIA: dict[str, str] = {
    "0,0": "Hold position.",
    "0,1": "Step north, toward +y.",
    "1,0": "Step east, toward +x.",
    "0,-1": "Step south, toward -y.",
    "-1,0": "Step west, toward -x.",
}


def build_questions() -> dict[str, Any]:
    """One fan-out call per tick: the step plus the termination flag."""
    from typesafe_sdk import Choice, Noul

    return {
        "step": Choice(
            instructions=(
                "Which unit step moves the robot toward the goal without entering an obstacle box"
            ),
            criteria=STEP_CRITERIA,
        ),
        "reached": Noul(instructions="The robot has reached the goal and should stop"),
    }


# --- scene file --------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Scene:
    """The static half of the world state, supplied as JSON::

    {
      "frame_id": "world",
      "goal": {"label": "bed", "xy": [-3.567, -1.332]},
      "room_bounds": [-6.0, -4.0, 6.0, 4.0],
      "obstacles": [
        {"label": "sofa", "min_xy": [1.0, 0.5], "max_xy": [2.5, 1.8]}
      ]
    }
    """

    frame_id: str
    goal_label: str
    goal_xy: tuple[float, float]
    room_bounds: tuple[float, float, float, float]
    obstacles: list[Obstacle]


def load_scene(path: Path) -> Scene:
    raw = json.loads(Path(path).expanduser().read_text())
    goal = raw["goal"]
    return Scene(
        frame_id=str(raw.get("frame_id", "world")),
        goal_label=str(goal["label"]),
        goal_xy=tuple(goal["xy"]),  # type: ignore[arg-type]
        room_bounds=tuple(raw["room_bounds"]),  # type: ignore[arg-type]
        obstacles=[
            Obstacle(
                label=str(o["label"]),
                min_xy=tuple(o["min_xy"]),  # type: ignore[arg-type]
                max_xy=tuple(o["max_xy"]),  # type: ignore[arg-type]
            )
            for o in raw.get("obstacles", [])
        ],
    )


# --- the agent ---------------------------------------------------------------


class TypeSafePolicyConfig(AgentConfig):
    model: str = "jev-1.13"
    scene_json: Path = Path()  # required; see Scene for the schema
    max_ticks: int = 60
    tick_s: float = 1.0
    speed: float = 0.4
    min_confidence: float = 0.35  # below this, hold still
    reached_noul: float = 0.8
    odom_topic: str = "/odom"
    # MovementManager owns /cmd_vel in the go2 stack; publish upstream of it.
    cmd_topic: str = "/nav_cmd_vel"


class TypeSafePolicy(Agent):
    """Drive the robot with Jev decisions. Publishes Twist, calls no skill."""

    config: TypeSafePolicyConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pose: PoseStamped | None = None
        self._pose_seen = threading.Event()

    def preflight(self, environment: Environment) -> None:
        if self.config.modules:
            raise ValueError("TypeSafePolicy calls no tools; leave modules empty")
        if not self.config.scene_json.is_file():
            raise FileNotFoundError(f"scene_json not found: {self.config.scene_json}")
        load_scene(self.config.scene_json)  # fail before anything starts

    def run(
        self, inputs: str, env: RunningEnvironment, run_dir: Path, *, timeout_s: float
    ) -> Trajectory:
        from typesafe_sdk import TypeSafeClient

        from dimos.core.transport_factory import make_transport

        raw = run_dir / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        trajectory = TrajectoryBuilder(inputs, name=type(self).__name__, model=self.config.model)

        scene = load_scene(self.config.scene_json)
        client = TypeSafeClient()
        questions = build_questions()

        odom = make_transport(self.config.odom_topic, PoseStamped)
        cmd = make_transport(self.config.cmd_topic, Twist)
        for transport in (odom, cmd):
            transport.start()
        odom.subscribe(self._on_odom)

        deadline = time.monotonic() + timeout_s
        ended: EndedBy = "max_steps"
        try:
            if not self._pose_seen.wait(min(10.0, timeout_s)):
                raise TimeoutError(f"no pose on {self.config.odom_topic}")
            for tick in range(self.config.max_ticks):
                if time.monotonic() >= deadline:
                    ended = "timeout"
                    break

                pose, state = self.observe(scene, tick)
                started = time.time()
                response = client.system_one(state=state.encode(), questions=questions)
                self._trace(trajectory, raw, tick, state, response, started)

                if response.answers["reached"].noul > self.config.reached_noul:
                    ended = "answer"
                    break

                # Hold the commanded twist for the whole tick (open loop).
                cmd.publish(self.twist(response.answers["step"], pose))
                time.sleep(self.config.tick_s)
        finally:
            cmd.publish(Twist.zero())
            for transport in (odom, cmd):
                transport.stop()
        return trajectory.build(ended)

    def _on_odom(self, pose: PoseStamped) -> None:
        self._pose = pose
        self._pose_seen.set()

    def observe(self, scene: Scene, tick: int) -> tuple[PoseStamped, WorldState]:
        """Static scene + live pose -> the state Jev sees."""
        pose = self._pose
        if pose is None:
            raise LookupError("no pose yet")
        return pose, WorldState(
            frame_id=scene.frame_id,
            robot_xy=(pose.position.x, pose.position.y),
            robot_yaw_deg=math.degrees(_yaw(pose)),
            goal_label=scene.goal_label,
            goal_xy=scene.goal_xy,
            room_bounds=scene.room_bounds,
            obstacles=scene.obstacles,
            ticks_elapsed=tick,
        )

    def twist(self, step: Any, pose: PoseStamped) -> Twist:
        """World-frame unit step -> body-frame Twist. cmd_vel is body-frame."""
        if step.confidence < self.config.min_confidence:
            return Twist.zero()
        vx, vy = STEPS[str(step.choice)]
        speed = self.config.speed
        yaw = _yaw(pose)
        return Twist(
            linear=(
                speed * (math.cos(yaw) * vx + math.sin(yaw) * vy),
                speed * (-math.sin(yaw) * vx + math.cos(yaw) * vy),
                0.0,
            )
        )

    def _trace(
        self,
        trajectory: TrajectoryBuilder,
        raw: Path,
        tick: int,
        state: WorldState,
        response: Any,
        started: float,
    ) -> None:
        """Persist the call and record one ATIF step.

        The runner counts ``raw/NNN-request.json`` for request_attempts, so the
        naming matters.
        """
        request_path = raw / f"{tick:03d}-request.json"
        response_path = raw / f"{tick:03d}-response.json"
        request_path.write_text(
            json.dumps({"body": {"state": state.encode()}, "started_at": started}, indent=2)
        )
        answers = {name: _answer_json(a) for name, a in response.answers.items()}
        response_path.write_text(json.dumps({"answers": answers}, indent=2, default=str))
        usage = getattr(response, "usage", None)
        trajectory.step(
            message=json.dumps(answers),
            request=request_path,
            response=response_path,
            model_name=self.config.model,
            metrics=Metrics(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
            ),
            at=started,
            latency_s=time.time() - started,
        )


def _answer_json(answer: Any) -> dict[str, Any]:
    """Flatten a Jev answer for the trace; graders read these back."""
    return {
        name: getattr(answer, name)
        for name in ("choice", "score", "noul", "confidence", "probabilities")
        if getattr(answer, name, None) is not None
    }


def _yaw(pose: PoseStamped) -> float:
    q = pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
