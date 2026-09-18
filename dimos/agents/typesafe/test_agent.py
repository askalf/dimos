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
from collections.abc import Iterator
import threading
import time
from typing import Any

from dimos_lcm.vision_msgs import (
    BoundingBox3D,
    Detection3D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
import pytest

from dimos.agents.typesafe.agent import TypeSafeAgent
from dimos.agents.typesafe.client import API_KEY_ENV
from dimos.core.transport import LCMTransport, pLCMTransport
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray


def _choice(label: str, options: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": label,
        "confidence": 1.0,
        "probabilities": {o: float(o == label) for o in options},
    }


def _answers(
    x: str = "none", y: str = "none", yaw: str = "none", stop: float = 0.0
) -> dict[str, Any]:
    return {
        "target": _choice("chair", ("chair", "none")),
        "drive.x": _choice(x, ("forward", "none", "backward")),
        "drive.y": _choice(y, ("left", "none", "right")),
        "drive.yaw": _choice(yaw, ("turn_left", "none", "turn_right")),
        "stop": {"type": "noul", "noul": stop},
    }


class FakeClient:
    def __init__(self) -> None:
        self.answers = _answers()
        self.states: list[dict[str, Any]] = []
        self.fail = False

    def system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        self.states.append(state)
        if self.fail:
            raise RuntimeError("boom")
        return self.answers

    def close(self) -> None:
        pass


@pytest.fixture
def agent(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TypeSafeAgent, FakeClient, list[Twist]]]:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    a = TypeSafeAgent(
        rate_hz=10.0, publish_hz=20.0, linear_accel=100.0, angular_accel=100.0, stale_s=5.0
    )
    a.odom.transport = LCMTransport("/test_typesafe/odom", PoseStamped)
    a.cmd_vel.transport = LCMTransport("/test_typesafe/cmd_vel", Twist)
    for name in (
        "detections_3d",
        "detections_2d",
        "lidar",
        "human_input",
        "agent",
        "agent_idle",
        "world_state",
    ):
        getattr(a, name).transport = pLCMTransport(f"/test_typesafe/{name}")
    twists: list[Twist] = []
    unsub = a.cmd_vel.transport.subscribe(twists.append)
    a.start()
    fake = FakeClient()
    a._client = fake  # type: ignore[assignment]
    yield a, fake, twists
    unsub()
    a.stop()


def _scene(a: TypeSafeAgent) -> None:
    a.odom.transport.publish(PoseStamped(position=(0, 0, 0.4), frame_id="world"))
    d = Detection3D()
    d.header = Header(1.0, "world")
    d.results = [ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id="chair", score=0.9))]
    d.results_length = 1
    d.bbox = BoundingBox3D(center=Pose(position=(3.0, 0.0, 0.3)), size=Vector3(0.5, 0.5, 0.6))
    a.detections_3d.transport.publish(
        Detection3DArray(detections_length=1, header=Header(1.0, "world"), detections=[d])
    )
    time.sleep(0.1)


def _wait(pred: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_idle_without_goal(agent: tuple[TypeSafeAgent, FakeClient, list[Twist]]) -> None:
    a, fake, twists = agent
    time.sleep(0.4)
    assert fake.states == []
    assert not any(t.linear.x or t.angular.z for t in twists)


def test_goal_drives_then_stops_and_clears(
    agent: tuple[TypeSafeAgent, FakeClient, list[Twist]],
) -> None:
    a, fake, twists = agent
    fake.answers = _answers(x="forward", y="left")
    _scene(a)
    a.set_goal("go to the chair")

    assert _wait(lambda: any(t.linear.x > 0.4 and t.linear.y > 0.4 for t in twists))
    assert fake.states[-1]["goal"] == "go to the chair"
    assert fake.states[-1]["robot"]["heading"] == "east"
    assert a.robot_state()["motion"] == "driving"

    fake.answers = _answers(stop=0.95)
    assert _wait(lambda: a.goal() is None)
    assert twists[-1].is_zero()
    assert a.robot_state()["motion"] == "idle"


def test_stop_word_clears_goal_immediately(
    agent: tuple[TypeSafeAgent, FakeClient, list[Twist]],
) -> None:
    a, fake, twists = agent
    fake.answers = _answers(x="forward")
    _scene(a)
    a.set_goal("go forward")
    assert _wait(lambda: any(t.linear.x > 0.4 for t in twists))
    a.set_goal("stop")
    assert a.goal() is None
    assert _wait(lambda: twists[-1].is_zero())


def test_inference_failure_zeroes_target(
    agent: tuple[TypeSafeAgent, FakeClient, list[Twist]],
) -> None:
    a, fake, twists = agent
    fake.answers = _answers(x="forward")
    _scene(a)
    a.set_goal("go forward")
    assert _wait(lambda: any(t.linear.x > 0.4 for t in twists))
    fake.fail = True
    assert _wait(lambda: twists[-1].is_zero())
    assert a.goal() == "go forward"


def test_deadman_zeroes_when_inference_stalls(
    agent: tuple[TypeSafeAgent, FakeClient, list[Twist]],
) -> None:
    a, fake, twists = agent
    a.config.deadman_s = 0.3
    fake.answers = _answers(x="forward")
    _scene(a)
    a.set_goal("go forward")
    assert _wait(lambda: any(t.linear.x > 0.4 for t in twists))
    block = threading.Event()
    fake.system_one = lambda state, questions: block.wait(5) or _answers(x="forward")  # type: ignore[method-assign]
    assert _wait(lambda: twists[-1].is_zero(), timeout=3.0)
    block.set()


def test_holds_without_odom_or_detections(
    agent: tuple[TypeSafeAgent, FakeClient, list[Twist]],
) -> None:
    a, fake, twists = agent
    fake.answers = _answers(x="forward")
    a.set_goal("go to the chair")
    time.sleep(0.4)
    assert fake.states == []
    assert a.robot_state()["motion"] == "holding"
    a.odom.transport.publish(PoseStamped(position=(0, 0, 0.4), frame_id="world"))
    time.sleep(0.4)
    assert fake.states == []  # odom alone is not enough: nothing to drive toward


def test_arrival_by_distance_stops_and_clears(
    agent: tuple[TypeSafeAgent, FakeClient, list[Twist]],
) -> None:
    a, fake, twists = agent
    a.config.stops_to_clear_goal = 2
    fake.answers = _answers(x="forward")
    _scene(a)
    a.odom.transport.publish(
        PoseStamped(position=(2.8, 0, 0.4), frame_id="world")
    )  # 0.2 m from the chair
    time.sleep(0.1)
    a.set_goal("go to the chair")
    assert _wait(lambda: a.goal() is None)
    assert not any(t.linear.x > 0.1 for t in twists)
