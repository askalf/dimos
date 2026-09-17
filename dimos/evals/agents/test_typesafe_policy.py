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

"""Offline tests for the TypeSafe policy: no API key, no simulator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typesafe_sdk import Choice, ChoiceAnswer, Noul

from dimos.evals.agents.typesafe_policy import (
    STEP_CRITERIA,
    STEPS,
    TypeSafePolicy,
    build_questions,
    load_scene,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

SCENE = {
    "frame_id": "world",
    "goal": {"label": "bed", "xy": [-3.5, -1.3]},
    "room_bounds": [-6.0, -4.0, 6.0, 4.0],
    "obstacles": [{"label": "sofa", "min_xy": [1.0, 0.5], "max_xy": [2.5, 1.8]}],
}


@pytest.fixture
def scene_json(tmp_path: Path) -> Path:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(SCENE))
    return path


def _pose(yaw_rad: float) -> PoseStamped:
    return PoseStamped(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, math.sin(yaw_rad / 2), math.cos(yaw_rad / 2)),
        frame_id="world",
    )


def _choice(key: str, confidence: float = 0.9) -> ChoiceAnswer:
    return ChoiceAnswer(choice=key, confidence=confidence, probabilities={key: confidence})


def test_steps_and_criteria_agree() -> None:
    """Every option the model can pick maps to a vector, and vice versa."""
    assert set(STEPS) == set(STEP_CRITERIA)


def test_questions_are_one_choice_and_one_noul() -> None:
    questions = build_questions()
    assert isinstance(questions["step"], Choice)
    assert isinstance(questions["reached"], Noul)


def test_scene_round_trips(scene_json: Path) -> None:
    scene = load_scene(scene_json)
    assert scene.goal_label == "bed"
    assert scene.goal_xy == (-3.5, -1.3)
    assert [o.label for o in scene.obstacles] == ["sofa"]


def test_observe_merges_static_scene_with_live_pose(scene_json: Path) -> None:
    policy = TypeSafePolicy(scene_json=scene_json)
    policy._pose = _pose(math.radians(90.0))
    _, state = policy.observe(load_scene(scene_json), tick=3)
    assert state.goal_xy == (-3.5, -1.3)
    assert state.robot_yaw_deg == pytest.approx(90.0)
    assert state.ticks_elapsed == 3
    assert "sofa" in json.dumps(state.encode())


@pytest.mark.parametrize(
    ("key", "expected"),
    [("0,1", (0.0, 0.4)), ("1,0", (0.4, 0.0)), ("-1,0", (-0.4, 0.0)), ("0,0", (0.0, 0.0))],
)
def test_twist_at_zero_yaw_is_world_frame(
    scene_json: Path, key: str, expected: tuple[float, float]
) -> None:
    policy = TypeSafePolicy(scene_json=scene_json)
    twist = policy.twist(_choice(key), _pose(0.0))
    assert (twist.linear.x, twist.linear.y) == pytest.approx(expected)


def test_twist_rotates_world_step_into_body_frame(scene_json: Path) -> None:
    """Facing north, a world-north step must come out as body-forward."""
    policy = TypeSafePolicy(scene_json=scene_json)
    twist = policy.twist(_choice("0,1"), _pose(math.radians(90.0)))
    assert twist.linear.x == pytest.approx(0.4)
    assert twist.linear.y == pytest.approx(0.0, abs=1e-9)


def test_low_confidence_holds_still(scene_json: Path) -> None:
    policy = TypeSafePolicy(scene_json=scene_json, min_confidence=0.5)
    twist = policy.twist(_choice("0,1", confidence=0.2), _pose(0.0))
    assert (twist.linear.x, twist.linear.y) == (0.0, 0.0)


def test_preflight_rejects_modules(scene_json: Path) -> None:
    """The policy calls no tools, so it must never pull in a skill container."""
    policy = TypeSafePolicy(scene_json=scene_json, modules=("unitree-skill-container",))
    with pytest.raises(ValueError, match="no tools"):
        policy.preflight(None)  # type: ignore[arg-type]


def test_preflight_rejects_missing_scene(tmp_path: Path) -> None:
    policy = TypeSafePolicy(scene_json=tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        policy.preflight(None)  # type: ignore[arg-type]
