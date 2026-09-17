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
from typing import Any

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

SHIPPED_SCENE = Path(__file__).parents[1] / "suites" / "scenes" / "apartment_detections.json"


def _det(
    id_: str, label: str, cx: float, cy: float, cz: float, sx: float, sy: float, sz: float
) -> dict[str, Any]:
    """One entry in the DimSim detections schema (PR #4208)."""
    return {
        "id": id_,
        "label": label,
        "score": 1.0,
        "center_xyz": [cx, cy, cz],
        "size_xyz": [sx, sy, sz],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


SCENE = {
    "frame_id": "world",
    "timestamp": 0.0,
    "count": 6,
    "detections": [
        _det("couch", "Modern L-shaped sectional", 1.0, 4.0, 0.4, 2.0, 1.0, 0.8),  # goal
        _det("table", "Coffee table", 3.0, 0.0, 0.25, 1.0, 1.0, 0.5),
        _det("plate", "Dinner plate", 3.0, 0.0, 0.52, 0.2, 0.2, 0.02),  # on the table
        _det("header", "wall-south-header", 5.0, 0.0, 2.7, 0.2, 1.0, 0.9),  # overhead
        _det("wall-n", "wall-north", -5.0, 0.0, 1.5, 0.2, 12.0, 3.0),
        _det("wall-s", "wall-south", 5.0, 0.0, 1.5, 0.2, 12.0, 3.0),
    ],
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


def test_criteria_carry_no_compass_words() -> None:
    """Wall labels use a different compass; the step wording must not."""
    for text in STEP_CRITERIA.values():
        assert not any(word in text.lower() for word in ("north", "south", "east", "west"))


def test_questions_are_one_choice_and_one_noul() -> None:
    questions = build_questions()
    assert isinstance(questions["step"], Choice)
    assert isinstance(questions["reached"], Noul)


# --- scene loading ---------------------------------------------------------------


def test_scene_drops_overhead_and_contained_boxes(scene_json: Path) -> None:
    """The door header passes over the robot; the plate sits inside the table."""
    scene = load_scene(scene_json, "sectional")
    assert [o.label for o in scene.obstacles] == [
        "Modern L-shaped sectional",
        "Coffee table",
        "wall-north",
        "wall-south",
    ]


def test_scene_goal_box_and_room_bounds(scene_json: Path) -> None:
    scene = load_scene(scene_json, "sectional")
    assert scene.goal_label == "Modern L-shaped sectional"
    assert scene.goal_xy == (1.0, 4.0)
    assert scene.goal_box == (0.0, 3.5, 2.0, 4.5)
    assert scene.room_bounds == (-5.1, -6.0, 5.1, 6.0)  # from the walls, not the furniture


def test_scene_unknown_goal_raises(scene_json: Path) -> None:
    with pytest.raises(LookupError, match="hot tub"):
        load_scene(scene_json, "hot tub")


def test_shipped_apartment_scene_is_ground_truth() -> None:
    """Pins the copied PR #4208 snapshot and the filter: 15 walls + 33 objects.

    Of 107 detections, 43 are overhead (headers, wall cabinets, the TV) and 16
    are contained footprints (shelf and tabletop clutter, bedding, and one
    dining chair tucked fully under its table).
    """
    scene = load_scene(SHIPPED_SCENE, "sectional")
    labels = [o.label for o in scene.obstacles]
    assert sum(label.startswith(("wall", "yard")) for label in labels) == 15
    assert len(labels) == 48
    assert scene.goal_box == (0.156, 3.028, 1.956, 5.735)
    assert not any("header" in label for label in labels)


def test_observe_merges_static_scene_with_live_pose(scene_json: Path) -> None:
    policy = TypeSafePolicy(scene_json=scene_json)
    policy._pose = _pose(math.radians(90.0))
    _, state = policy.observe(load_scene(scene_json, "sectional"), tick=3)
    assert state.goal_box == (0.0, 3.5, 2.0, 4.5)
    assert state.robot_yaw_deg == pytest.approx(90.0)
    assert state.ticks_elapsed == 3
    assert "wall-north" in json.dumps(state.encode())


# --- controller ------------------------------------------------------------------


def test_aligned_step_drives_forward(scene_json: Path) -> None:
    """Facing +x and told +x: drive, no turn. linear.y is never used —
    DimSim's ground model ignores it."""
    policy = TypeSafePolicy(scene_json=scene_json)
    twist = policy.twist(_choice("1,0"), _pose(0.0))
    assert twist.linear.x == pytest.approx(0.2)
    assert twist.linear.y == 0.0
    assert twist.angular.z == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("key", "yaw_deg", "sign"),
    [("0,1", 0.0, +1), ("0,-1", 0.0, -1), ("-1,0", 90.0, +1), ("1,0", 90.0, -1)],
)
def test_misaligned_step_turns_in_place(
    scene_json: Path, key: str, yaw_deg: float, sign: int
) -> None:
    """Off by 90 degrees: rotate toward the target, no forward motion."""
    policy = TypeSafePolicy(scene_json=scene_json)
    twist = policy.twist(_choice(key), _pose(math.radians(yaw_deg)))
    assert twist.linear.x == 0.0
    assert twist.angular.z == pytest.approx(sign * 0.5)


def test_heading_error_wraps_across_pi(scene_json: Path) -> None:
    """Facing -170 deg and told -x (+180): that is a 10 deg error, not 350."""
    policy = TypeSafePolicy(scene_json=scene_json)
    twist = policy.twist(_choice("-1,0"), _pose(math.radians(-170.0)))
    assert twist.linear.x == pytest.approx(0.2)  # aligned enough to drive
    assert twist.angular.z == pytest.approx(-math.radians(10.0))  # small right correction


def test_hold_step_is_zero_regardless_of_heading(scene_json: Path) -> None:
    policy = TypeSafePolicy(scene_json=scene_json)
    twist = policy.twist(_choice("0,0"), _pose(math.radians(37.0)))
    assert (twist.linear.x, twist.linear.y, twist.angular.z) == (0.0, 0.0, 0.0)


def test_low_confidence_holds_still(scene_json: Path) -> None:
    policy = TypeSafePolicy(scene_json=scene_json, min_confidence=0.5)
    twist = policy.twist(_choice("0,1", confidence=0.2), _pose(0.0))
    assert (twist.linear.x, twist.linear.y) == (0.0, 0.0)


# --- preflight -------------------------------------------------------------------


def test_preflight_rejects_modules(scene_json: Path) -> None:
    """The policy calls no tools, so it must never pull in a skill container."""
    policy = TypeSafePolicy(scene_json=scene_json, modules=("unitree-skill-container",))
    with pytest.raises(ValueError, match="no tools"):
        policy.preflight(None)  # type: ignore[arg-type]


def test_preflight_rejects_missing_scene(tmp_path: Path) -> None:
    policy = TypeSafePolicy(scene_json=tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        policy.preflight(None)  # type: ignore[arg-type]


def test_preflight_rejects_unknown_goal(scene_json: Path) -> None:
    policy = TypeSafePolicy(scene_json=scene_json, goal_label="jacuzzi")
    with pytest.raises(LookupError):
        policy.preflight(None)  # type: ignore[arg-type]
