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
"""Drive questions and the answers -> joystick decoding (pure functions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dimos.agents.typesafe.client import choice, noul

_CONTEXT = "Read `goal`, `robot`, `objects` (each has `bearing` and `distance`) and `room.sectors` (each has `state`)."

AXES: dict[str, tuple[str, str]] = {
    "x": ("forward", "backward"),
    "y": ("left", "right"),
    "yaw": ("turn_left", "turn_right"),
}


def _opt(what: str, not_for: str, examples: list[str]) -> dict[str, Any]:
    return {"what": what, "not_for": not_for, "examples": examples}


def drive_questions() -> dict[str, dict[str, Any]]:
    return {
        "drive.x": choice(
            {
                "question": "Should the robot move straight ahead, reverse, or neither, to get closer to the target named in `goal`?",
                "context": _CONTEXT,
            },
            {
                "forward": _opt(
                    "the target is ahead / ahead_left / ahead_right and `room.sectors.ahead.state` is not blocked",
                    "target beside or behind the robot; ahead is blocked",
                    ["chair ahead, mid, ahead clear"],
                ),
                "none": _opt(
                    "the target is beside or behind the robot, is touching, or ahead is blocked",
                    "target ahead with a clear path",
                    ["person left, near", "chair ahead, touching"],
                ),
                "backward": _opt(
                    "the robot is touching an obstacle ahead and must back away",
                    "any case where turning or stopping would do",
                    ["ahead blocked at 0.3 m and target behind"],
                ),
            },
        ),
        "drive.y": choice(
            {
                "question": "Should the robot strafe left, right, or neither, to line up with the target named in `goal` or sidestep an obstacle?",
                "context": _CONTEXT,
            },
            {
                "left": _opt(
                    "the target is ahead_left or left and that side is not blocked",
                    "target ahead or right",
                    ["door ahead_left, near"],
                ),
                "none": _opt(
                    "the target is straight ahead, behind, or strafing would hit an obstacle",
                    "target clearly off to one side",
                    ["chair ahead, mid"],
                ),
                "right": _opt(
                    "the target is ahead_right or right and that side is not blocked",
                    "target ahead or left",
                    ["table ahead_right, near"],
                ),
            },
        ),
        "drive.yaw": choice(
            {
                "question": "Should the robot rotate in place counter-clockwise, clockwise, or not at all, so the target named in `goal` is ahead?",
                "context": _CONTEXT,
            },
            {
                "turn_left": _opt(
                    "the target bearing is left, ahead_left, or behind_left",
                    "target ahead or on the right",
                    ["person left, far"],
                ),
                "none": _opt(
                    "the target is ahead", "target off to a side or behind", ["bed ahead, mid"]
                ),
                "turn_right": _opt(
                    "the target bearing is right, ahead_right, behind_right, or behind",
                    "target ahead or on the left",
                    ["chair behind", "door right, near"],
                ),
            },
        ),
        "stop": noul(
            {"question": "Should the robot stop moving right now?", "context": _CONTEXT},
            {
                "true": "the target named in `goal` is touching or near and ahead, or `goal` asks to stop, or the target is not in `objects`, or a collision is imminent",
                "false": "the target is visible in `objects`, not yet reached, and there is a clear direction to move",
            },
        ),
    }


@dataclass(frozen=True)
class Drive:
    x: float
    y: float
    yaw: float
    stop: bool
    confidence: float
    labels: tuple[str, str, str]

    @property
    def is_zero(self) -> bool:
        return self.stop or (self.x == 0 and self.y == 0 and self.yaw == 0)


def _axis(
    answer: dict[str, Any], pos: str, neg: str, min_confidence: float, blend: bool
) -> tuple[float, str, float]:
    probs = answer.get("probabilities") or {}
    conf = float(answer.get("confidence", 0.0))
    label = str(answer.get("choice", "none"))
    if blend:
        return float(probs.get(pos, 0.0)) - float(probs.get(neg, 0.0)), label, conf
    if conf < min_confidence or label == "none":
        return 0.0, "none" if conf < min_confidence else label, conf
    return (1.0 if label == pos else -1.0), label, conf


def decode_drive(
    answers: dict[str, Any],
    *,
    min_confidence: float = 0.5,
    blend: bool = False,
    stop_threshold: float = 0.7,
) -> Drive:
    stop = float(answers.get("stop", {}).get("noul", 0.0)) >= stop_threshold
    vals: dict[str, float] = {}
    labels: list[str] = []
    confs: list[float] = []
    for axis, (pos, neg) in AXES.items():
        v, label, conf = _axis(answers.get(f"drive.{axis}", {}), pos, neg, min_confidence, blend)
        vals[axis] = v
        labels.append(label)
        confs.append(conf)
    if stop:
        vals = dict.fromkeys(vals, 0.0)
    return Drive(
        vals["x"],
        vals["y"],
        vals["yaw"],
        stop,
        min(confs) if confs else 0.0,
        (labels[0], labels[1], labels[2]),
    )
