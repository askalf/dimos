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

"""The adapter: TypeSafe questions for driving, and their answers decoded into a Drive."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict

from typing_extensions import NotRequired

Text = str | Mapping[str, object] | list[object]


class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    instructions: Text
    criteria: Mapping[str, Text | None]


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: Text
    criteria: NotRequired[Mapping[str, Text]]


Question = ChoiceQuestion | NoulQuestion


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


Answers = dict[str, ChoiceAnswer | NoulAnswer]


def choice(instructions: Text, criteria: Mapping[str, Text | None]) -> ChoiceQuestion:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: Text, criteria: Mapping[str, Text]) -> NoulQuestion:
    return {"type": "noul", "instructions": instructions, "criteria": criteria}


AXES = (("x", "forward", "backward"), ("y", "left", "right"), ("yaw", "turn_left", "turn_right"))
_CONTEXT = "Read `goal`, `robot`, `objects` (each has `bearing` and `distance`) and `room` (each sector has `state`)."


def _q(question: str) -> Text:
    return {"question": question, "context": _CONTEXT}


def _opt(what: str, not_for: str, example: str) -> Text:
    return {"what": what, "not_for": not_for, "examples": [example]}


def questions(labels: tuple[str, ...]) -> dict[str, Question]:
    qs: dict[str, Question] = {
        "drive.x": choice(
            _q(
                "Should the robot move straight ahead, reverse, or neither, to get closer to the target named in `goal`?"
            ),
            {
                "forward": _opt(
                    "the target is ahead, ahead_left or ahead_right and `room.ahead.state` is not blocked",
                    "target beside or behind; ahead blocked",
                    "chair ahead, mid, ahead clear",
                ),
                "none": _opt(
                    "the target is beside or behind, is touching, or ahead is blocked",
                    "target ahead with a clear path",
                    "person left, near",
                ),
                "backward": _opt(
                    "the robot is touching an obstacle ahead and must back away",
                    "any case where turning or stopping would do",
                    "ahead blocked at 0.3 m",
                ),
            },
        ),
        "drive.y": choice(
            _q(
                "Should the robot strafe left, right, or neither, to line up with the target named in `goal` or sidestep an obstacle?"
            ),
            {
                "left": _opt(
                    "the target is ahead_left or left and that side is not blocked",
                    "target ahead or right",
                    "door ahead_left, near",
                ),
                "none": _opt(
                    "the target is straight ahead, behind, or strafing would hit an obstacle",
                    "target clearly off to one side",
                    "chair ahead, mid",
                ),
                "right": _opt(
                    "the target is ahead_right or right and that side is not blocked",
                    "target ahead or left",
                    "table ahead_right, near",
                ),
            },
        ),
        "drive.yaw": choice(
            _q(
                "Should the robot rotate in place counter-clockwise, clockwise, or not at all, so the target named in `goal` is ahead?"
            ),
            {
                "turn_left": _opt(
                    "the target bearing is left, ahead_left or behind_left",
                    "target ahead or on the right",
                    "person left, far",
                ),
                "none": _opt(
                    "the target is ahead", "target off to a side or behind", "bed ahead, mid"
                ),
                "turn_right": _opt(
                    "the target bearing is right, ahead_right, behind_right or behind",
                    "target ahead or on the left",
                    "chair behind",
                ),
            },
        ),
        "stop": noul(
            _q("Should the robot stop moving right now?"),
            {
                "true": "the target named in `goal` has `distance` touching, or `goal` asks to stop, or the target is not in `objects`, or `room.ahead.state` is blocked while moving forward",
                "false": "the target is in `objects` with `distance` near, mid or far and there is a clear direction to move; being near is not a reason to stop",
            },
        ),
    }
    if labels:
        qs["target"] = choice(
            "Which entry of `objects` is the thing `goal` asks to go to? Match by `label`.",
            {**dict.fromkeys(labels), "none": "`goal` names nothing that is in `objects`"},
        )
    return qs


@dataclass(frozen=True)
class Drive:
    x: float
    y: float
    yaw: float
    stop: bool
    confidence: float
    labels: tuple[str, str, str]
    target: str | None

    @property
    def is_zero(self) -> bool:
        return self.stop or not (self.x or self.y or self.yaw)


def _choice(answers: Answers, key: str) -> ChoiceAnswer | None:
    a = answers.get(key)
    return a if a is not None and a["type"] == "choice" else None


def decode(answers: Answers, *, min_confidence: float, stop_threshold: float) -> Drive:
    s = answers.get("stop")
    stop = s is not None and s["type"] == "noul" and s["noul"] >= stop_threshold
    vals: list[float] = []
    labels: list[str] = []
    confs: list[float] = []
    for axis, pos, _neg in AXES:
        a = _choice(answers, f"drive.{axis}")
        label, conf = (a["choice"], a["confidence"]) if a else ("none", 0.0)
        if conf < min_confidence:
            label = "none"
        vals.append(0.0 if stop or label == "none" else 1.0 if label == pos else -1.0)
        labels.append(label)
        confs.append(conf)
    t = _choice(answers, "target")
    target = (
        t["choice"] if t and t["choice"] != "none" and t["confidence"] >= min_confidence else None
    )
    return Drive(
        vals[0], vals[1], vals[2], stop, min(confs), (labels[0], labels[1], labels[2]), target
    )
