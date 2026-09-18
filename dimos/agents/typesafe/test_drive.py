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
from dimos.agents.typesafe.drive import Answers, ChoiceAnswer, decode, questions


def choice(label: str, *options: str, confidence: float = 0.9) -> ChoiceAnswer:
    return {
        "type": "choice",
        "choice": label,
        "confidence": confidence,
        "probabilities": {o: float(o == label) for o in options},
    }


def answers(
    x: str = "none", y: str = "none", yaw: str = "none", conf: float = 0.9, stop: float = 0.0
) -> Answers:
    return {
        "target": choice("chair", "chair", "none"),
        "drive.x": choice(x, "forward", "none", "backward", confidence=conf),
        "drive.y": choice(y, "left", "none", "right", confidence=conf),
        "drive.yaw": choice(yaw, "turn_left", "none", "turn_right", confidence=conf),
        "stop": {"type": "noul", "noul": stop},
    }


def _decode(a: Answers):  # type: ignore[no-untyped-def]
    return decode(a, min_confidence=0.5, stop_threshold=0.7)


def test_questions_shape() -> None:
    assert set(questions(())) == {"drive.x", "drive.y", "drive.yaw", "stop"}
    q = questions(("chair", "person"))
    assert q["target"]["type"] == "choice" and set(q["target"]["criteria"]) == {
        "chair",
        "person",
        "none",
    }
    assert set(q["drive.x"]["criteria"]) == {"forward", "none", "backward"}


def test_axes_compose() -> None:
    d = _decode(answers(x="forward", y="left"))
    assert (d.x, d.y, d.yaw, d.stop, d.labels, d.target) == (
        1.0,
        1.0,
        0.0,
        False,
        ("forward", "left", "none"),
        "chair",
    )


def test_low_confidence_axis_is_zero() -> None:
    d = _decode(answers(x="forward", conf=0.3))
    assert (d.x, d.labels[0], d.confidence) == (0.0, "none", 0.3)


def test_stop_overrides_axes() -> None:
    assert _decode(answers(x="forward", yaw="turn_right", stop=0.9)).is_zero


def test_target_none_dropped() -> None:
    a = answers(x="forward")
    a["target"] = choice("none", "chair", "none")
    assert _decode(a).target is None
