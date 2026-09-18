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
from dimos.agents.typesafe.drive import decode_drive, drive_questions


def _choice(label: str, probs: dict[str, float], confidence: float) -> dict:
    return {"type": "choice", "choice": label, "probabilities": probs, "confidence": confidence}


def _answers(x="none", y="none", yaw="none", conf=0.9, stop=0.0, finished=0.0) -> dict:
    return {
        "drive.x": _choice(
            x,
            {
                "forward": 1.0 if x == "forward" else 0.0,
                "backward": 1.0 if x == "backward" else 0.0,
                "none": 1.0 if x == "none" else 0.0,
            },
            conf,
        ),
        "drive.y": _choice(
            y,
            {
                "left": 1.0 if y == "left" else 0.0,
                "right": 1.0 if y == "right" else 0.0,
                "none": 1.0 if y == "none" else 0.0,
            },
            conf,
        ),
        "drive.yaw": _choice(
            yaw,
            {
                "turn_left": 1.0 if yaw == "turn_left" else 0.0,
                "turn_right": 1.0 if yaw == "turn_right" else 0.0,
                "none": 1.0 if yaw == "none" else 0.0,
            },
            conf,
        ),
        "stop": {"type": "noul", "noul": stop},
        "finished": {"type": "noul", "noul": finished},
    }


def test_questions_are_one_choice_per_axis_plus_stop() -> None:
    q = drive_questions()
    assert set(q) == {"drive.x", "drive.y", "drive.yaw", "stop", "finished"}
    assert set(q["drive.x"]["criteria"]) == {"forward", "none", "backward"}
    assert q["stop"]["type"] == "noul"


def test_forward_and_strafe_compose() -> None:
    d = decode_drive(_answers(x="forward", y="left"))
    assert (d.x, d.y, d.yaw) == (1.0, 1.0, 0.0)
    assert not d.stop and d.labels == ("forward", "left", "none")


def test_low_confidence_axis_is_zero() -> None:
    d = decode_drive(_answers(x="forward", conf=0.3), min_confidence=0.5)
    assert d.x == 0.0 and d.labels[0] == "none"
    assert d.confidence == 0.3


def test_stop_overrides_axes() -> None:
    d = decode_drive(_answers(x="forward", yaw="turn_right", stop=0.9))
    assert d.stop and d.is_zero and (d.x, d.y, d.yaw) == (0.0, 0.0, 0.0)


def test_finished_stops_and_flags() -> None:
    d = decode_drive(_answers(x="forward", finished=0.9))
    assert d.finished and d.stop and d.is_zero
    assert not decode_drive(_answers(x="forward", finished=0.5)).finished


def test_blend_uses_probability_difference() -> None:
    a = _answers()
    a["drive.x"] = _choice("forward", {"forward": 0.6, "none": 0.1, "backward": 0.3}, 0.4)
    d = decode_drive(a, blend=True)
    assert abs(d.x - 0.3) < 1e-9


def test_target_question_lists_object_labels() -> None:
    q = drive_questions(("chair", "person"))
    assert set(q["target"]["criteria"]) == {"chair", "person", "none"}
    assert "target" not in drive_questions()


def test_target_decoded_and_none_dropped() -> None:
    a = _answers(x="forward")
    a["target"] = _choice("chair", {"chair": 0.9, "none": 0.1}, 0.8)
    assert decode_drive(a).target == "chair"
    a["target"] = _choice("none", {"chair": 0.1, "none": 0.9}, 0.8)
    assert decode_drive(a).target is None
