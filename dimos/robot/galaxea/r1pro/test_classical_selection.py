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

"""Selection must preserve every requested attribute before any motion."""

import pytest

from dimos.robot.galaxea.r1pro.classical_selection import resolve_classical_object


@pytest.fixture
def rows():
    return [
        dict(
            id=f"object_{i + 1}",
            object=f"task_object_{i + 1}",
            index=i,
            kind=kind,
            shape="box",
            rgba=rgba,
            left_m=side,
            distance_m=distance,
            upright=True,
            released=True,
            settled=True,
            support_geoms=["table"],
        )
        for i, (kind, rgba, side, distance) in enumerate(
            [
                ("drink_carton", [0.1, 0.2, 0.8, 1], -0.4, 0.5),
                ("drink_carton", [0.1, 0.2, 0.8, 1], 0.4, 0.9),
                ("toy_block", [0.1, 0.2, 0.8, 1], 0.6, 0.7),
                ("drink_carton", [0.8, 0.1, 0.1, 1], 0.5, 0.6),
            ]
        )
    ]


def test_description_requires_color_kind_and_robot_relative_side(rows):
    assert resolve_classical_object(rows, "blue carton on the left") == 1


def test_nearest_ranks_only_objects_matching_the_description(rows):
    assert resolve_classical_object(rows, "nearest blue carton") == 0


def test_wrong_color_does_not_silently_substitute_the_nearest_carton(rows):
    with pytest.raises(ValueError, match="No available object matches all attributes"):
        resolve_classical_object(rows, "green carton on the left")


def test_ambiguous_color_and_type_requires_an_id(rows):
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_classical_object(rows, "blue carton")


def test_explicit_unavailable_id_is_not_replaced(rows):
    rows[0]["released"] = False
    with pytest.raises(ValueError, match="not a supported"):
        resolve_classical_object(rows, "object_1")


def test_robot_rotation_changes_left_right_selection(rows):
    for row in rows:
        row["left_m"] *= -1
    assert resolve_classical_object(rows, "blue carton on the left") == 0


def test_unknown_qualifier_is_not_ignored(rows):
    with pytest.raises(ValueError, match="Unsupported attribute"):
        resolve_classical_object(rows, "tiny blue carton")


def test_spatial_tie_does_not_arbitrarily_choose_an_id(rows):
    rows[1]["distance_m"] = rows[0]["distance_m"]
    with pytest.raises(ValueError, match="tied"):
        resolve_classical_object(rows, "nearest blue carton")
