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

"""ReplicaCAD staging sc1; inventory-based book absence is a best-effort reference."""

from functools import partial

from dimos.evals.suites.lib.habitat_qa import (
    REPLICACAD_DATASET,
    boolean,
    case,
    choice,
    count,
    measurement,
    order,
)
from dimos.evals.types import Suite

_case = partial(
    case,
    "replicacad_v3_sc1_staging_00",
    "v3_sc1_staging_00",
    "REPLICACAD_DATASET_CONFIG",
    REPLICACAD_DATASET,
)
SUITE: Suite = [
    _case(
        "bicycles",
        "How many bicycles are in the scene? Return only the count.",
        count(1),
        {"object-count", "count"},
    ),
    _case(
        "beanbags",
        "How many beanbag seats are in the scene? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "chairs",
        "How many chairs are in the scene, excluding beanbags and the sofa? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "plants",
        "How many indoor potted plants are in the scene? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "sofa_exists",
        "Is there a sofa in the scene? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "beanbag_exists",
        "Are there any beanbag seats in the scene? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "fridge_exists",
        "Is there a refrigerator in the scene? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "sofa_width",
        "What is the approximate sofa width, in meters? Return only the number.",
        measurement(2.14, 0.1, 0.4),
        {"dimensions", "numeric"},
    ),
    _case(
        "bicycle_height",
        "What is the approximate height of the bicycle, in meters? Return only the number.",
        measurement(0.98, 0.06, 0.25),
        {"dimensions", "numeric"},
    ),
    _case(
        "sofa_stand_distance",
        "What is the approximate horizontal straight-line distance between the centers of the sofa and TV stand, in meters? Return only the number.",
        measurement(4.77, 0.25, 1),
        {"distance", "numeric"},
    ),
    _case(
        "sofa_bike_distance",
        "What is the approximate horizontal straight-line distance between the centers of the sofa and bicycle, in meters? Return only the number.",
        measurement(6.67, 0.3, 1.2),
        {"distance", "numeric"},
    ),
    _case(
        "nearer_object",
        "Which object's center is closer to the sofa's center in a horizontal straight line? A) TV stand; B) Bicycle. Return only the letter.",
        choice("A"),
        {"distance", "single-choice"},
    ),
    _case(
        "height_order",
        "Order these objects from shortest to tallest. A) TV stand; B) Bicycle; C) Sofa. Return all three letters once in order, optionally separated by commas.",
        order("ACB"),
        {"dimensions", "ranking"},
    ),
    _case(
        "books_exists",
        "Are there any books in the scene? Return only yes or no.",
        boolean("no"),
        {"existence", "boolean"},
    ),
]
