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

"""ReplicaCAD apt_1 authored-placement references; see matching suite_draft."""

from functools import partial

from dimos.evals.suites.lib.habitat_qa import (
    REPLICACAD_DATASET,
    boolean,
    case,
    count,
    measurement,
    order,
)
from dimos.evals.types import Suite

_case = partial(case, "replicacad_apt_1", "apt_1", "REPLICACAD_DATASET_CONFIG", REPLICACAD_DATASET)
SUITE: Suite = [
    _case(
        "bicycles",
        "How many bicycles are in the scene? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "beanbags",
        "How many beanbag seats are in the scene? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "stools",
        "How many stools are in the scene? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "chairs",
        "How many chairs are in the scene, excluding stools and beanbags? Return only the count.",
        count(6),
        {"object-count", "count"},
    ),
    _case(
        "plants",
        "How many indoor potted plants are in the scene? Return only the count.",
        count(3),
        {"object-count", "count"},
    ),
    _case(
        "remotes",
        "How many remote controls are in the scene? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "umbrella",
        "Is there an umbrella in the scene? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "books",
        "How many individual books are in the scene? Return only the count.",
        count(21),
        {"object-count", "count"},
    ),
    _case(
        "sofa_width",
        "What is the approximate width of the sofa, in meters? Return only the number.",
        measurement(2.14, 0.10, 0.40),
        {"dimensions", "numeric"},
    ),
    _case(
        "stand_height",
        "What is the approximate height of the TV stand, in meters? Return only the number.",
        measurement(0.60, 0.05, 0.20),
        {"dimensions", "numeric"},
    ),
    _case(
        "sofa_stand_distance",
        "What is the approximate horizontal straight-line distance between the centers of the sofa and TV stand, in meters? Return only the number.",
        measurement(6.04, 0.30, 1.2),
        {"distance", "numeric"},
    ),
    _case(
        "nearest_bicycle",
        "What is the approximate horizontal straight-line distance from the center of the sofa to the center of the nearest bicycle, in meters? Return only the number.",
        measurement(1.15, 0.15, 0.60),
        {"distance", "numeric"},
    ),
    _case(
        "height_order",
        "Order these objects from shortest to tallest. A) Sofa; B) TV stand; C) A bicycle. Return all three letters once in order, optionally separated by commas.",
        order("BAC"),
        {"dimensions", "ranking"},
    ),
]
