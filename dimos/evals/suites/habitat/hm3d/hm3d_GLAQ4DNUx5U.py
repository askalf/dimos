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

"""Annotation-derived HM3D references; semantic data stays grader-side."""

from functools import partial

from dimos.evals.suites.lib.habitat_qa import HM3D_ANNOTATED_DATASET, boolean, case, choice, count
from dimos.evals.types import Suite

_case = partial(
    case,
    "hm3d_GLAQ4DNUx5U",
    "00861-GLAQ4DNUx5U",
    "HM3D_ANNOTATED_DATASET_CONFIG",
    HM3D_ANNOTATED_DATASET,
)
SUITE: Suite = [
    _case(
        "mural_location",
        "Which room has a large colorful graffiti-style wall mural? A) Kitchen; B) Bathroom; C) Bedroom; D) Utility room. Return only the letter.",
        choice("C"),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "every_desk_chair",
        "Does every desk have a desk chair? Return only yes or no.",
        boolean("yes"),
        {"spatial-relation", "boolean"},
    ),
    _case(
        "beds",
        "How many beds are in the scanned home? Return only the count.",
        count(4),
        {"object-count", "count"},
    ),
    _case(
        "televisions",
        "How many televisions are in the scanned home? Return only the count.",
        count(3),
        {"object-count", "count"},
    ),
    _case(
        "toilets",
        "How many toilets are in the scanned home? Return only the count.",
        count(4),
        {"object-count", "count"},
    ),
    _case(
        "washing_machines",
        "How many washing machines are in the utility room? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "exercise_bike_exists",
        "Is there an exercise bike in the home? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "exercise_bike_location",
        "Which type of room contains the exercise bike? A) Kitchen; B) Bedroom; C) Bathroom; D) Garage. Return only the letter.",
        choice("B"),
        {"object-location", "single-choice"},
    ),
    _case(
        "fridge_location",
        "Which type of room contains the refrigerator? A) Living room; B) Bedroom; C) Utility/laundry room; D) Bathroom. Return only the letter.",
        choice("C"),
        {"object-location", "single-choice"},
    ),
    _case(
        "ironing_board_exists",
        "Is there an ironing board in the utility room? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "vacuum_cleaners",
        "How many vacuum cleaners are in the scanned home? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "bedroom_tvs",
        "How many bedrooms contain a television? Return only the count.",
        count(2),
        {"spatial-relation", "count"},
    ),
    _case(
        "every_bedroom_tv",
        "Does every bedroom have a television? Return only yes or no.",
        boolean("no"),
        {"spatial-relation", "boolean"},
    ),
    _case(
        "ovens",
        "How many ovens are in the kitchen? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
]
