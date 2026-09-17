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

"""User-reviewed hssd_108736884_177263634 QA; scene context is in ../SCENES.md."""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.suites.lib.habitat_qa import hssd_case as case, parsed
from dimos.evals.types import Suite

_case = partial(case, "108736884_177263634")
SUITE: Suite = [
    _case(
        "bathroom_plant_exists",
        "Is there a plant in any bathroom? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"existence", "boolean"},
    ),
    _case(
        "bedrooms",
        "How many bedrooms are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(3, v)),
        {"rooms", "count"},
    ),
    _case(
        "toilets",
        "How many toilets are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(3, v)),
        {"object-count", "count"},
    ),
    _case(
        "red_potted_plant_location",
        "Which room contains the red potted plant? A) Office; B) Bedroom; C) Kitchen; D) Living room. Return only A, B, C, or D.",
        lambda o: exact("D", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "kitchen_area",
        "What is the approximate kitchen floor area, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(41.40, v, tolerance=2.5, band=9)),
        {"area", "numeric"},
    ),
    _case(
        "kitchen_counter_windows",
        "How many windows are along the kitchen counter? Return only the count.",
        parsed(first_number, lambda v: exact(3, v)),
        {"object-count", "count"},
    ),
    _case(
        "bathtub_shape_match",
        "Are the bathtubs in the home the same shape? Return only yes or no.",
        parsed(yes_no, lambda v: exact("no", v)),
        {"visual-attribute", "boolean"},
    ),
    _case(
        "fridge_height",
        "What is the approximate refrigerator height, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(1.77, v, tolerance=0.1, band=0.4)),
        {"dimensions", "numeric"},
    ),
    _case(
        "area_order",
        "Order these rooms from smallest to largest floor area. A) Dining room; B) Kitchen; C) Office. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("CAB", v)),
        {"area", "ranking"},
    ),
]
