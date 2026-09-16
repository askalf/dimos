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

"""Reviewed HSSD QA. Three extra references remain unresolved in the draft:
laundry physical-machine count, refrigerator state and bedroom doorway width.
These are omitted instead of being given placeholder scorers.
"""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.suites.lib.hssd_qa import case, parsed
from dimos.evals.types import Suite

_case = partial(case, "108736884_177263634")
SUITE: Suite = [
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
        "laptop_location",
        "Which room contains the laptop? A) Office; B) Bedroom; C) Kitchen; D) Living room. Return only A, B, C, or D.",
        lambda o: exact("A", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "kitchen_area",
        "What is the approximate kitchen floor area, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(41.40, v, tolerance=2.5, band=9)),
        {"area", "numeric"},
    ),
    _case(
        "dining_area",
        "What is the approximate dining-room floor area, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(32.15, v, tolerance=2, band=7)),
        {"area", "numeric"},
    ),
    _case(
        "largest_bedroom",
        "What is the approximate area of the largest bedroom, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(44.38, v, tolerance=3, band=10)),
        {"area", "numeric"},
    ),
    _case(
        "kitchen_perimeter",
        "What is the approximate kitchen perimeter, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(29.17, v, tolerance=1.5, band=5)),
        {"perimeter", "numeric"},
    ),
    _case(
        "bathtubs",
        "How many bathtubs are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(2, v)),
        {"object-count", "count"},
    ),
    _case(
        "fridge_height",
        "What is the approximate refrigerator height, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(1.77, v, tolerance=0.1, band=0.4)),
        {"dimensions", "numeric"},
    ),
    _case(
        "kitchen_fridge",
        "Does the kitchen contain a refrigerator? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"existence", "boolean"},
    ),
    _case(
        "area_order",
        "Order these rooms from smallest to largest floor area. A) Dining room; B) Kitchen; C) Office. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("CAB", v)),
        {"area", "ranking"},
    ),
    _case(
        "dining_table_length",
        "What is the approximate length of the dining table, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(2.13, v, tolerance=0.1, band=0.4)),
        {"dimensions", "numeric"},
    ),
]
