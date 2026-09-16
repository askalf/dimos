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

"""Reviewed HSSD two-kitchen home; evidence in matching suite_draft.

Bed count and refrigerator coverage follow user review. Region-entry path
ranking is approximate and tagged draft-reference; doorway count stays pending.
"""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.suites.lib.hssd_qa import case, parsed
from dimos.evals.types import Suite

_case = partial(case, "108736851_177263586")
SUITE: Suite = [
    _case(
        "bedrooms",
        "How many bedrooms are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(4, v)),
        {"rooms", "count"},
    ),
    _case(
        "kitchens",
        "How many kitchen areas are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(2, v)),
        {"rooms", "count"},
    ),
    _case(
        "bathrooms",
        "How many bathrooms are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(3, v)),
        {"rooms", "count"},
    ),
    _case(
        "tv_location",
        "Which room contains the television? A) Living room; B) Office; C) Bedroom; D) Kitchen. Return only A, B, C, or D.",
        lambda o: exact("B", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "office_sofa",
        "Is there a sectional sofa in the office? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"existence", "boolean"},
    ),
    _case(
        "living_area",
        "What is the approximate living-room floor area, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(123.96, v, tolerance=6, band=25)),
        {"area", "numeric"},
    ),
    _case(
        "office_area",
        "What is the approximate office floor area, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(48.01, v, tolerance=3, band=10)),
        {"area", "numeric"},
    ),
    _case(
        "largest_bedroom",
        "What is the approximate area of the largest bedroom, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(37.23, v, tolerance=2.5, band=8)),
        {"area", "numeric"},
    ),
    _case(
        "larger_kitchen",
        "What is the approximate area of the larger kitchen, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(23.50, v, tolerance=1.5, band=6)),
        {"area", "numeric"},
    ),
    _case(
        "office_perimeter",
        "What is the approximate office perimeter, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(30.10, v, tolerance=1.5, band=5)),
        {"perimeter", "numeric"},
    ),
    _case(
        "area_order",
        "Order these rooms from smallest to largest area. A) Office; B) Living room; C) Dining room. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("CAB", v)),
        {"area", "ranking"},
    ),
    _case(
        "washer_height",
        "Approximately how tall is the washing machine, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(1.25, v, tolerance=0.1, band=0.35)),
        {"dimensions", "numeric"},
    ),
    _case(
        "fridge_height",
        "Approximately how tall is the refrigerator, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(2.22, v, tolerance=0.12, band=0.45)),
        {"dimensions", "numeric"},
    ),
    _case(
        "beds",
        "How many beds are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(4, v)),
        {"object-count", "count"},
    ),
    _case(
        "every_kitchen_fridge",
        "Does every kitchen have a refrigerator? Return only yes or no.",
        parsed(yes_no, lambda v: exact("no", v)),
        {"spatial-relation", "boolean"},
    ),
    _case(
        "office_path_order",
        "Rank these rooms by shortest walking distance to enter them from the office doorway facing the hallway, nearest first, for a robot of radius 0.25 m. A) Larger kitchen; B) Dining room; C) Laundry room. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("CBA", v)),
        {"distance", "ranking", "draft-reference"},
    ),
]
