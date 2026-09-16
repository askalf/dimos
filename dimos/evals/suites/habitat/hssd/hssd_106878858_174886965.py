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

"""Reviewed original HSSD home; physical bed count follows human review.

See the matching suite_draft for path origin conventions and metadata evidence.
Exterior-opening question is observed from indoors, not an outdoor task.
"""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.suites.lib.hssd_qa import case, parsed
from dimos.evals.types import Suite

_case = partial(case, "106878858_174886965")
SUITE: Suite = [
    _case(
        "bedrooms",
        "How many bedrooms are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(4, v)),
        {"rooms", "count"},
    ),
    _case(
        "bathrooms",
        "How many bathrooms are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(2, v)),
        {"rooms", "count"},
    ),
    _case(
        "largest_bedroom_area",
        "What is the approximate area of the largest bedroom, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(46.72, v, tolerance=3, band=10)),
        {"area", "numeric"},
    ),
    _case(
        "smallest_bedroom_area",
        "What is the approximate area of the smallest bedroom, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(9.64, v, tolerance=0.75, band=3)),
        {"area", "numeric"},
    ),
    _case(
        "garage_area",
        "What is the approximate garage floor area, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(38.37, v, tolerance=2.5, band=8)),
        {"area", "numeric"},
    ),
    _case(
        "dining_perimeter",
        "What is the approximate dining-room perimeter, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(22.54, v, tolerance=1, band=4)),
        {"perimeter", "numeric"},
    ),
    _case(
        "laptop_location",
        "Which room contains the laptop? A) Bedroom; B) Office; C) Living room; D) Kitchen. Return only A, B, C, or D.",
        lambda o: exact("B", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "mower_location",
        "Which area contains the lawn mower? A) Kitchen; B) Laundry area; C) Office; D) Garage. Return only A, B, C, or D.",
        lambda o: exact("D", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "washing_machines",
        "How many washing machines are in the laundry room? Return only the count.",
        parsed(first_number, lambda v: exact(2, v)),
        {"object-count", "count"},
    ),
    _case(
        "office_sofa",
        "Is there a sofa in the office? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"existence", "boolean"},
    ),
    _case(
        "fridge_height",
        "Approximately how tall is the refrigerator, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(2.40, v, tolerance=0.12, band=0.5)),
        {"dimensions", "numeric"},
    ),
    _case(
        "area_order",
        "Order these spaces from smallest to largest floor area. A) Garage; B) Office; C) Living room. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("BCA", v)),
        {"area", "ranking"},
    ),
    _case(
        "beds",
        "How many beds are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(4, v)),
        {"object-count", "count"},
    ),
    _case(
        "garage_bedroom_doorways",
        "What is the minimum number of doorways from the garage to the largest bedroom? Return only the count.",
        parsed(first_number, lambda v: exact(4, v)),
        {"connectivity", "count"},
    ),
    _case(
        "entryway_path_order",
        "Order these objects from nearest to farthest by collision-free travel distance from the entrance hall opening into the living room, for a robot of radius 0.25 m. A) Office laptop; B) Kitchen refrigerator; C) Garage mower. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("ABC", v)),
        {"distance", "ranking", "draft-reference"},
    ),
    _case(
        "garage_car_color",
        "What color is the car in the garage? A) Blue; B) Red; C) White; D) Black. Return only A, B, C, or D.",
        lambda o: exact("B", o.trajectory.final_answer.strip().upper()),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "exterior_opening",
        "Is there an open doorway or passage from the home to the outside? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"object-state", "boolean"},
    ),
]
