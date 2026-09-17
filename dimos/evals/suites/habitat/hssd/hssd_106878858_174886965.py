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

"""User-reviewed hssd_106878858_174886965 QA; scene context is in ../SCENES.md."""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.suites.lib.hssd_qa import case, parsed
from dimos.evals.types import Suite

_case = partial(case, "106878858_174886965")
SUITE: Suite = [
    _case(
        "bathroom_floor_pattern_match",
        "Do all the bathrooms have the same floor pattern? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"visual-attribute", "boolean"},
    ),
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
        # Source Habitat (-9.217360,.158400,-4.237486), static navmesh .25/.60 m.
        # .15 m goal grid within 1.5 m of anchors: laptop 3.254, fridge 8.632,
        # mower 11.717 m; the other tested entryway openings preserve this order.
        parsed(ranking, lambda v: rank_order("ABC", v)),
        {"distance", "ranking"},
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
