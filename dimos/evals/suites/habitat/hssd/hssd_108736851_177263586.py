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

"""User-reviewed hssd_108736851_177263586 QA; scene context is in ../SCENES.md."""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking
from dimos.evals.suites.lib.hssd_qa import case, parsed
from dimos.evals.types import Suite

_case = partial(case, "108736851_177263586")
SUITE: Suite = [
    _case(
        "dining_chairs",
        "How many chairs are around the dining table? Return only the count.",
        parsed(first_number, lambda v: exact(8, v)),
        {"object-count", "count"},
    ),
    _case(
        "curved_sofa_table_shape",
        "What shape is the tabletop between the two quarter-circle sofas? A) Circular; B) Square; C) Rectangular; D) Triangular. Return only A, B, C, or D.",
        lambda o: exact("A", o.trajectory.final_answer.strip().upper()),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "side_table_sides",
        "How many sides do the tabletops beside the blue sofa in the living room have? A) 4; B) 5; C) 6; D) 8. Return only A, B, C, or D.",
        lambda o: exact("C", o.trajectory.final_answer.strip().upper()),
        {"visual-attribute", "single-choice"},
    ),
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
        "tv_location",
        "Which room contains the television? A) Living room; B) Office; C) Bedroom; D) Kitchen. Return only A, B, C, or D.",
        lambda o: exact("B", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "living_area",
        "What is the approximate living-room floor area, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(123.96, v, tolerance=6, band=25)),
        {"area", "numeric"},
    ),
    _case(
        "largest_bedroom",
        "What is the approximate area of the largest bedroom, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(37.23, v, tolerance=2.5, band=8)),
        {"area", "numeric"},
    ),
    _case(
        "area_order",
        "Order these rooms from smallest to largest area. A) Office; B) Living room; C) Dining room. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("CAB", v)),
        {"area", "ranking"},
    ),
    _case(
        "beds",
        "How many beds are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(4, v)),
        {"object-count", "count"},
    ),
    _case(
        "office_path_order",
        "Rank these rooms by shortest walking distance to enter them from the office doorway facing the hallway, nearest first, for a robot of radius 0.25 m. A) Larger kitchen; B) Dining room; C) Laundry room. Return all letters once in order, optionally separated by commas.",
        # Source Habitat (10.973,.177897,-.232), static navmesh .25/.60 m.
        # Nearest sampled points inside region polygons: laundry 6.233,
        # dining 21.470, larger kitchen 21.595 m. The last two nearly tie;
        # this is a room-entry convention, not a center-distance ranking.
        parsed(ranking, lambda v: rank_order("CBA", v)),
        {"distance", "ranking"},
    ),
]
