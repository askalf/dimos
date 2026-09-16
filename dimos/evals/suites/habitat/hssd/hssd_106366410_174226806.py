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

"""Reviewed HSSD gym/piano home. Scene-specific evidence is in the matching draft.

User corrections override toilet metadata. Laundry total and path ordering
retain draft-reference tags. HSSD runtime requires a compatible navmesh.
"""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.suites.lib.hssd_qa import case, parsed
from dimos.evals.types import Suite

_case = partial(case, "106366410_174226806")
SUITE: Suite = [
    _case(
        "gym_exists",
        "Is there a dedicated exercise room? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"existence", "boolean"},
    ),
    _case(
        "treadmill_location",
        "Which room contains the treadmill? A) Office; B) Gym; C) Bedroom; D) Living room. Return only A, B, C, or D.",
        lambda o: exact("B", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "piano_location",
        "Which room contains the grand piano? A) Dining room; B) Office; C) Living room; D) Gym. Return only A, B, C, or D.",
        lambda o: exact("C", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "computer_location",
        "Which room contains the desktop computer? A) Office; B) Bedroom; C) Living room; D) Gym. Return only A, B, C, or D.",
        lambda o: exact("D", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "refrigerators",
        "How many refrigerator-freezer units are in the kitchen? Return only the count.",
        parsed(first_number, lambda v: exact(2, v)),
        {"object-count", "count"},
    ),
    _case(
        "living_area",
        "What is the approximate area of the living room, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(54.12, v, tolerance=3, band=12)),
        {"area", "numeric"},
    ),
    _case(
        "gym_area",
        "What is the approximate area of the gym, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(23.08, v, tolerance=1.5, band=6)),
        {"area", "numeric"},
    ),
    _case(
        "office_perimeter",
        "What is the approximate perimeter of the office, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(19.20, v, tolerance=1, band=4)),
        {"perimeter", "numeric"},
    ),
    _case(
        "bedrooms",
        "How many bedrooms are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(1, v)),
        {"rooms", "count"},
    ),
    _case(
        "bedroom_sofa",
        "Is there a sofa in the bedroom? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"existence", "boolean"},
    ),
    _case(
        "refrigerator_height",
        "What is the approximate height of a kitchen refrigerator, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(1.77, v, tolerance=0.1, band=0.4)),
        {"dimensions", "numeric"},
    ),
    _case(
        "area_order",
        "Order these rooms from smallest to largest area. A) Office; B) Bedroom; C) Kitchen. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("CAB", v)),
        {"area", "ranking"},
    ),
    _case(
        "laundry_appliances",
        "How many washing and drying machines are in the laundry room? Return only the count.",
        parsed(first_number, lambda v: exact(3, v)),
        {"object-count", "count", "draft-reference"},
    ),
    _case(
        "toilet_room_bathtub",
        "Does the room containing the toilet also contain a bathtub? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"spatial-relation", "boolean"},
    ),
    _case(
        "bedroom_path_order",
        "Order these objects from nearest to farthest by collision-free travel distance from the bedroom doorway facing the hallway, for a robot of radius 0.25 m. A) Grand piano; B) Dining table; C) Treadmill. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("BCA", v)),
        {"distance", "ranking", "draft-reference"},
    ),
]
