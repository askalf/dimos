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

"""User-reviewed hssd_106366410_174226806 QA; scene context is in ../SCENES.md."""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.suites.lib.habitat_qa import hssd_case as case, parsed
from dimos.evals.types import Suite

_case = partial(case, "106366410_174226806")
SUITE: Suite = [
    _case(
        "gym_exists",
        "Is there an exercise area in the home? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"existence", "boolean"},
    ),
    _case(
        "red_trash_bin_location",
        "Which room contains the red trash bin? A) Kitchen; B) Combined gym/office; C) Bedroom; D) Living room. Return only A, B, C, or D.",
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
        "refrigerators",
        "How many refrigerator-freezer units are in the kitchen? Return only the count.",
        parsed(first_number, lambda v: exact(2, v)),
        {"object-count", "count"},
    ),
    _case(
        "gym_area",
        "What is the approximate floor area of the gym zone, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(23.08, v, tolerance=1.5, band=6)),
        {"area", "numeric"},
    ),
    _case(
        "bedrooms",
        "How many bedrooms are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(1, v)),
        {"rooms", "count"},
    ),
    _case(
        "bed_relative_to_sofa",
        "For someone seated on the bedroom sofa facing forward, is the bed to their left or right? A) Left; B) Right. Return only A or B.",
        lambda o: exact("A", o.trajectory.final_answer.strip().upper()),
        {"spatial-relation", "single-choice"},
    ),
    _case(
        "refrigerator_height",
        "What is the approximate height of a kitchen refrigerator, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(1.77, v, tolerance=0.1, band=0.4)),
        {"dimensions", "numeric"},
    ),
    _case(
        "area_order",
        "Order these areas from smallest to largest floor area. A) Office zone; B) Bedroom; C) Kitchen. Return all letters once in order, optionally separated by commas.",
        parsed(ranking, lambda v: rank_order("CAB", v)),
        {"area", "ranking"},
    ),
    _case(
        "laundry_appliances",
        "How many washing and drying machines are in the laundry room? Return only the count.",
        parsed(first_number, lambda v: exact(3, v)),
        {"object-count", "count"},
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
        # Source Habitat (2.494610,.150866,-1.863723), static navmesh .25/.60 m.
        # .15 m goal grid within 1.5 m of anchors: table 6.812, treadmill 7.173,
        # piano 16.730 m. Table/treadmill separation is approach-sensitive.
        parsed(ranking, lambda v: rank_order("BCA", v)),
        {"distance", "ranking"},
    ),
]
