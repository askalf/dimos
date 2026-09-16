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

"""Habitat test scan, distinct from both DimSim and ReplicaCAD apartments."""

from functools import partial

from dimos.evals.suites.lib.habitat_qa import TEST_APARTMENT, boolean, case, choice, count
from dimos.evals.types import Suite

_case = partial(
    case,
    "habitat_test_apartment_1",
    TEST_APARTMENT,
    "HABITAT_TEST_DATASET_CONFIG",
    "default",
    scene_env="HABITAT_TEST_SCENE",
)
SUITE: Suite = [
    _case(
        "mirror_shape",
        "What shape is the wall mirror above the dining-room sideboard? A) Rectangular; B) Circular; C) Triangular; D) Hexagonal. Return only the letter.",
        choice("B"),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "window_covering",
        "What type of window covering is used in the living room? A) Horizontal blinds; B) Fabric curtains; C) Exterior shutters; D) No covering. Return only the letter.",
        choice("A"),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "dining_door_state",
        "Is the dining-room door open or closed? A) Closed; B) Open. Return only the letter.",
        choice("B"),
        {"object-state", "single-choice"},
    ),
    _case(
        "tv_location",
        "Which room contains the wall-mounted television? A) Dining room; B) Bedroom; C) Living room; D) Bathroom. Return only the letter.",
        choice("C"),
        {"object-location", "single-choice"},
    ),
    _case(
        "round_mirror_location",
        "Which room contains the round wall mirror? A) Dining room; B) Bathroom; C) Bedroom; D) Living room. Return only the letter.",
        choice("A"),
        {"object-location", "single-choice"},
    ),
    _case(
        "tv_exists",
        "Is there a television in the living room? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "potted_tree_exists",
        "Is there a potted tree in the living room? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "chess_decoration_count",
        "How many oversized chess-piece decorations are on the console beneath the television? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "serving_stand_tiers",
        "How many tiers does the serving stand on the dining table have? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "below_round_mirror",
        "What is directly below the round wall mirror? A) Bed; B) Sofa; C) Bathtub; D) Sideboard. Return only the letter.",
        choice("D"),
        {"spatial-relation", "single-choice"},
    ),
    _case(
        "tv_above_console",
        "Is the television mounted on the wall above the console? Return only yes or no.",
        boolean("yes"),
        {"spatial-relation", "boolean"},
    ),
    _case(
        "coffee_table_between",
        "Is there a coffee table between the sectional sofa and the television? Return only yes or no.",
        boolean("yes"),
        {"spatial-relation", "boolean"},
    ),
    _case(
        "sideboard_mirror_count",
        "How many round wall mirrors are visible above the dining-room sideboard? Return only the count.",
        count(1),
        {"object-count", "count"},
    ),
]
