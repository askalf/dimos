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

"""Visual-reference HM3D questions; evidence and pending candidates in suite_draft."""

from functools import partial

from dimos.evals.suites.lib.habitat_qa import (
    HM3D_DATASET,
    boolean,
    case,
    choice,
    count,
    measurement,
    order,
)
from dimos.evals.types import Suite

_case = partial(case, "hm3d_CFVBbU9Rsyb", "00337-CFVBbU9Rsyb", "HM3D_DATASET_CONFIG", HM3D_DATASET)
SUITE: Suite = [
    _case(
        "sofa_color",
        "What color is the sofa beneath the framed trousers display in the middle-level living area? A) Blue; B) Green; C) Red; D) White. Return only the letter.",
        choice("C"),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "kitchen_cabinet_color",
        "What color are the lower kitchen cabinets beside the dining table with rectangular placemats? A) Blue-gray; B) Red; C) Black; D) Yellow. Return only the letter.",
        choice("A"),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "washing_machine_exists",
        "Is there a washing machine in the scanned environment? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "washing_machine_location",
        "Which type of room contains the washing machine beneath the long worktop? A) Bedroom; B) Utility room; C) Living room; D) Bathroom. Return only the letter.",
        choice("B"),
        {"object-location", "single-choice"},
    ),
    _case(
        "utility_high_chair_count",
        "How many child high chairs stand in front of the utility-room worktop? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "fire_extinguisher_exists",
        "Is there a wall-mounted fire extinguisher beside a stair landing? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "red_sofa_cushion_count",
        "How many dark throw cushions are on the red sofa below the framed trousers display? Return only the count.",
        count(2),
        {"object-count", "count"},
    ),
    _case(
        "below_trousers_display",
        "What is directly below the framed trousers display in the red-sofa living area? A) Bed; B) Dining table; C) Washing machine; D) Sofa. Return only the letter.",
        choice("D"),
        {"object-location", "single-choice"},
    ),
    _case(
        "bed_below_skylight_exists",
        "Is there a bed beneath a skylight in a room with a sloped wooden ceiling? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "bunk_beds_exists",
        "Are there bunk beds in the upper-level sleeping area? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "blue_armchair_bedroom_wardrobe_state",
        "Is the wardrobe in the bedroom with the blue armchair and balcony doors open or closed? A) Open; B) Closed. Return only the letter.",
        choice("B"),
        {"object-state", "single-choice"},
    ),
    _case(
        "location_elevation_order",
        "Order these locations from lowest to highest floor elevation. A) Upper-level bunk-bed area; B) Utility room with the washing machine; C) Living area with the red sofa below the framed trousers display. Return all three letters once in order, optionally separated by commas.",
        order("BCA"),
        {"elevation", "ranking"},
    ),
    _case(
        "bunk_utility_height_difference",
        "Approximately how much higher is the floor of the bunk-bed area than the floor of the utility room, in meters? Return only the number.",
        measurement(5.60, 0.20, 1.0),
        {"elevation", "numeric"},
    ),
]
