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

"""Visual-reference questions for the ornate HM3D scan."""

from functools import partial

from dimos.evals.suites.lib.habitat_qa import HM3D_DATASET, boolean, case, choice, count
from dimos.evals.types import Suite

_case = partial(case, "hm3d_NBg5UqG3di3", "00770-NBg5UqG3di3", "HM3D_DATASET_CONFIG", HM3D_DATASET)
SUITE: Suite = [
    _case(
        "corridor_panel_color",
        "What color are the wall panels in the corridor with the gilded vaulted ceiling? A) Green; B) White; C) Red; D) Blue. Return only the letter.",
        choice("C"),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "floor_pattern",
        "What is the dominant pattern of the wood floor in the blue-patterned room? A) Checkerboard; B) Herringbone; C) Plain parallel strips; D) Hexagons. Return only the letter.",
        choice("B"),
        {"visual-attribute", "single-choice"},
    ),
    _case(
        "fireplace_exists",
        "Is there a fireplace in the room with blue patterned upper walls and wooden lower panels? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "fireplace_location",
        "Which room contains the fireplace beneath the exposed wooden ceiling? A) Red-and-gold corridor; B) Blue-patterned room with wooden lower walls; C) White corridor; D) Pale-blue decorative room. Return only the letter.",
        choice("B"),
        {"object-location", "single-choice"},
    ),
    _case(
        "radiator_below_window",
        "Are there radiators beneath the windows in the blue-patterned room? Return only yes or no.",
        boolean("yes"),
        {"spatial-relation", "boolean"},
    ),
    _case(
        "blue_room_windows",
        "How many windows are on the long exterior wall of the blue-patterned room? Return only the count.",
        count(3),
        {"object-count", "count"},
    ),
    _case(
        "hallway_extinguisher",
        "Is there a fire extinguisher in the white corridor? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "mirrored_doors",
        "Are there mirrored door panels at an entrance to a white-paneled room? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "arched_passage",
        "Is there an arched passage in the pale-blue decorative room? Return only yes or no.",
        boolean("yes"),
        {"existence", "boolean"},
    ),
    _case(
        "open_white_door",
        "Is the white door leading from the red-and-gold corridor into a white room open or closed? A) Open; B) Closed. Return only the letter.",
        choice("A"),
        {"object-state", "single-choice"},
    ),
]
