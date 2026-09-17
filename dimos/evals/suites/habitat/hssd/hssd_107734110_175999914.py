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

"""User-reviewed hssd_107734110_175999914 QA; scene context is in ../SCENES.md."""

from functools import partial

from dimos.evals.scorers import exact, first_number, numeric, yes_no
from dimos.evals.suites.lib.hssd_qa import case, parsed
from dimos.evals.types import Suite

_case = partial(case, "107734110_175999914")
SUITE: Suite = [
    _case(
        "bedrooms",
        "How many bedrooms are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(1, v)),
        {"rooms", "count"},
    ),
    _case(
        "closets",
        "How many separate closet spaces are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(2, v)),
        {"rooms", "count"},
    ),
    _case(
        "piano_location",
        "Which room contains the piano? A) Bedroom; B) Office; C) Kitchen; D) Living room. Return only A, B, C, or D.",
        lambda o: exact("D", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "computer_location",
        "Which room contains the desktop computer? A) Office; B) Living room; C) Kitchen; D) Bedroom. Return only A, B, C, or D.",
        lambda o: exact("A", o.trajectory.final_answer.strip().upper()),
        {"object-location", "single-choice"},
    ),
    _case(
        "sofa_office",
        "Is there a sofa in the office? Return only yes or no.",
        parsed(yes_no, lambda v: exact("yes", v)),
        {"existence", "boolean"},
    ),
    _case(
        "televisions",
        "How many televisions are in the home? Return only the count.",
        parsed(first_number, lambda v: exact(3, v)),
        {"object-count", "count"},
    ),
    _case(
        "living_area",
        "What is the approximate living-room area, in square meters? Return only the number.",
        parsed(first_number, lambda v: numeric(51.53, v, tolerance=3, band=10)),
        {"area", "numeric"},
    ),
    _case(
        "piano_width",
        "What is the approximate width of the digital piano, in meters? Return only the number.",
        parsed(first_number, lambda v: numeric(1.33, v, tolerance=0.08, band=0.3)),
        {"dimensions", "numeric"},
    ),
    _case(
        "office_doorway_radius",
        "What is the largest circular robot radius that fits through the office doorway in 2D, based on the structural opening width, in meters? Return only the number.",
        # Stage slice Y=1 m at X=-4.79/-4.85/-4.90: Z gap [-3.997643,-3.017643].
        # Width .98 m / 2 is structural clearance only, not a furnished-route guarantee.
        parsed(first_number, lambda v: numeric(0.49, v, tolerance=0.025, band=0.10)),
        {"clearance", "numeric"},
    ),
    _case(
        "piano_computer_distance",
        "How far is the piano from the office computer in a horizontal straight line, in meters? Return only the number.",
        # Horizontal visual-AABB centers including node transforms/instance scale: 11.635427 m.
        parsed(first_number, lambda v: numeric(11.64, v, tolerance=0.5, band=2)),
        {"distance", "numeric"},
    ),
]
