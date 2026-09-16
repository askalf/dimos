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

"""Reviewed indoor QA for original furnished HSSD 104348463_171513588.

HSSD_DATASET_CONFIG points to the original downloaded dataset config.
Three physical rooms are user-confirmed. Kitchen and living measurements refer
to annotated zones, not separately enclosed rooms. Runtime requires a navmesh.
References: ../../../suite_draft/habitat/hssd/hssd_104348463_171513588.md.
"""

from collections.abc import Callable
import os
from typing import TypeVar

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.evals.environments.habitat import HabitatEnvironment
from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.types import EvalCase, Outcome, Suite

T = TypeVar("T")
INSTRUCTION = (
    "You are answering questions about a live simulated home. You control the robot, "
    "and its sensor recording grows as it observes the environment. Initial observations "
    "do not cover the whole home. Move around to gather the evidence needed to answer "
    "the question. Inspect relevant interior rooms for counts and absence claims. "
    "Only indoor areas are in scope. Use observations rather than assumptions about "
    "a typical home. When you have enough evidence, return the answer in the requested format."
)


def _environment() -> HabitatEnvironment:
    return HabitatEnvironment(
        scene_dataset_config=os.environ.get(
            "HSSD_DATASET_CONFIG",
            str(
                DIMOS_PROJECT_ROOT
                / "target/habitat/data/hssd-hab/hssd-hab.scene_dataset_config.json"
            ),
        ),
        scene_id="104348463_171513588",
        seed=0,
        blueprint=["habitat-nav", "mcp-server", "observe-skill"],
    )


def _parsed(parser: Callable[[str], T], score: Callable[[T], float]) -> Callable[[Outcome], float]:
    def grade(outcome: Outcome) -> float:
        try:
            value = parser(outcome.trajectory.final_answer)
        except ValueError:
            return 0.0
        return score(value)

    return grade


SUITE: Suite = [
    EvalCase(
        id="hssd_104348463_171513588_bedrooms",
        inputs=INSTRUCTION + "\n\nHow many bedrooms are in the home? Return only the count.",
        environment=_environment(),
        grade=_parsed(first_number, lambda v: exact(1, v)),
        timeout_s=1200,
        tags=frozenset({"rooms", "count"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_bathrooms",
        inputs=INSTRUCTION + "\n\nHow many bathrooms are in the home? Return only the count.",
        environment=_environment(),
        grade=_parsed(first_number, lambda v: exact(1, v)),
        timeout_s=1200,
        tags=frozenset({"rooms", "count"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_largest_room",
        inputs=INSTRUCTION
        + "\n\nWhich room has the largest floor area? A) Bedroom; B) Bathroom; C) Combined living/dining room. Return only A, B, or C.",
        environment=_environment(),
        grade=lambda o: exact("C", o.trajectory.final_answer.strip().upper()),
        timeout_s=1200,
        tags=frozenset({"rooms", "area", "single-choice"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_bedroom_area",
        inputs=INSTRUCTION
        + "\n\nWhat is the approximate bedroom floor area, in square meters? Return only the number.",
        environment=_environment(),
        grade=_parsed(first_number, lambda v: numeric(29.78, v, tolerance=2, band=7)),
        timeout_s=1200,
        tags=frozenset({"area", "numeric"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_kitchen_area",
        inputs=INSTRUCTION
        + "\n\nWhat is the approximate floor area of the kitchen zone, in square meters? Return only the number.",
        environment=_environment(),
        grade=_parsed(first_number, lambda v: numeric(26.30, v, tolerance=1.5, band=6)),
        timeout_s=1200,
        tags=frozenset({"area", "numeric"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_living_perimeter",
        inputs=INSTRUCTION
        + "\n\nWhat is the approximate perimeter of the living/dining area, excluding the kitchen zone, in meters? Return only the number.",
        environment=_environment(),
        grade=_parsed(first_number, lambda v: numeric(25.65, v, tolerance=1, band=4)),
        timeout_s=1200,
        tags=frozenset({"perimeter", "numeric"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_computer_location",
        inputs=INSTRUCTION
        + "\n\nWhich area contains the desktop computer? A) Kitchen; B) Bedroom; C) Bathroom; D) Living area. Return only A, B, C, or D.",
        environment=_environment(),
        grade=lambda o: exact("B", o.trajectory.final_answer.strip().upper()),
        timeout_s=1200,
        tags=frozenset({"object-location", "single-choice"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_televisions",
        inputs=INSTRUCTION + "\n\nHow many televisions are in the home? Return only the count.",
        environment=_environment(),
        grade=_parsed(first_number, lambda v: exact(2, v)),
        timeout_s=1200,
        tags=frozenset({"object-count", "count"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_computer_exists",
        inputs=INSTRUCTION
        + "\n\nIs there a desktop computer in the bedroom? Return only yes or no.",
        environment=_environment(),
        grade=_parsed(yes_no, lambda v: exact("yes", v)),
        timeout_s=1200,
        tags=frozenset({"existence", "boolean"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_round_table",
        inputs=INSTRUCTION
        + "\n\nWhat is the approximate diameter of the round dining table, in meters? Return only the number.",
        environment=_environment(),
        grade=_parsed(first_number, lambda v: numeric(1.52, v, tolerance=0.1, band=0.35)),
        timeout_s=1200,
        tags=frozenset({"dimensions", "numeric"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_area_order",
        inputs=INSTRUCTION
        + "\n\nOrder these areas from smallest to largest floor area. A) Kitchen zone; B) Bedroom; C) Bathroom. Return all three letters once in order, optionally separated by commas.",
        environment=_environment(),
        grade=_parsed(ranking, lambda v: rank_order("CAB", v)),
        timeout_s=1200,
        tags=frozenset({"area", "ranking"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_fridge_exists",
        inputs=INSTRUCTION + "\n\nIs there a refrigerator in the home? Return only yes or no.",
        environment=_environment(),
        grade=_parsed(yes_no, lambda v: exact("yes", v)),
        timeout_s=1200,
        tags=frozenset({"existence", "boolean"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_room_count",
        inputs=INSTRUCTION + "\n\nHow many rooms are in the home? Return only the count.",
        environment=_environment(),
        grade=_parsed(first_number, lambda v: exact(3, v)),
        timeout_s=1200,
        tags=frozenset({"rooms", "count"}),
    ),
    EvalCase(
        id="hssd_104348463_171513588_wardrobe_state",
        inputs=INSTRUCTION
        + "\n\nIs the wardrobe open or closed? A) Open; B) Closed. Return only A or B.",
        environment=_environment(),
        grade=lambda o: exact("A", o.trajectory.final_answer.strip().upper()),
        timeout_s=1200,
        tags=frozenset({"object-state", "single-choice"}),
    ),
]
