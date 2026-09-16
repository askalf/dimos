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

"""Common launch and plain-answer handling for scene-specific HSSD QA."""

from collections.abc import Callable
import os
from typing import TypeVar

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.evals.environments.habitat import HabitatEnvironment
from dimos.evals.types import EvalCase, Outcome

T = TypeVar("T")
INSTRUCTION = (
    "You are answering questions about a live simulated home. You control the robot, "
    "and its sensor recording grows as it observes the environment. Initial observations "
    "do not cover the whole home. Move around to gather the evidence needed to answer "
    "the question. Inspect relevant interior rooms for counts and absence claims. "
    "Indoor areas, including an attached garage, are in scope. Exterior openings may "
    "be observed from indoors; do not leave the home. Use observations rather than "
    "assumptions about a typical home. Return the answer in the requested format."
)


def parsed(parser: Callable[[str], T], score: Callable[[T], float]) -> Callable[[Outcome], float]:
    def grade(outcome: Outcome) -> float:
        try:
            value = parser(outcome.trajectory.final_answer)
        except ValueError:
            return 0.0
        return score(value)

    return grade


def case(
    scene_id: str, suffix: str, question: str, grade: Callable[[Outcome], float], tags: set[str]
) -> EvalCase:
    """Make a fresh Habitat environment without sharing episode state across cases."""
    return EvalCase(
        id=f"hssd_{scene_id}_{suffix}",
        inputs=INSTRUCTION + "\n\n" + question,
        environment=HabitatEnvironment(
            scene_dataset_config=os.environ.get(
                "HSSD_DATASET_CONFIG",
                str(
                    DIMOS_PROJECT_ROOT
                    / "target/habitat/data/hssd-hab/hssd-hab.scene_dataset_config.json"
                ),
            ),
            scene_id=scene_id,
            seed=0,
            blueprint=["habitat-nav", "mcp-server", "observe-skill"],
        ),
        grade=grade,
        timeout_s=1200,
        tags=frozenset(tags),
    )
