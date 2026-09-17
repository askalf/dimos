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

"""Launch and scoring helpers for user-reviewed Habitat suites."""

from collections.abc import Callable
from functools import partial
import os

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.evals.environments.habitat import HabitatEnvironment
from dimos.evals.scorers import exact, first_number, numeric, rank_order, ranking, yes_no
from dimos.evals.suites.lib.hssd_qa import INSTRUCTION, parsed
from dimos.evals.types import EvalCase, Outcome

DATA_ROOT = DIMOS_PROJECT_ROOT / "target/habitat/data/versioned_data"
HM3D_ROOT = DATA_ROOT / "hm3d-0.2/hm3d/example"
HM3D_DATASET = str(HM3D_ROOT / "hm3d_example_basis.scene_dataset_config.json")
HM3D_ANNOTATED_DATASET = str(HM3D_ROOT / "hm3d_annotated_example_basis.scene_dataset_config.json")
REPLICACAD_DATASET = str(DATA_ROOT / "replica_cad_dataset/replicaCAD.scene_dataset_config.json")
TEST_APARTMENT = str(DATA_ROOT / "habitat_test_scenes/apartment_1.glb")


def count(expected: int) -> Callable[[Outcome], float]:
    return parsed(first_number, partial(exact, expected))


def boolean(expected: str) -> Callable[[Outcome], float]:
    return parsed(yes_no, partial(exact, expected))


def choice(expected: str) -> Callable[[Outcome], float]:
    return lambda outcome: exact(expected, outcome.trajectory.final_answer.strip().upper())


def measurement(reference: float, tolerance: float, band: float) -> Callable[[Outcome], float]:
    return parsed(first_number, partial(numeric, reference, tolerance=tolerance, band=band))


def order(expected: str) -> Callable[[Outcome], float]:
    return parsed(ranking, partial(rank_order, expected))


def case(
    prefix: str,
    scene_id: str,
    dataset_env: str,
    default_dataset: str,
    suffix: str,
    question: str,
    grade: Callable[[Outcome], float],
    tags: set[str],
    *,
    scene_env: str | None = None,
) -> EvalCase:
    """Create an independent seeded episode with question-specific tags."""
    return EvalCase(
        id=f"{prefix}_{suffix}",
        inputs=INSTRUCTION + "\n\n" + question,
        environment=HabitatEnvironment(
            scene_dataset_config=os.environ.get(dataset_env, default_dataset),
            scene_id=os.environ.get(scene_env, scene_id) if scene_env else scene_id,
            seed=0,
            blueprint=["habitat-nav", "mcp-server", "observe-skill"],
        ),
        grade=grade,
        timeout_s=1200,
        tags=frozenset(tags),
    )
