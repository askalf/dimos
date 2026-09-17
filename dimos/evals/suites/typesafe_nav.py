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

"""Point-goal navigation driven by a TypeSafe (Jev) policy in DimSim.

The blueprint carries no skill container, so MCP exposes zero tools: the only
way the agent affects the world is the Twist it publishes.

    dimos evals run dimos.evals.suites.typesafe_nav \
        --agent dimos.evals.agents.typesafe_policy \
        --set scene_json=dimos/evals/suites/scenes/apartment_bed.json
"""

from __future__ import annotations

from pathlib import Path

from dimos.evals.environments.dimsim import DimSimEnvironment
from dimos.evals.scorers import ramp
from dimos.evals.types import EvalCase, Outcome, Suite, recording
from dimos.msgs.geometry_msgs.Vector3 import Vector3

SCENES = Path(__file__).parent / "scenes"
GOAL = Vector3(-3.567, -1.332, 0.0)


def reached_goal(outcome: Outcome) -> float:
    """Where it stopped (70%) and how directly it got there (30%)."""
    with recording(outcome) as store:
        poses = [entry.data.position for entry in store.streams.odom]
    if not poses:
        raise LookupError("no odometry recorded")
    arrival = ramp((GOAL - poses[-1]).length(), band=0.5)
    travelled = sum((poses[i + 1] - poses[i]).length() for i in range(len(poses) - 1))
    ideal = (GOAL - poses[0]).length()
    directness = min(1.0, ideal / travelled) if travelled > 0 else 0.0
    return 0.7 * arrival + 0.3 * directness


SUITE: Suite = [
    EvalCase(
        id="typesafe_nav_bed",
        inputs="navigate to the bed",
        environment=DimSimEnvironment(
            # No skill container: MCP comes up with zero tools exposed.
            blueprint=["unitree-go2", "mcp-server"],
            scene="apartment",
        ),
        grade=reached_goal,
        timeout_s=180.0,
        threshold=0.6,
        tags=frozenset({"typesafe", "navigation", "dimsim"}),
    )
]
