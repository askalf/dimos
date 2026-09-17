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
        --set scene_json=dimos/evals/suites/scenes/apartment_detections.json

``apartment_detections.json`` is DimSim's ground-truth snapshot of the
apartment: objects and walls with real extents, in the ROS world frame odometry
uses. Copied verbatim from PR #4208 (``misc/DimSim/scenes/apartment/
object_detections.json`` at 7fd0e2f17); regenerate with
``SceneClient.export_object_detections`` once that lands.
"""

from __future__ import annotations

from dimos.evals.environments.dimsim import DimSimEnvironment
from dimos.evals.scorers import ramp
from dimos.evals.types import EvalCase, Outcome, Suite, recording

GOAL_LABEL = "sectional"
# The couch's real footprint (minx, miny, maxx, maxy); test_typesafe_nav pins
# it to the scene file. The centre is inside the couch and unreachable, so
# arrival is measured to the box edge.
GOAL_BOX = (0.156, 3.028, 1.956, 5.735)
# Matches the DimSim-native go-to-couch rubric (objectDistance thresholdM: 2.0).
ARRIVAL_BAND_M = 2.0
MIN_TRAVEL_M = 0.1


def distance_to_box(x: float, y: float, box: tuple[float, float, float, float]) -> float:
    """Euclidean distance from a point to an axis-aligned box; 0 inside."""
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return (dx * dx + dy * dy) ** 0.5


def reached_goal(outcome: Outcome) -> float:
    """Where it stopped (70%) and how directly it got there (30%)."""
    with recording(outcome) as store:
        poses = [entry.data.position for entry in store.streams.odom]
    if not poses:
        raise LookupError("no odometry recorded")
    start, end = poses[0], poses[-1]
    arrival = ramp(distance_to_box(end.x, end.y, GOAL_BOX), band=ARRIVAL_BAND_M)
    travelled = sum((poses[i + 1] - poses[i]).length() for i in range(len(poses) - 1))
    ideal = distance_to_box(start.x, start.y, GOAL_BOX)
    # A robot that never moved has no path to be direct about; odom jitter must
    # not turn 1e-9 m of travel into full directness credit.
    directness = min(1.0, ideal / travelled) if travelled >= MIN_TRAVEL_M else 0.0
    return 0.7 * arrival + 0.3 * directness


SUITE: Suite = [
    EvalCase(
        id="typesafe_nav_couch",
        inputs="navigate to the couch",
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
