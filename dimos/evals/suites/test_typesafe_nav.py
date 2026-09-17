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

"""The grader over synthetic odom recordings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from dimos.evals.agents.lib.trajectory_builder import TrajectoryBuilder
from dimos.evals.agents.typesafe_policy import load_scene
from dimos.evals.scorers import ramp
from dimos.evals.suites.typesafe_nav import (
    ARRIVAL_BAND_M,
    GOAL_BOX,
    GOAL_LABEL,
    distance_to_box,
    reached_goal,
)
from dimos.evals.types import Outcome
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

SCENE = Path(__file__).parent / "scenes" / "apartment_detections.json"
SPAWN = (3.0, 2.0)  # DimSim apartment spawnPoint, in the ROS world frame


def _outcome(tmp_path: Path, xy: Sequence[tuple[float, float]]) -> Outcome:
    db = tmp_path / "memory.db"
    store = SqliteStore(path=str(db))
    odom = store.stream("odom", PoseStamped)
    for i, (x, y) in enumerate(xy):
        odom.append(
            PoseStamped(position=(x, y, 0.0), orientation=(0, 0, 0, 1), frame_id="world"),
            ts=1000.0 + i,
        )
    store.stop()
    trajectory = TrajectoryBuilder("navigate to the couch", name="test").build("answer")
    return Outcome(trajectory=trajectory, artifacts={"recording": db})


def test_goal_box_matches_the_scene_file() -> None:
    assert load_scene(SCENE, GOAL_LABEL).goal_box == GOAL_BOX


@pytest.mark.parametrize(
    ("xy", "expected"),
    [
        ((1.0, 4.0), 0.0),  # inside
        ((1.0, 2.028), 1.0),  # straight below the bottom edge
        ((3.0, 2.0), ((3.0 - 1.956) ** 2 + (2.0 - 3.028) ** 2) ** 0.5),  # off a corner
    ],
)
def test_distance_to_box(xy: tuple[float, float], expected: float) -> None:
    assert distance_to_box(*xy, GOAL_BOX) == pytest.approx(expected)


def test_robot_that_never_moved_earns_no_directness(tmp_path: Path) -> None:
    """Regression: odom jitter must not earn directness credit. Sitting at
    spawn is still 1.46 m from the couch, inside the 2 m band, so arrival
    alone is what remains."""
    score = reached_goal(_outcome(tmp_path, [SPAWN] * 50))
    assert score == pytest.approx(0.7 * ramp(distance_to_box(*SPAWN, GOAL_BOX), ARRIVAL_BAND_M))


def test_touching_the_couch_scores_full_arrival(tmp_path: Path) -> None:
    """Where the second live run actually ended: pressed against the couch."""
    end = (1.23, 3.19)  # inside the box, 0.16 m past its south edge
    n = 20
    path = [
        (SPAWN[0] + (end[0] - SPAWN[0]) * i / n, SPAWN[1] + (end[1] - SPAWN[1]) * i / n)
        for i in range(n + 1)
    ]
    score = reached_goal(_outcome(tmp_path, path))
    ideal = distance_to_box(*SPAWN, GOAL_BOX)
    travelled = ((end[0] - SPAWN[0]) ** 2 + (end[1] - SPAWN[1]) ** 2) ** 0.5
    assert score == pytest.approx(0.7 * 1.0 + 0.3 * min(1.0, ideal / travelled))
    assert score >= 0.6
