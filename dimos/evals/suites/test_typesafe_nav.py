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
from dimos.evals.suites.typesafe_nav import ARRIVAL_BAND_M, GOAL, reached_goal
from dimos.evals.types import Outcome
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

SPAWN = (3.0, 2.0)


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


def test_robot_that_never_moved_scores_zero(tmp_path: Path) -> None:
    """Regression: odom jitter must not earn directness credit. This scored 0.3
    on the first live run, where the robot sat at spawn for 60 ticks."""
    assert reached_goal(_outcome(tmp_path, [SPAWN] * 50)) == 0.0


def test_straight_line_into_band_scores_arrival_plus_directness(tmp_path: Path) -> None:
    end = (GOAL.x + 0.4, GOAL.y - 0.4)  # inside the 2 m band
    n = 20
    path = [
        (SPAWN[0] + (end[0] - SPAWN[0]) * i / n, SPAWN[1] + (end[1] - SPAWN[1]) * i / n)
        for i in range(n + 1)
    ]
    score = reached_goal(_outcome(tmp_path, path))
    dist = ((GOAL.x - end[0]) ** 2 + (GOAL.y - end[1]) ** 2) ** 0.5
    assert score == pytest.approx(0.7 * (1 - dist / ARRIVAL_BAND_M) + 0.3 * 1.0)
