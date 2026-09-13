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

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.home_kinematics import bounded_joint_positions


def test_ik_rounding_produces_a_legal_planner_goal_without_moving_interior_joints():
    positions = np.array([-2.0494900921325745, 0.30419309, 0.125])
    lower = np.array([-2.04949, -2.04949, -2.04949])
    upper = np.array([0.304193, 0.304193, 0.304193])

    result = bounded_joint_positions(positions, lower, upper)

    np.testing.assert_array_equal(result, [-2.04949, 0.304193, 0.125])


@pytest.mark.parametrize("position", [-2.050, 0.305, float("nan"), float("inf")])
def test_invalid_ik_goals_are_rejected_instead_of_clipped(position):
    with pytest.raises(RuntimeError, match="exceeds conservative joint limits"):
        bounded_joint_positions(np.array([position]), np.array([-2.04949]), np.array([0.304193]))
