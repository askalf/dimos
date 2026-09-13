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

"""Preserve navigation corner sampling and stop pose deviation before collision."""

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.apartment_route import (
    apartment_approach,
    navigation_tracking_error,
    refine_apartment_route,
)
from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport


def test_clear_native_corners_are_retained_for_the_velocity_controller(mocker):
    checker = mocker.Mock(spec=PlanarTransport)
    checker.start = np.zeros(3)
    checker.clear_pose_segment.return_value = True
    path = [[0.2, 0, 0], [0.4, 0.1, 0], [0.5, 0.3, 0], [0.5, 0.5, 0]]

    result = refine_apartment_route(checker, path)

    assert result == [[0, 0, 0], *path]
    checker.shorten_path.assert_not_called()


def test_tracking_allowance_accounts_for_translation_and_rotational_displacement():
    assert navigation_tracking_error([[0, 0, 0], [1, 0, 0]], [0.5, 0.03, 0.02]) == pytest.approx(
        0.05
    )


def test_in_place_turn_across_yaw_wrap_stays_on_its_checked_path():
    path = [[1, 2, np.deg2rad(179)], [1, 2, np.deg2rad(-179)]]
    assert navigation_tracking_error(path, [1, 2, np.deg2rad(-179.5)]) == pytest.approx(0)


def test_duplicate_start_does_not_hide_a_lateral_departure():
    path = [[0, 0, 0], [0, 0, 0], [1, 0, 0]]
    assert navigation_tracking_error(path, [0.5, 0.06, 0]) == pytest.approx(0.06)


def test_nonfinite_odometry_cannot_bypass_tracking_guard():
    with pytest.raises(ValueError, match="finite pose"):
        navigation_tracking_error([[0, 0, 0], [1, 0, 0]], [float("nan"), 0, 0])


def test_arrival_checks_the_short_turn_across_pi(mocker):
    checker = mocker.Mock(spec=PlanarTransport)
    checker.start = np.array([0, 0, np.deg2rad(179)])
    checker.clear_pose_segment.return_value = True
    transit, docking = apartment_approach(checker, np.array([1, 2, np.deg2rad(-179)]))
    assert docking[2] - transit[2] == pytest.approx(np.deg2rad(2))
