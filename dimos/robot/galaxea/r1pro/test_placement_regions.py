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

"""Placement candidates respect physical fit, occupancy and region coordinates."""

import math

import pytest

from dimos.robot.galaxea.r1pro.placement_regions import (
    PlacementObstacle,
    PlacementRegion,
    placement_candidates,
)


def test_oversized_object_stops_without_attempting_rearrangement():
    region = PlacementRegion("tray", (0, 0, 0.7), (0.05, 0.03), ("floor",))
    assert placement_candidates(region, radius=0.04, half_height=0.08) == ()


def test_full_region_has_no_candidate_and_overhead_obstacle_does_not_block():
    region = PlacementRegion("desk", (0, 0, 0.7), (0.1, 0.1), ("desk_top",))
    occupied = PlacementObstacle((0, 0), (0.1, 0.1), 0.7, 0.8)
    assert placement_candidates(region, radius=0.02, half_height=0.05, obstacles=(occupied,)) == ()
    overhead = PlacementObstacle((0, 0), (0.1, 0.1), 1.8, 2.0)
    assert placement_candidates(region, radius=0.02, half_height=0.05, obstacles=(overhead,))[
        0
    ] == (0, 0, 0.75)


def test_rotated_region_contains_complete_footprint_and_support_height():
    region = PlacementRegion("counter", (1, 2, 0.8), (0.12, 0.04), ("top",), yaw=math.pi / 2)
    points = placement_candidates(region, radius=0.02, half_height=0.05)
    assert points[0] == pytest.approx((1, 2, 0.85))
    assert all(region.contains(p, 0.02, 0.05) for p in points)
    assert not region.contains((1.05, 2, 0.85), 0.02, 0.05)
    assert not region.contains((1, 2, 0.95), 0.02, 0.05)


def test_clearance_includes_open_gripper_not_only_object_footprint():
    region = PlacementRegion("small", (0, 0, 0.7), (0.025, 0.025), ("top",))
    neighbor = PlacementObstacle((0, 0.055), (0.01, 0.01), 0.7, 0.8)
    assert placement_candidates(
        region, radius=0.01, half_height=0.05, obstacles=(neighbor,), gripper_half_width=0.02
    )[0] == (0, 0, 0.75)
    assert placement_candidates(region, radius=0.01, half_height=0.05, obstacles=(neighbor,)) == ()


def test_turning_the_hand_changes_finger_clearance_without_changing_object_fit():
    region = PlacementRegion("small", (0, 0, 0.7), (0.025, 0.025), ("top",))
    neighbor = PlacementObstacle((0, 0.055), (0.01, 0.01), 0.7, 0.8)
    assert placement_candidates(region, radius=0.01, half_height=0.05, obstacles=(neighbor,)) == ()
    assert placement_candidates(
        region, radius=0.01, half_height=0.05, obstacles=(neighbor,), gripper_yaw=math.pi / 2
    ) == ((0.0, 0.0, 0.75),)
