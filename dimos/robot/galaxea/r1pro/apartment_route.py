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

"""Bounded clearance adjustments to a KronkNav route for measured held geometry."""

from itertools import pairwise
import time

import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport

# Transit reserves 6 cm for tracking and stops at 4 cm, leaving braking room.
CLASSICAL_NAVIGATION_CLEARANCE_M = 0.06
CLASSICAL_TRACKING_LIMIT_M = 0.04


def navigation_tracking_error(path: list[list[float]], pose: list[float]) -> float:
    """Bound deviation from a checked pose path, using a 1 m lever arm for yaw.

    Translation plus rotational displacement accounts for the carried envelope.
    Projection includes yaw, so an in-place turn remains on its checked path.
    """
    points = np.asarray(path, dtype=float).copy()
    actual = np.asarray(pose, dtype=float).copy()
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 2
        or actual.shape != (3,)
        or not np.isfinite(points).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("Tracking requires a finite pose and a complete planar path")
    points[:, 2] = np.unwrap(points[:, 2])
    actual[2] = points[0, 2] + np.arctan2(
        np.sin(actual[2] - points[0, 2]), np.cos(actual[2] - points[0, 2])
    )
    segments = np.diff(points, axis=0)
    fraction = np.clip(
        np.sum((actual - points[:-1]) * segments, axis=1)
        / np.maximum(np.sum(segments * segments, axis=1), 1e-12),
        0,
        1,
    )
    error = actual - points[:-1] - fraction[:, None] * segments
    return float(np.min(np.linalg.norm(error[:, :2], axis=1) + np.abs(error[:, 2])))


def apartment_approach(
    checker: PlanarTransport, preposition: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Choose a nearby transit stop and arrival turn with the route's clearance."""
    if preposition.shape != (3,) or not np.isfinite(preposition).all():
        raise ValueError("Expected a finite assessed approach pose")
    yaw = preposition[2]
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    docking_yaw = checker.start[2] + np.arctan2(
        np.sin(yaw - checker.start[2]), np.cos(yaw - checker.start[2])
    )
    for stand_off in (0.28, 0.33, 0.38, 0.43, 0.48):
        for lateral in (0.0, 0.05, -0.05):
            docking = preposition.copy()
            docking[:2] += -stand_off * forward + lateral * left
            docking[2] = docking_yaw
            transit = docking.copy()
            transit[2] = checker.start[2]
            if checker.clear_pose_segment(transit, docking):
                return transit, docking
    raise RuntimeError("No apartment approach and arrival turn with the required clearance")


def refine_apartment_route(checker: PlanarTransport, path: list[list[float]]) -> list[list[float]]:
    """Retain the native route and exact endpoints; execute only fully checked sweeps.

    KronkNav's circular graph clearance can miss wrist-camera protrusions in a
    tight passage. Search small local offsets of its interior waypoints, bounded
    to ten centimetres. This cannot substitute a different room-level route.
    """
    poses = np.asarray(path, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("Expected at least two finite planar poses")
    poses = np.vstack([checker.start, poses])
    poses[:, 2] = np.unwrap(poses[:, 2])

    def checked(candidate: NDArray[np.float64]) -> list[list[float]]:
        # A clear shortcut can introduce sharp corners the velocity controller
        # cannot track within its clearance. Keep the native route sampling.
        if not all(checker.clear_pose_segment(a, b) for a, b in pairwise(candidate)):
            raise RuntimeError("Full robot sweep exceeds the available route clearance")
        return [[float(value) for value in pose] for pose in candidate]

    try:
        return checked(poses)
    except RuntimeError:
        pass
    # Layered corridor search: each native waypoint keeps its yaw and may move
    # by at most 10 cm. Penalize changes in offset to avoid introducing a kink.
    # A whole-path translation cannot clear obstacles on alternating sides.
    deadline = time.monotonic() + 15
    offsets = np.array(
        [(0.0, 0.0)]
        + [
            (radius * x, radius * y)
            for radius in (0.025, 0.05, 0.075, 0.1)
            for x, y in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ]
    )
    layers = [poses[0:1]]
    costs = np.zeros(1)
    parents: list[NDArray[np.int64]] = []
    for index in range(1, len(poses)):
        candidates = np.tile(poses[index], (1 if index == len(poses) - 1 else len(offsets), 1))
        if index != len(poses) - 1:
            candidates[:, :2] += offsets
        current_costs = np.full(len(candidates), np.inf)
        predecessors = np.full(len(candidates), -1, dtype=np.int64)
        previous = layers[-1]
        previous_offsets = previous[:, :2] - poses[index - 1, :2]
        for node, candidate in enumerate(candidates):
            offset = candidate[:2] - poses[index, :2]
            changes = np.linalg.norm(previous_offsets - offset, axis=1)
            scores = costs + np.linalg.norm(offset) + 4 * changes
            # At most 5 cm of offset change per native edge; every swept edge
            # still needs to clear the complete robot and measured cargo.
            scores[changes > 0.05 + 1e-9] = np.inf
            for parent in np.argsort(scores):
                if time.monotonic() >= deadline:
                    raise RuntimeError("No clear full-body route within the native path corridor")
                if not np.isfinite(scores[parent]):
                    break
                if checker.clear_pose_segment(previous[parent], candidate):
                    current_costs[node] = scores[parent]
                    predecessors[node] = parent
                    break
        if not np.isfinite(current_costs).any():
            raise RuntimeError("No clear full-body route within ten centimetres of the native path")
        layers.append(candidates)
        parents.append(predecessors)
        costs = current_costs
    node = int(np.argmin(costs))
    result = [layers[-1][node]]
    for index in range(len(parents) - 1, -1, -1):
        node = int(parents[index][node])
        result.append(layers[index][node])
    return [[float(value) for value in pose] for pose in reversed(result)]
