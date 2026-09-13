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

import time

import numpy as np

from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport


def refine_apartment_route(checker: PlanarTransport, path: list[list[float]]) -> list[list[float]]:
    """Retain the native route and exact endpoints; execute only fully checked sweeps.

    KronkNav's circular graph clearance can miss wrist-camera protrusions in a
    tight passage. Try small parallel offsets of its interior waypoints, bounded
    to ten centimetres. This cannot substitute a different room-level route.
    """
    poses = np.asarray(path, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("Expected at least two finite planar poses")
    try:
        return checker.shorten_path(path)
    except RuntimeError:
        pass
    deadline = time.monotonic() + 15
    for radius in (0.025, 0.05, 0.075, 0.1):
        for direction in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if time.monotonic() >= deadline:
                raise RuntimeError("No clear full-body route within the native path corridor")
            adjusted = poses.copy()
            adjusted[1:-1, :2] += radius * np.asarray(direction)
            try:
                return checker.shorten_path(adjusted.tolist())
            except RuntimeError:
                continue
    raise RuntimeError("No clear full-body route within ten centimetres of the native path")
