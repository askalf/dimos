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

"""Projection math shared by the standalone Python 3.9 renderer and offline tests."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def perspective(fov_y: float, aspect: float, near: float, far: float) -> NDArray[np.float64]:
    scale = 1 / math.tan(fov_y / 2)
    return np.array(
        [
            [scale / aspect, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float64,
    )


def ceiling_cutaway(
    projection: NDArray[np.float64], view: NDArray[np.float64], height: float
) -> NDArray[np.float64]:
    """Replace the near plane with world Y=height, retaining geometry below it.

    An ordinary near plane tilts with the camera and cuts through the room when
    orbiting. Oblique clipping keeps this cut horizontal for every orbit angle.
    The eye must remain above the cut plane.
    """
    plane = np.linalg.inv(view).T @ np.array([0, -1, 0, height])
    corner = np.linalg.inv(projection) @ np.array(
        [1 if plane[0] >= 0 else -1, 1 if plane[1] >= 0 else -1, 1, 1]
    )
    denominator = float(plane @ corner)
    if denominator <= 1e-8:
        raise ValueError("Cutaway plane does not intersect the camera frustum")
    result = projection.copy()
    result[2, :] = plane * (2 / denominator) - projection[3, :]
    return result
