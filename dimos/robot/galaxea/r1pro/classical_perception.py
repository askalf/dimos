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

"""Simulated multi-view depth segmentation for the classical manipulation demo.

Object identities come from the scene, and XYZ points are ray intersections with
its current geometry. This isolates manipulation from detector errors; it is not
an RGB semantic perception benchmark or a sensor mounted on the real robot.
"""

import time

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def segmented_object_cloud(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    *,
    resolution: int = 64,
    radius: float = 0.16,
) -> PointCloud2:
    """Sample visible depth from five virtual views, retaining only the selected body.

    Rays hit the whole scene: a table, neighboring object or hand occludes points
    rather than becoming part of the selected object's input. No live data writes.
    """
    target = data.body(body_id).xpos.copy()
    chunks = []
    coordinates = np.linspace(-radius, radius, resolution)
    u, v = np.meshgrid(coordinates, coordinates)
    for offset in ((0, 0, 0.6), (0.5, 0, 0.3), (-0.5, 0, 0.3), (0, 0.5, 0.3), (0, -0.5, 0.3)):
        origin = target + offset
        forward = target - origin
        forward /= np.linalg.norm(forward)
        reference = (
            np.array([0.0, 1.0, 0.0]) if abs(forward[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
        )
        right = np.cross(forward, reference)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rays = target + u.ravel()[:, None] * right + v.ravel()[:, None] * up - origin
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        ids = np.full(len(rays), -1, dtype=np.int32)
        distances = np.zeros(len(rays))
        mujoco.mj_multiRay(  # type: ignore[attr-defined]
            model, data, origin, rays.ravel(), None, True, -1, ids, distances, None, len(rays), 2.0
        )
        valid = (ids >= 0) & (distances >= 0)
        valid &= model.geom_bodyid[np.maximum(ids, 0)] == body_id
        chunks.append(origin + distances[valid, None] * rays[valid])
    points: NDArray[np.float64] = np.concatenate(chunks)
    if len(points) < 32:
        raise RuntimeError("Selected object has insufficient visible depth; no grasp was requested")
    _, unique = np.unique(np.round(points / 0.0015).astype(np.int64), axis=0, return_index=True)
    return PointCloud2.from_numpy(
        points[unique].astype(np.float32), frame_id="world", timestamp=time.time()
    )
