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

"""Geometric placement on named horizontal support regions, independent of ACT."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlacementRegion:
    """A measured horizontal support rectangle in world coordinates."""

    name: str
    center: tuple[float, float, float]
    half_size: tuple[float, float]
    support_geoms: tuple[str, ...]
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.support_geoms
            or not np.isfinite((*self.center, *self.half_size, self.yaw)).all()
            or min(self.half_size) <= 0
        ):
            raise ValueError("A region needs finite geometry, positive area and physical supports")

    @property
    def rotation(self) -> NDArray[np.float64]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

    def contains(
        self,
        position: tuple[float, float, float],
        radius: float,
        half_height: float,
        tolerance: float = 0.008,
    ) -> bool:
        relative = self.rotation.T @ (np.asarray(position[:2]) - self.center[:2])
        return bool(
            np.all(np.abs(relative) + radius <= self.half_size)
            and abs(position[2] - half_height - self.center[2]) <= tolerance
        )


@dataclass(frozen=True)
class PlacementObstacle:
    """World-aligned footprint and vertical bounds of another body or tray rim."""

    center: tuple[float, float]
    half_size: tuple[float, float]
    bottom: float
    top: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite((*self.center, *self.half_size, self.bottom, self.top)).all()
            or min(self.half_size) < 0
            or self.bottom > self.top
        ):
            raise ValueError("Obstacle bounds must be finite and ordered")


def placement_candidates(
    region: PlacementRegion,
    *,
    radius: float,
    half_height: float,
    obstacles: tuple[PlacementObstacle, ...] = (),
    clearance: float = 0.008,
    gripper_half_width: float = 0.065,
    spacing: float = 0.02,
) -> tuple[tuple[float, float, float], ...]:
    """Return stable empty goals, nearest the region centre first; empty means full.

    Radius conservatively encloses the object's XY footprint. The open gripper
    extends along world Y; clearance for approach/release is separate from fit.
    Reachability and demonstrated workspace coverage must still be checked by
    the caller. This routine neither moves nor rearranges existing objects.
    """
    if (
        not np.isfinite((radius, half_height, clearance, gripper_half_width, spacing)).all()
        or min(radius, half_height, spacing) <= 0
        or min(clearance, gripper_half_width) < 0
    ):
        raise ValueError("Use positive object dimensions/spacing and nonnegative clearances")
    bounds = np.asarray(region.half_size) - radius - clearance
    if np.any(bounds < 0):
        return ()
    # Centre-anchored integer lattice includes the exact centre of a small region.
    axes = [
        np.arange(-math.floor(b / spacing), math.floor(b / spacing) + 1) * spacing for b in bounds
    ]
    offsets = sorted(
        ((float(x), float(y)) for x in axes[0] for y in axes[1]),
        key=lambda xy: (xy[0] ** 2 + xy[1] ** 2, xy[0], xy[1]),
    )
    half_footprint = np.array([radius + clearance, max(radius, gripper_half_width) + clearance])
    candidates = []
    for offset in offsets:
        xy = np.asarray(region.center[:2]) + region.rotation @ offset
        if any(
            obstacle.top > region.center[2] + 0.002
            and obstacle.bottom < region.center[2] + 2 * half_height + 0.08
            and np.all(
                np.abs(xy - obstacle.center) < np.asarray(obstacle.half_size) + half_footprint
            )
            for obstacle in obstacles
        ):
            continue
        candidates.append((float(xy[0]), float(xy[1]), region.center[2] + half_height))
    return tuple(candidates)
