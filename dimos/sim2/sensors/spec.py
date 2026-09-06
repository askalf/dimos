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

"""Mounted devices reference concrete models, without a model-name registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Mount:
    link: str
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@runtime_checkable
class RayPattern(Protocol):
    @property
    def min_range(self) -> float: ...
    @property
    def max_range(self) -> float: ...

    def directions(self) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class Camera:
    name: str
    mount: Mount
    width: int = 640
    height: int = 480
    fovy: float = 60.0
    rate_hz: float = 10.0
    depth: bool = True


@dataclass(frozen=True)
class Lidar:
    name: str
    mount: Mount
    model: RayPattern
    rate_hz: float = 10.0
    # Optional world-frame cutoff in degrees for ideal mapping scans.
    maximum_world_elevation: float | None = None


@dataclass(frozen=True)
class Imu:
    name: str
    mount: Mount
    rate_hz: float = 200.0


Sensor: TypeAlias = Camera | Lidar | Imu
