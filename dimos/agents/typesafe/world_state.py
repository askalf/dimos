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
"""Fold the latest perception messages into the JSON state the model reads."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from dimos.msgs.sensor_msgs.PointCloud2 import SECTOR_NAMES

if TYPE_CHECKING:
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
    from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray

_BEARINGS_2D = ("far_left", "left", "center", "right", "far_right")


def bearing_word(rel_angle: float) -> str:
    """8-way word for an angle in the robot frame (0 = ahead, +pi/2 = left)."""
    return SECTOR_NAMES[round(rel_angle / (math.pi / 4)) % 8]


def distance_word(d: float) -> str:
    if d < 0.5:
        return "touching"
    if d < 1.5:
        return "near"
    if d < 4.0:
        return "mid"
    return "far"


def _objects_3d(
    dets: Detection3DArray, pose: PoseStamped | None, max_objects: int
) -> list[dict[str, Any]]:
    out = dets.to_json()
    if pose is not None:
        for o in out:
            dx, dy = o["position"]["x"] - pose.x, o["position"]["y"] - pose.y
            d = math.hypot(dx, dy)
            o["distance_m"] = round(d, 2)
            o["distance"] = distance_word(d)
            rel = math.atan2(dy, dx) - pose.yaw
            o["bearing"] = bearing_word(rel)
            o["bearing_deg"] = round(math.degrees(math.atan2(math.sin(rel), math.cos(rel))), 1)
        out.sort(key=lambda o: o["distance_m"])
    return out[:max_objects]


def _objects_2d(
    dets: Detection2DArray, image_width: int, image_height: int, max_objects: int
) -> list[dict[str, Any]]:
    out = dets.to_json()
    for o in out:
        b = o["bbox"]
        frac = b["cx"] / image_width if image_width else 0.5
        o["bearing"] = _BEARINGS_2D[min(4, max(0, int(frac * 5)))]
        area = (b["w"] * b["h"]) / float(image_width * image_height or 1)
        o["size"] = (
            "filling_view"
            if area > 0.4
            else "large"
            if area > 0.15
            else "medium"
            if area > 0.03
            else "small"
        )
    out.sort(key=lambda o: -(o["bbox"]["w"] * o["bbox"]["h"]))
    return out[:max_objects]


def build_world_state(
    *,
    goal: str | None,
    pose: PoseStamped | None,
    detections_3d: Detection3DArray | None = None,
    detections_2d: Detection2DArray | None = None,
    lidar: PointCloud2 | None = None,
    robot: dict[str, Any] | None = None,
    max_objects: int = 20,
    image_width: int = 1280,
    image_height: int = 720,
    lidar_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unavailable: list[str] = []
    state: dict[str, Any] = {"goal": goal or ""}

    robot_section: dict[str, Any] = dict(robot or {})
    if pose is not None:
        robot_section.update(pose.to_json())
    else:
        unavailable.append("pose")
    state["robot"] = robot_section

    if detections_3d is not None:
        state["objects"] = _objects_3d(detections_3d, pose, max_objects)
    elif detections_2d is not None:
        state["objects"] = _objects_2d(detections_2d, image_width, image_height, max_objects)
    else:
        state["objects"] = []
        unavailable.append("objects")

    if lidar is not None:
        state["room"] = lidar.to_json(pose, **(lidar_kwargs or {}))
    else:
        unavailable.append("room")

    if unavailable:
        state["unavailable"] = unavailable
    return state
