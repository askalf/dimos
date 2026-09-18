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

"""The JSON state the model reads: goal, pose, objects with word buckets, room sectors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

from typing_extensions import NotRequired

from dimos.msgs.geometry_msgs.PoseStamped import PoseJson, XyzJson
from dimos.msgs.sensor_msgs.PointCloud2 import SECTOR_NAMES, SectorJson
from dimos.msgs.vision_msgs.Detection2DArray import BBoxJson

if TYPE_CHECKING:
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
    from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray

MAX_OBJECTS = 20  # closest first; the model reads words, not a scene graph
_DISTANCE = ((0.5, "touching"), (1.5, "near"), (4.0, "mid"), (math.inf, "far"))
_BEARING_2D = ("far_left", "left", "center", "right", "far_right")
_SIZE_2D = ((0.4, "filling_view"), (0.15, "large"), (0.03, "medium"), (-1.0, "small"))


class ObjectState(TypedDict):
    label: str
    score: float
    bearing: str
    position: NotRequired[XyzJson]  # 3D
    bearing_deg: NotRequired[float]
    distance: NotRequired[str]
    distance_m: NotRequired[float]
    bbox: NotRequired[BBoxJson]  # 2D
    size: NotRequired[str]


class RobotState(PoseJson):
    motion: str


class WorldState(TypedDict):
    goal: str
    robot: RobotState
    objects: list[ObjectState]
    room: NotRequired[dict[str, SectorJson]]


def bearing_word(rel: float) -> str:
    """8-way word for an angle in the robot frame (0 ahead, +pi/2 left)."""
    return SECTOR_NAMES[round(rel / (math.pi / 4)) % 8]


def distance_word(d: float) -> str:
    return next(word for limit, word in _DISTANCE if d < limit)


def _objects_3d(dets: Detection3DArray, pose: PoseStamped) -> list[ObjectState]:
    out: list[ObjectState] = []
    for d in dets.to_json():
        dx, dy = d["position"]["x"] - pose.x, d["position"]["y"] - pose.y
        dist, rel = math.hypot(dx, dy), math.atan2(dy, dx) - pose.yaw
        out.append(
            {
                "label": d["label"],
                "score": d["score"],
                "position": d["position"],
                "bearing": bearing_word(rel),
                "bearing_deg": round(math.degrees(math.atan2(math.sin(rel), math.cos(rel))), 1),
                "distance": distance_word(dist),
                "distance_m": round(dist, 2),
            }
        )
    return sorted(out, key=lambda o: o["distance_m"])[:MAX_OBJECTS]


def _objects_2d(dets: Detection2DArray, image_size: tuple[int, int]) -> list[ObjectState]:
    w, h = image_size
    out: list[ObjectState] = []
    for d in dets.to_json():
        b = d["bbox"]
        area = b["w"] * b["h"] / (w * h)
        out.append(
            {
                "label": d["label"],
                "score": d["score"],
                "bbox": b,
                "bearing": _BEARING_2D[min(4, int(5 * b["cx"] / w))],
                "size": next(word for limit, word in _SIZE_2D if area > limit),
            }
        )
    return sorted(out, key=lambda o: -o["bbox"]["w"] * o["bbox"]["h"])[:MAX_OBJECTS]


def build_world_state(
    goal: str,
    pose: PoseStamped,
    motion: str,
    *,
    detections_3d: Detection3DArray | None,
    detections_2d: Detection2DArray | None,
    lidar: PointCloud2 | None,
    image_size: tuple[int, int],
    lidar_band: tuple[float, float, float],
) -> WorldState:
    if detections_3d is not None:
        objects = _objects_3d(detections_3d, pose)
    elif detections_2d is not None:
        objects = _objects_2d(detections_2d, image_size)
    else:
        objects = []
    state: WorldState = {
        "goal": goal,
        "robot": {**pose.to_json(), "motion": motion},
        "objects": objects,
    }
    if lidar is not None:
        z_min, z_max, max_range = lidar_band
        state["room"] = lidar.to_json(pose, z_min=z_min, z_max=z_max, max_range=max_range)
    return state
