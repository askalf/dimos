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
import math

from dimos_lcm.vision_msgs import (
    BoundingBox2D,
    BoundingBox3D,
    Detection2D,
    Detection3D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Pose2D,
)
import numpy as np

from dimos.agents.typesafe.world_state import bearing_word, build_world_state, distance_word
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray


def _pose(x: float, y: float, yaw_deg: float) -> PoseStamped:
    q = PoseStamped(position=(x, y, 0.4), frame_id="world")
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion

    q.orientation = Quaternion.from_euler(Vector3(0, 0, math.radians(yaw_deg)))
    return q


def _det3d(label: str, x: float, y: float, score: float = 0.9) -> Detection3DArray:
    d = Detection3D()
    d.header = Header(1.0, "world")
    d.results = [ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id=label, score=score))]
    d.results_length = 1
    d.bbox = BoundingBox3D(center=Pose(position=(x, y, 0.3)), size=Vector3(0.5, 0.5, 0.6))
    return Detection3DArray(detections_length=1, header=Header(1.0, "world"), detections=[d])


def _det2d(label: str, cx: float, w: float = 100, h: float = 100) -> Detection2DArray:
    d = Detection2D()
    d.header = Header(1.0, "camera")
    d.results = [ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id=label, score=0.8))]
    d.results_length = 1
    center = Pose2D()
    center.position.x, center.position.y = cx, 360.0
    d.bbox = BoundingBox2D(center=center, size_x=w, size_y=h)
    return Detection2DArray(detections_length=1, header=Header(1.0, "camera"), detections=[d])


def test_pose_to_json_heading() -> None:
    j = _pose(1, 2, 90).to_json()
    assert j["position"] == {"x": 1.0, "y": 2.0, "z": 0.4}
    assert j["yaw_deg"] == 90.0
    assert j["heading"] == "north"
    assert _pose(0, 0, -45).to_json()["heading"] == "south_east"


def test_bearing_and_distance_words() -> None:
    assert bearing_word(0.0) == "ahead"
    assert bearing_word(math.pi / 2) == "left"
    assert bearing_word(-math.pi / 2) == "right"
    assert bearing_word(math.pi) == "behind"
    assert [distance_word(d) for d in (0.2, 1.0, 3.0, 6.0)] == ["touching", "near", "mid", "far"]


def test_objects_3d_relative_to_pose() -> None:
    # robot at origin facing +y (north): a chair at (-2, 0) is to its left.
    state = build_world_state(
        goal="go to the chair", pose=_pose(0, 0, 90), detections_3d=_det3d("chair", -2.0, 0.0)
    )
    (obj,) = state["objects"]
    assert obj["label"] == "chair"
    assert obj["bearing"] == "left"
    assert obj["distance"] == "mid"
    assert obj["distance_m"] == 2.0
    assert "objects" not in state.get("unavailable", [])


def test_objects_2d_bearing_and_size() -> None:
    state = build_world_state(
        goal="x", pose=None, detections_2d=_det2d("person", cx=1200.0, w=640, h=600)
    )
    (obj,) = state["objects"]
    assert obj["bearing"] == "far_right"
    assert obj["size"] == "filling_view"
    assert "pose" in state["unavailable"]


def test_pointcloud_sectors_in_robot_frame() -> None:
    pose = _pose(1.0, 1.0, 90)  # facing north
    pts = np.array(
        [[1.0, 1.3, 0.5], [3.0, 1.0, 0.5], [1.0, -2.0, 0.5], [1.0, 1.0, 5.0]]
    )  # ahead 0.3, right 2, behind 3, above (dropped)
    room = PointCloud2.from_numpy(pts, frame_id="world").to_json(pose, max_points=10)
    s = room["sectors"]
    assert s["ahead"]["state"] == "blocked" and s["ahead"]["clear_m"] == 0.3
    assert s["right"]["state"] == "clear" and s["right"]["clear_m"] == 2.0
    assert s["behind"]["clear_m"] == 3.0
    assert s["left"]["clear_m"] == 5.0
    assert len(room["points"]) == 3


def test_world_state_marks_missing_inputs() -> None:
    state = build_world_state(goal="", pose=None)
    assert state["goal"] == ""
    assert state["objects"] == []
    assert state["unavailable"] == ["pose", "objects", "room"]
