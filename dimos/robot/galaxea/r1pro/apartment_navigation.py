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

"""KronkNav map/goal bridge for the apartment's independent manipulation skills."""

from pathlib import Path as FilePath
import threading
import time
from typing import Any, Protocol

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.robot.galaxea.r1pro.primitive_spec import PrimitiveSimSpec
from dimos.spec.utils import Spec

APARTMENT_FRAME = "object_carrying_footprint"
APARTMENT_NAV_TASK = "object_navigation"


class ApartmentSimSpec(PrimitiveSimSpec, Protocol):
    def validate_apartment_posture(
        self, trajectory: JointTrajectory, index: int, arm: str
    ) -> None: ...
    def assess_object_reachability(
        self, primitive: str, index: int, arm: str = "auto", region: str = "tray"
    ) -> dict[str, Any]: ...
    def prepare_reachable_primitive(
        self, primitive: str, arm: str, index: int, region: str, stance: dict[str, Any]
    ) -> dict[str, Any]: ...
    def prepare_object_navigation(
        self, destination: str, arm: str = "right", stance: list[float] | None = None
    ) -> dict[str, Any]: ...
    def validate_object_navigation(self, path: list[list[float]]) -> list[list[float]]: ...
    def finish_object_navigation(self) -> None: ...


class ApartmentNavigationSpec(Spec, Protocol):
    def request_object_route(self, goal: list[float], offset: list[float], cloud: str) -> None: ...
    def object_route_status(self) -> dict[str, Any]: ...


class ApartmentNavigation(Module):
    base_odom: In[PoseStamped]
    navigation_tf: Out[TFMessage]
    global_map: Out[PointCloud2]
    goal: Out[PointStamped]
    planned_path: In[Path]
    surface_map: In[PointCloud2]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()
        self._offset = np.zeros(2)
        self._goal: np.ndarray | None = None
        self._path: Path | None = None
        self._requested = float("inf")
        self._map_ready = False
        self._pending_goal: PointStamped | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.base_odom.subscribe(self._odom)))
        self.register_disposable(Disposable(self.planned_path.subscribe(self._route)))
        self.register_disposable(Disposable(self.surface_map.subscribe(self._surface)))

    def _odom(self, pose: PoseStamped) -> None:
        with self._lock:
            yaw = pose.orientation.euler[2]
            c, s = np.cos(yaw), np.sin(yaw)
            xy = (
                np.array([pose.position.x, pose.position.y])
                + np.array([[c, -s], [s, c]]) @ self._offset
            )
        self.navigation_tf.publish(
            TFMessage(
                Transform(
                    translation=Vector3(float(xy[0]), float(xy[1]), 0),
                    rotation=pose.orientation,
                    frame_id="world",
                    child_frame_id=APARTMENT_FRAME,
                    ts=pose.ts,
                )
            )
        )

    def _route(self, path: Path) -> None:
        with self._lock:
            if path.ts >= self._requested:
                self._path = path

    def _surface(self, cloud: PointCloud2) -> None:
        with self._lock:
            if len(cloud) == 0:
                return
            self._map_ready = True
            pending, self._pending_goal = self._pending_goal, None
        if pending is not None:
            self.goal.publish(pending)

    @rpc
    def request_object_route(self, goal: list[float], offset: list[float], cloud: str) -> None:
        """Submit a full-map navigation request after the departure turn has completed."""
        target, footprint = np.asarray(goal), np.asarray(offset)
        if (
            target.shape != (3,)
            or footprint.shape != (2,)
            or not np.isfinite(np.r_[target, footprint]).all()
        ):
            raise ValueError("Expected finite planar goal and carrying-footprint offset")
        with self._lock:
            self._goal, self._offset = target, footprint
            self._path = None
            self._requested = time.time()
            c, s = np.cos(target[2]), np.sin(target[2])
            xy = target[:2] + np.array([[c, -s], [s, c]]) @ footprint
            message = PointStamped(x=float(xy[0]), y=float(xy[1]), z=0, frame_id="world")
            ready = self._map_ready
            self._pending_goal = None if ready else message
        if ready:
            self.goal.publish(message)
        else:
            points = np.load(FilePath(cloud), allow_pickle=False)
            self.global_map.publish(
                PointCloud2.from_numpy(points, frame_id="world", timestamp=time.time())
            )

    @rpc
    def object_route_status(self) -> dict[str, Any]:
        """Return only complete routes; physical sweep validation is still required."""
        with self._lock:
            path, goal = self._path, self._goal
            if path is None or goal is None:
                return dict(ready=False, map_ready=self._map_ready)
            if path.frame_id != "world":
                return dict(ready=False, error="KronkNav route must use the world frame")
            if len(path.poses) < 2:
                return dict(ready=False, error="KronkNav found no traversable route")
            c, s = np.cos(goal[2]), np.sin(goal[2])
            offset = np.array([[c, -s], [s, c]]) @ self._offset
            poses = np.array(
                [[p.position.x - offset[0], p.position.y - offset[1], goal[2]] for p in path.poses]
            )
            if not np.isfinite(poses).all() or np.linalg.norm(poses[-1, :2] - goal[:2]) > 0.02:
                return dict(ready=False, error="KronkNav returned an incomplete route")
            return dict(ready=True, path=poses.tolist())
