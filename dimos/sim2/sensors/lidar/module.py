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

"""Ideal world-frame scan output for the existing DimOS mapping stack."""

import numpy as np
from pydantic import InstanceOf

from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.sim2.sensors.lidar.raycast import Raycaster
from dimos.sim2.sensors.module import SensorModule, SensorModuleConfig
from dimos.sim2.sensors.spec import Lidar


class LidarModuleConfig(SensorModuleConfig):
    sensor: InstanceOf[Lidar]
    root_body: str


class LidarModule(SensorModule):
    config: LidarModuleConfig
    pointcloud: Out[PointCloud2]

    def open(self) -> None:
        robot = self.config.robot_id
        self._site = self.reader.model.site(f"{robot}/sensor/{self.config.sensor.name}").id
        self._rays = self.config.sensor.model.directions()
        self._raycaster = Raycaster(
            self.reader.model,
            self.reader.model.body(f"{robot}/{self.config.root_body}").id,
        )

    def capture(self) -> None:
        data = self.reader.data
        origin = data.site_xpos[self._site]
        rays = self._rays @ data.site_xmat[self._site].reshape(3, 3).T
        elevation_limit = self.config.sensor.maximum_world_elevation
        if elevation_limit is not None:
            rays = rays[rays[:, 2] <= np.sin(np.deg2rad(elevation_limit)) + 1e-12]
        points = self._raycaster.cast(
            data,
            origin,
            rays,
            self.config.sensor.model.min_range,
            self.config.sensor.model.max_range,
        )
        self.pointcloud.publish(
            PointCloud2.from_numpy(
                points.astype(np.float32),
                frame_id="world",
                timestamp=self.reader.timestamp,
            )
        )
