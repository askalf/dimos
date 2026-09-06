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

from dataclasses import replace
from uuid import uuid4

import mujoco
import numpy as np
import pytest

from dimos.robot.unitree.g1.sim2 import G1_GROOT
from dimos.sim2.sensors.lidar.module import LidarModule
from dimos.sim2.sensors.reader import WorldReader
from dimos.sim2.sensors.spec import Lidar

pytestmark = pytest.mark.mujoco


@pytest.fixture
def module():
    sensor = next(s for s in G1_GROOT.sensors if isinstance(s, Lidar))
    instance = LidarModule(
        robot_id="g1", root_body="pelvis", sensor=sensor, instance_name=f"test-lidar-{uuid4().hex}"
    )
    try:
        yield instance
    finally:
        instance.stop()


@pytest.mark.parametrize("pitch", [0.0, -0.35, 0.35])
def test_g1_scan_excludes_ceiling_after_mount_rotation(pitch, module, mocker):
    sensor = module.config.sensor
    mount = sensor.mount
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <geom type="plane" size="10 10 0.1"/>
            <geom type="box" pos="0 0 3" size="10 10 0.1"/>
            <body name="g1/pelvis" pos="0 0 1" euler="0 {pitch} 0">
              <geom type="sphere" size="0.1"/>
              <site name="g1/sensor/lidar" pos="{" ".join(map(str, mount.xyz))}"
                    euler="{" ".join(map(str, mount.rpy))}"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    module.reader = mocker.Mock(spec=WorldReader, model=model, data=data, timestamp=123.0)
    publish = mocker.patch.object(module.pointcloud, "publish")
    module.open()

    module.capture()

    publish.assert_called_once()
    cloud = publish.call_args.args[0]
    points = cloud.points().numpy()
    assert len(points) > 1000
    assert points[:, 2] == pytest.approx(np.zeros(len(points)), abs=1e-6)
    assert cloud.frame_id == "world"
    assert cloud.ts == 123.0

    module.config.sensor = replace(sensor, maximum_world_elevation=None)
    module.capture()
    unfiltered = publish.call_args.args[0].points().numpy()
    assert np.count_nonzero(unfiltered[:, 2] > 2.8) > 100
