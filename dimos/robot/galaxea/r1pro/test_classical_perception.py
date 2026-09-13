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

"""Virtual depth is current world-frame geometry, with instance separation."""

import mujoco
import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.classical_perception import segmented_object_cloud


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
        <geom type="plane" size="2 2 .1"/>
        <body name="target" pos=".3 -.4 .2"><freejoint/><geom type="sphere" size=".04"/></body>
        <body name="neighbor" pos=".4 -.4 .2"><geom type="box" size=".03 .03 .05"/></body>
        </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_depth_contains_only_selected_surface_and_preserves_live_state(scene):
    model, data = scene
    before = data.qpos.copy()
    points = segmented_object_cloud(model, data, model.body("target").id).points_f32()
    distances = np.linalg.norm(points - data.body("target").xpos, axis=1)
    np.testing.assert_allclose(distances, 0.04, atol=1e-6)
    np.testing.assert_array_equal(data.qpos, before)


def test_cloud_follows_changed_pose_in_world_coordinates(scene):
    model, data = scene
    data.qpos[:3] = [1.0, 0.6, 0.4]
    mujoco.mj_forward(model, data)
    cloud = segmented_object_cloud(model, data, model.body("target").id)
    assert cloud.frame_id == "world"
    distances = np.linalg.norm(cloud.points_f32() - [1.0, 0.6, 0.4], axis=1)
    np.testing.assert_allclose(distances, 0.04, atol=1e-6)
