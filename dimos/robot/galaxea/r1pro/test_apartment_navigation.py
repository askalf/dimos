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

"""Apartment navigation preserves frames, rejects partial paths and waits for map readiness."""

import numpy as np
import pytest

from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.galaxea.r1pro.apartment_navigation import ApartmentNavigation
from dimos.robot.galaxea.r1pro.navigation_sim import pose_message
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState


@pytest.fixture
def navigation(mocker, tmp_path):
    module = ApartmentNavigation()
    mocker.patch.object(module.goal, "publish")
    mocker.patch.object(module.global_map, "publish")
    cloud = tmp_path / "map.npy"
    np.save(cloud, np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    yield module, cloud
    module.stop()


def test_goal_waits_for_native_map_acknowledgement(navigation):
    module, cloud = navigation
    module.request_object_route([1, 0, 0], [0.2, 0], str(cloud))
    module.goal.publish.assert_not_called()
    module._surface(PointCloud2.from_numpy(np.array([[0.0, 0.0, 0.0]])))
    module.goal.publish.assert_called_once()
    goal = module.goal.publish.call_args.args[0]
    assert goal.x == pytest.approx(1.2)
    assert goal.y == 0


def test_stale_and_partial_routes_cannot_reach_execution(navigation):
    module, cloud = navigation
    module.request_object_route([1, 0, 0], [0.2, 0], str(cloud))
    module._route(
        Path(poses=[pose_message([0.2, 0, 0]), pose_message([1.2, 0, 0])], frame_id="world", ts=0)
    )
    assert not module.object_route_status()["ready"]
    module._route(
        Path(poses=[pose_message([0.2, 0, 0]), pose_message([0.8, 0, 0])], frame_id="world")
    )
    assert module.object_route_status() == dict(
        ready=False, error="KronkNav returned an incomplete route"
    )


def test_native_footprint_path_is_converted_back_to_base_positions(navigation):
    module, cloud = navigation
    module.request_object_route([1, 2, np.pi / 2], [0.2, 0], str(cloud))
    module._route(
        Path(poses=[pose_message([0, 0.2, 0]), pose_message([1, 2.2, 0])], frame_id="world")
    )
    result = module.object_route_status()
    assert result["ready"]
    np.testing.assert_allclose(result["path"], [[0, 0, np.pi / 2], [1, 2, np.pi / 2]], atol=1e-8)


@pytest.mark.parametrize("arm,side", [("right", -0.32), ("left", 0.32)])
def test_docking_preserves_arm_workspace_after_turning(arm, side):
    target = np.array([2.0, 3.0, 0.8])
    pose = PrimitiveSceneState.preposition_pose(arm, target, yaw=np.pi / 2)
    rotation = np.array([[0, -1], [1, 0]])
    np.testing.assert_allclose(rotation.T @ (target[:2] - pose[:2]), [0.4, side])
    assert pose[2] == pytest.approx(np.pi / 2)


def test_route_in_another_frame_is_refused(navigation):
    module, cloud = navigation
    module.request_object_route([1, 0, 0], [0, 0], str(cloud))
    module._route(Path(poses=[pose_message([0, 0, 0]), pose_message([1, 0, 0])], frame_id="map"))
    assert module.object_route_status() == dict(
        ready=False, error="KronkNav route must use the world frame"
    )
