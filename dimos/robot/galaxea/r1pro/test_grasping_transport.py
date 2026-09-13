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

"""Path shortening retains actual robot/environment collision constraints."""

from itertools import pairwise
import threading

import mujoco
import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.grasping_blueprint import R1ProGraspingSim
from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport
from dimos.robot.galaxea.r1pro.object_packing_scene import sample_layout
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState


@pytest.fixture
def checker():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="base_link" pos="0 0 .3">
        <joint name="r1pro/base_x" type="slide" axis="1 0 0"/>
        <joint name="r1pro/base_y" type="slide" axis="0 1 0"/>
        <joint name="r1pro/base_yaw" type="hinge" axis="0 0 1"/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
      <body name="wall" pos=".5 0 .3"><geom type="box" size=".1 .15 .3"/></body>
      <body name="task_bin" pos="0 0 .6">
        <freejoint name="task_tray_free"/>
        <geom type="box" size=".1 .1 .01" contype="0" conaffinity="0"/>
      </body>
    </worldbody><actuator>
      <position name="r1pro/base_x" joint="r1pro/base_x"/>
      <position name="r1pro/base_y" joint="r1pro/base_y"/>
      <position name="r1pro/base_yaw" joint="r1pro/base_yaw"/>
    </actuator></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return PlanarTransport(model, data, cargo_bodies=())


def test_shortening_does_not_cut_through_a_wall(checker):
    path = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, 0.0, 0.0]]
    shortened = checker.shorten_path(path)
    assert shortened[0] == path[0]
    assert shortened[-1] == path[-1]
    assert len(shortened) > 2
    assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(shortened))


def test_redundant_native_points_can_be_shortened_when_sweep_is_clear(checker):
    assert checker.shorten_path([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.6, 0.0]]) == [
        [0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0],
    ]


def test_invalid_destination_is_not_repaired_into_a_partial_route(checker):
    with pytest.raises(RuntimeError, match="cannot clear"):
        checker.shorten_path([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])


def test_twist_base_can_plan_with_twenty_manipulation_joints(checker, mocker):
    sim = R1ProGraspingSim(dof=20)
    sim._engine = mocker.Mock(model=checker.model, data=checker.probe, _lock=threading.RLock())
    sim.cargo_bodies = ()
    try:
        path = sim.plan_transport(0.0, 0.6, 0.0)
        assert path[-1] == pytest.approx([0.0, 0.6, 0.0])
        assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(path))
    finally:
        sim._engine = None
        sim.stop()


def test_docking_uses_clear_elbow_around_furniture(checker):
    path = checker.plan((1.0, 0.5))
    assert path == [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 0.5, 0.0]]
    assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(path))


def test_grid_search_has_a_time_limit_when_simple_docking_paths_are_blocked(checker):
    with pytest.raises(RuntimeError, match="planning timed out"):
        checker.plan((1.0, 0.0), timeout=0.0)


@pytest.fixture
def cargo_scene():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="base_link" pos="0 0 .3">
        <joint name="r1pro/base_x" type="slide" axis="1 0 0"/>
        <joint name="r1pro/base_y" type="slide" axis="0 1 0"/>
        <joint name="r1pro/base_yaw" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size=".05"/>
      </body>
      <body name="held_right" pos="0 .4 .3"><freejoint/>
        <geom type="sphere" size=".04"/>
      </body>
      <body name="held_left" pos="0 -.4 .3"><freejoint/>
        <geom type="sphere" size=".04"/>
      </body>
      <body name="unheld_object" pos=".5 .4 .3"><freejoint/>
        <geom type="sphere" size=".04"/>
      </body>
      <body name="task_bin" pos=".5 -.4 .3"><freejoint name="task_tray_free"/>
        <geom type="box" size=".1 .1 .01"/>
      </body>
    </worldbody><actuator>
      <position name="r1pro/base_x" joint="r1pro/base_x"/>
      <position name="r1pro/base_y" joint="r1pro/base_y"/>
      <position name="r1pro/base_yaw" joint="r1pro/base_yaw"/>
    </actuator></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("held", [("held_right",), ("held_left",), ("held_right", "held_left")])
def test_carried_objects_route_around_stationary_objects_and_tray(cargo_scene, held):
    model, data = cargo_scene
    before = data.qpos.copy()
    checker = PlanarTransport(model, data, cargo_bodies=held, carry_tray=False, sweep_spacing=0.005)
    assert not checker.clear_segment(np.zeros(2), np.array([1.0, 0.0]))
    path = checker.plan((1.0, 0.0))
    assert path[-1] == [1.0, 0.0, 0.0]
    assert len(path) > 2
    assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(path))
    np.testing.assert_array_equal(checker.probe.body("task_bin").xpos, data.body("task_bin").xpos)
    np.testing.assert_array_equal(
        checker.probe.body("unheld_object").xpos, data.body("unheld_object").xpos
    )
    # Planning moves only a private copy, including the held objects.
    np.testing.assert_array_equal(data.qpos, before)


def test_existing_tray_transport_still_moves_the_tray_in_its_planning_copy(cargo_scene):
    model, data = cargo_scene
    checker = PlanarTransport(model, data, cargo_bodies=("held_right", "held_left"))
    assert checker.clear_segment(np.zeros(2), np.array([-0.5, 0.0]))
    np.testing.assert_allclose(
        checker.probe.body("task_bin").xpos, data.body("task_bin").xpos + np.array([-0.5, 0, 0])
    )


def test_primitive_can_move_forward_to_keep_a_far_target_in_its_workspace(checker, mocker):
    mocker.patch("dimos.robot.galaxea.r1pro.object_primitive_state.ObjectPackingState")
    scene = PrimitiveSceneState(checker.model, checker.probe, sample_layout(0), np.zeros(20))
    mocker.patch.object(scene, "transport_planner", return_value=checker)
    path = scene.preposition_path("right", np.array([0.6, 0.18, 0.77]))
    assert path[-1] == pytest.approx([0.2, 0.5, 0.0])
    assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(path))


def test_primitive_uses_a_nearby_base_pose_when_nominal_carrying_pose_is_blocked(checker, mocker):
    mocker.patch("dimos.robot.galaxea.r1pro.object_primitive_state.ObjectPackingState")
    scene = PrimitiveSceneState(checker.model, checker.probe, sample_layout(0), np.zeros(20))
    mocker.patch.object(scene, "transport_planner", return_value=checker)
    target = np.array([0.9, -0.08, 0.77])
    nominal = scene.preposition_pose("right", target)
    assert not checker.clear_pose_segment(nominal, nominal)
    path = scene.preposition_path("right", target)
    assert 0 < np.max(np.abs(np.asarray(path[-1]) - nominal)) <= 0.040001
    assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(path))


@pytest.mark.parametrize("arm, target_y", [("right", -0.32), ("left", 0.32)])
def test_primitive_can_stand_back_when_nearby_workspace_poses_hit_the_table(
    checker, mocker, arm, target_y
):
    mocker.patch("dimos.robot.galaxea.r1pro.object_primitive_state.ObjectPackingState")
    scene = PrimitiveSceneState(checker.model, checker.probe, sample_layout(0), np.zeros(20))
    mocker.patch.object(scene, "transport_planner", return_value=checker)
    target = np.array([0.75, target_y, 0.77])
    nominal = scene.preposition_pose(arm, target)
    assert all(
        not checker.clear_pose_segment(pose, pose)
        for pose in (
            nominal + np.array([x, y, 0]) for x in (-0.04, 0, 0.04) for y in (-0.04, 0, 0.04)
        )
    )

    path = scene.preposition_path(arm, target)

    assert 0.48 <= target[0] - path[-1][0] <= 0.520001
    assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(path))


def test_demonstration_can_prefer_longer_reach_without_disabling_collision_checks(checker, mocker):
    mocker.patch("dimos.robot.galaxea.r1pro.object_primitive_state.ObjectPackingState")
    scene = PrimitiveSceneState(checker.model, checker.probe, sample_layout(0), np.zeros(20))
    mocker.patch.object(scene, "transport_planner", return_value=checker)

    path = scene.preposition_path("right", np.array([0.6, 0.18, 0.77]), preferred_reach=0.48)

    assert path[-1] == pytest.approx([0.12, 0.5, 0.0])
    assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(path))
