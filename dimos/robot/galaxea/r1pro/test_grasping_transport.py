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

from dimos.robot.galaxea.r1pro.apartment_route import apartment_approach, refine_apartment_route
from dimos.robot.galaxea.r1pro.classical_sim import R1ProClassicalSim
from dimos.robot.galaxea.r1pro.grasping_blueprint import R1ProGraspingSim
from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport
from dimos.robot.galaxea.r1pro.object_packing_scene import sample_layout
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState
from dimos.robot.galaxea.r1pro.placement_regions import PlacementRegion


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


@pytest.mark.parametrize("pose", [[0, 0.4, 0.3], [0.31, 0.0, 0.2], [0.5, 0.0, 1.2]])
def test_rigid_collision_probe_matches_full_forward_dynamics(checker, pose):
    clear = checker.clear_pose_segment(np.array(pose), np.array(pose))
    full = mujoco.MjData(checker.model)
    full.qpos[:] = checker.probe.qpos
    mujoco.mj_forward(checker.model, full)

    np.testing.assert_array_equal(checker.probe.contact.geom, full.contact.geom)
    np.testing.assert_allclose(checker.probe.contact.dist, full.contact.dist, atol=1e-12)
    np.testing.assert_allclose(checker.probe.geom_xpos, full.geom_xpos, atol=1e-12)
    assert clear == (len(checker.collisions(full)) == 0)


def test_nearby_turn_is_a_local_body_adjustment(checker):
    goal = [-0.1, 0.1, -np.pi / 4]

    path = checker.plan_stance(goal)

    assert path == [[0.0, 0.0, 0.0], goal]
    assert all(checker.clear_pose_segment(np.array(a), np.array(b)) for a, b in pairwise(path))


def test_local_body_adjustment_does_not_replace_a_blocked_goal(checker):
    with pytest.raises(RuntimeError, match="No collision-free local body adjustment"):
        checker.plan_stance([0.5, 0.0, 0.4])


def test_local_body_adjustment_requires_room_navigation_for_distant_goals(checker):
    with pytest.raises(ValueError, match="within 70 cm"):
        checker.plan_stance([2.0, 0.0, 0.0])


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


def test_tracking_clearance_rejects_a_route_that_only_clears_the_nominal_robot(checker):
    wide = PlanarTransport(checker.model, checker.probe, cargo_bodies=(), collision_margin=0.06)
    # The robot's right face is 4 cm from the wall at this pose.
    pose = np.array([0.26, 0.0, 0.0])
    assert checker.clear_pose_segment(pose, pose)
    assert not wide.clear_pose_segment(pose, pose)
    assert wide.model.geom_margin.max() == pytest.approx(0.06)
    assert checker.model.geom_margin.max() == pytest.approx(0.02)


def test_approach_moves_a_tight_nominal_dock_to_preserve_tracking_clearance(checker):
    wide = PlanarTransport(checker.model, checker.probe, cargo_bodies=(), collision_margin=0.06)
    preposition = np.array([0.45, 0.0, np.pi])
    assert not wide.clear_pose_segment(np.array([0.73, 0, 0]), np.array([0.73, 0, np.pi]))

    transit, docking = apartment_approach(wide, preposition)

    assert not np.allclose(transit[:2], [0.73, 0])
    assert wide.clear_pose_segment(transit, docking)
    assert abs(docking[2] - transit[2]) <= np.pi


@pytest.fixture
def corridor_checker():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="base_link" pos="0 0 .3">
        <joint name="r1pro/base_x" type="slide" axis="1 0 0"/>
        <joint name="r1pro/base_y" type="slide" axis="0 1 0"/>
        <joint name="r1pro/base_yaw" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size=".05"/>
      </body>
      <body name="left_obstacle" pos=".35 .08 .3">
        <geom type="box" size=".05 .05 .1"/>
      </body>
      <body name="right_obstacle" pos=".75 -.08 .3">
        <geom type="box" size=".05 .05 .1"/>
      </body>
      <body name="task_bin" pos="0 0 .6">
        <geom type="box" size=".1 .1 .01" contype="0" conaffinity="0"/>
      </body>
    </worldbody><actuator>
      <position name="r1pro/base_x" joint="r1pro/base_x"/>
      <position name="r1pro/base_y" joint="r1pro/base_y"/>
      <position name="r1pro/base_yaw" joint="r1pro/base_yaw"/>
    </actuator></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return PlanarTransport(model, data, cargo_bodies=(), carry_tray=False, sweep_spacing=0.01)


def test_native_corridor_can_clear_obstacles_on_alternating_sides(corridor_checker):
    path = [[float(x), 0, 0] for x in np.linspace(0, 1.1, 23)]

    result = np.asarray(refine_apartment_route(corridor_checker, path))

    np.testing.assert_allclose(result[[0, -1]], [path[0], path[-1]])
    assert len(result) == len(path) + 1
    assert result[:, 1].min() < -0.04
    assert result[:, 1].max() > 0.04
    assert np.max(np.linalg.norm(result[1:, :2] - np.asarray(path)[:, :2], axis=1)) <= 0.1
    assert all(corridor_checker.clear_pose_segment(a, b) for a, b in pairwise(result))


def test_corridor_refinement_rejects_an_obstructed_endpoint(corridor_checker):
    with pytest.raises(RuntimeError, match="No clear full-body route"):
        refine_apartment_route(corridor_checker, [[0, 0, 0], [0.35, 0.08, 0]])


@pytest.fixture
def classical_navigation_scene(checker, mocker):
    sim = R1ProClassicalSim(navigation_clearance_m=0.06)
    try:
        scene = mocker.Mock(spec=PrimitiveSceneState, model=checker.model, data=checker.probe)
        scene.preposition_pose.return_value = np.array([1.0, 0.0, 0.0])
        scene.inventory.return_value = []
        scene.transport_planner.side_effect = lambda **kwargs: PlanarTransport(
            scene.model, scene.data, cargo_bodies=(), carry_tray=False, **kwargs
        )
        mocker.patch.object(sim, "_snapshot", return_value=scene)
        sim._regions = {
            "worktable": PlacementRegion("worktable", (1, 0, 0.7), (0.2, 0.2), ("support",))
        }
        sim._engine = mocker.Mock(_lock=threading.RLock())
        yield sim, scene
    finally:
        sim._engine = None
        sim.stop()


def test_classical_arrival_reserves_transit_clearance_and_settling_room(classical_navigation_scene):
    sim, scene = classical_navigation_scene

    result = sim.prepare_object_navigation("worktable")

    scene.transport_planner.assert_called_once_with(collision_margin=0.08)
    checker = PlanarTransport(
        scene.model, scene.data, cargo_bodies=(), carry_tray=False, collision_margin=0.08
    )
    assert checker.clear_pose_segment(*np.asarray(result["arrival"]))
    assert result["arrival"][0] == result["goal"]
