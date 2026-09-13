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

"""Reachability searches preserve physical state and constrain the whole manipulation corridor."""

from dataclasses import replace
from threading import RLock

import mujoco
import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.apartment_sim import R1ProApartmentSim
from dimos.robot.galaxea.r1pro.everyday_objects import sample_everyday_layout
from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport
from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState
from dimos.robot.galaxea.r1pro.object_primitive_task import ObjectPrimitiveTask
from dimos.robot.galaxea.r1pro.object_reachability import ObjectReachability, stance_candidates
from dimos.robot.galaxea.r1pro.primitive_scene import prepare_primitive_scene


@pytest.fixture
def task(tmp_path):
    layout = sample_everyday_layout(380000)
    layout = ObjectLayout(
        layout.seed, tuple(replace(o, mass=0.015) if o.kind == "cup" else o for o in layout.objects)
    )
    scene, layout = prepare_primitive_scene(tmp_path / "scene.xml", layout, "right")
    with ObjectPrimitiveTask(scene, layout, arm="right", images=False) as instance:
        yield instance


def test_stance_search_preserves_requested_hand_after_world_rotation():
    target = np.array([2.0, 3.0, 0.8])
    poses = stance_candidates(target, np.array([1.0, 2.0, np.pi / 2]), "left")
    rotation = np.array([[0, -1], [1, 0]])
    expected = target[:2] - rotation @ [0.42, 0.32]
    assert any(np.allclose(p, [*expected, np.pi / 2]) for p in poses)


@pytest.mark.self_hosted
@pytest.mark.mujoco
def test_cup_support_accumulates_small_contacts(task):
    index = next(i for i, o in enumerate(task.layout.objects) if o.kind == "cup")
    bid = task.model.body(task.layout.objects[index].name).id
    forces = []
    force = np.zeros(6)
    for i, contact in enumerate(task.data.contact):
        if bid in task.model.geom_bodyid[contact.geom] and contact.dist <= 0:
            mujoco.mj_contactForce(task.model, task.data, i, force)
            forces.append(float(force[0]))
    assert sum(forces) > 0.02
    assert max(forces) < 0.02
    assert task.geometry(index)["support_geoms"]


@pytest.mark.self_hosted
@pytest.mark.mujoco
def test_local_policy_context_preserves_target_and_excludes_only_distant_neighbors(task):
    full = task.state.goal()
    local = task.state.goal(neighbor_distance=0.001)
    np.testing.assert_array_equal(local[:24], full[:24])
    np.testing.assert_array_equal(local[24:], np.zeros(28))
    np.testing.assert_array_equal(task.state.goal(neighbor_distance=10), full)


@pytest.mark.self_hosted
@pytest.mark.mujoco
def test_reachability_never_mutates_live_robot_or_object_state(task):
    before = [a.copy() for a in (task.data.qpos, task.data.qvel, task.data.ctrl)]
    scene = PrimitiveSceneState(task.model, task.data, task.layout, task.home)
    planner = ObjectReachability(scene)
    target = task.data.body(task.layout.objects[0].name).xpos.copy()
    candidates = planner.candidates("pick", 0, [target], arm="right", limit=1)
    assert candidates
    assert all(c.arm == "right" for c in candidates)
    for old, current in zip(before, (task.data.qpos, task.data.qvel, task.data.ctrl), strict=True):
        np.testing.assert_array_equal(old, current)


@pytest.mark.self_hosted
@pytest.mark.mujoco
def test_place_reachability_can_release_both_fingers_after_a_real_grasp(task):
    task.select(0)
    target = task.data.body(task.bottle_id).xpos.copy()
    task.teacher_preposition(target)
    for _, action in task.teacher_pick():
        task.primitive_step(action)
    assert task.state.holding()
    before = task.data.qpos.copy()
    planner = ObjectReachability(PrimitiveSceneState(task.model, task.data, task.layout, task.home))
    stance = planner.evaluate("place", "right", 0, target, planner.transport.start)
    assert stance.arm == "right"
    np.testing.assert_array_equal(task.data.qpos, before)


@pytest.fixture
def simulator(task, mocker, tmp_path):
    module = R1ProApartmentSim(output=tmp_path, scene_package=None)
    module._engine = mocker.Mock(model=task.model, data=task.data, _lock=RLock())
    module._scene_state = PrimitiveSceneState(task.model, task.data, task.layout, task.home)
    yield module
    module._engine = None
    module.stop()


@pytest.mark.self_hosted
@pytest.mark.mujoco
def test_rejected_reachability_route_preserves_action_selection(simulator, task, mocker):
    live = simulator._scene_state
    live.select_pick("right", 1)
    live.arms["right"].peak_lift = 0.12
    simulator._active = ("pick", "right")
    before = task.data.qpos.copy()
    mocker.patch.object(PlanarTransport, "plan", side_effect=RuntimeError("Blocked route"))
    with pytest.raises(RuntimeError, match="Blocked route"):
        simulator.prepare_reachable_primitive(
            "pick",
            "right",
            0,
            "",
            dict(
                arm="right",
                target=task.data.body(task.layout.objects[0].name).xpos.tolist(),
                base_pose=[0, 0, 0],
            ),
        )
    assert live.arms["right"].selected == 1
    assert live.arms["right"].peak_lift == 0.12
    assert simulator._active == ("pick", "right")
    np.testing.assert_array_equal(task.data.qpos, before)


@pytest.mark.self_hosted
@pytest.mark.mujoco
def test_transport_recovery_clears_error_only_without_disturbance(simulator, task):
    simulator._transport_initial = simulator._scene_state.inventory()
    simulator._error = "Navigation stopped"
    assert simulator.primitive_recovery()["mode"] == "transport_hold"
    assert simulator.finish_primitive_recovery()["mode"] == "transport_hold"
    assert simulator._error is None
    before = simulator._scene_state.inventory()
    task.data.joint(task.layout.objects[0].joint).qpos[0] += 0.05
    mujoco.mj_forward(task.model, task.data)
    simulator._transport_initial = before
    simulator._error = "Navigation stopped"
    with pytest.raises(RuntimeError, match="Disturbed"):
        simulator.finish_primitive_recovery()
    assert simulator._error == "Navigation stopped"
