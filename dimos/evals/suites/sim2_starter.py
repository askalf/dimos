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

"""Three authored application tests in the populated kitchen.

These use known setup coordinates, not RGB-D object detection. A failed
grasp/navigation attempt is a real application outcome; scene truth only
scores it. No action teleports an object or drives a simulator joint.
"""

import time
from typing import cast

from dimos.evals.sim2 import contained, fresh_states, inside_region, touching
from dimos.evals.types import InteractiveEval, Suite
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.manipulation_spec import ManipulationSpec
from dimos.memory.store.base import Store
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.navigation.navigation_spec import NavigationInterfaceSpec
from dimos.porcelain.dimos import Dimos
from dimos.sim2.interaction import reset_scene
from dimos.sim2.scene_types import SceneState, SceneUpdate


def navigation_case() -> InteractiveEval:
    initial: SceneState | None = None

    def cancel(app: Dimos) -> None:
        cast("NavigationInterfaceSpec", app.get_module("ReplanningAStarPlanner")).cancel_goal()

    def setup(app: Dimos) -> None:
        nonlocal initial
        initial = reset_scene(app, before_reset=(cancel,))
        initial.regions["walkway-goal"]

    def action(app: Dimos) -> None:
        assert initial is not None
        target = initial.regions["walkway-goal"].pose.position
        goal = PoseStamped(Pose(target.x, target.y, 0), frame_id="world")
        if not cast("NavigationInterfaceSpec", app.get_module("ReplanningAStarPlanner")).set_goal(
            goal
        ):
            raise RuntimeError("navigation rejected the goal")

    def score(store: Store) -> float:
        assert initial is not None
        return float(
            all(
                inside_region(state.robots["g1"].position.to_tuple(), state.regions["walkway-goal"])
                for state in fresh_states(store, initial)
            )
        )

    return InteractiveEval(
        id="sim2_g1_navigate",
        inputs="Walk to the marked kitchen walkway area.",
        blueprint="unitree-g1-groot-wbc",
        simulator="mujoco",
        scene="kitchen",
        setup=setup,
        action=action,
        score=score,
        timeout_s=60,
        tags=frozenset({"sim2", "navigation"}),
    )


def manipulation_case(*, place: bool) -> InteractiveEval:
    initial: SceneState | None = None
    object_pose = Pose(0.30, -0.16, 0.926)

    def cancel(app: Dimos) -> None:
        arm = cast("ManipulationModule", app.get_module("ManipulationModule"))
        arm.cancel()
        arm.clear_planned_path()

    def setup(app: Dimos) -> None:
        nonlocal initial
        initial = reset_scene(
            app, SceneUpdate(poses={"block": object_pose}), before_reset=(cancel,)
        )
        initial.entities["block"]
        initial.regions["tray/interior"]

    def action(app: Dimos) -> None:
        arm = cast("ManipulationSpec", app.get_module("ManipulationModule"))
        group = arm.list_planning_groups()[0].id

        def move(x: float, y: float, z: float) -> bool:
            target = PoseStamped(Pose((x, y, z), (1, 0, 0, 0)), frame_id="world")
            plan = arm.plan_to_poses({group: target}, speed_scale=0.3)
            return plan.succeeded and arm.execute(timeout=20).succeeded

        if not arm.set_gripper_position(1.0, group).succeeded:
            return
        x, y, z = object_pose.position.to_tuple()
        if not move(x, y, z + 0.16) or not move(x, y, z):
            return
        if not arm.set_gripper_position(0.0, group).succeeded:
            return
        time.sleep(0.6)  # Physical gripper travel, not a success assertion.
        if not move(x, y, z + 0.20):
            return
        if place:
            assert initial is not None
            center = initial.regions["tray/interior"].pose.position
            if not move(center.x, center.y, center.z + 0.20):
                return
            if not move(center.x, center.y, center.z + 0.04):
                return
            arm.set_gripper_position(1.0, group)
            time.sleep(0.6)
            move(center.x, center.y, center.z + 0.20)

    def score(store: Store) -> float:
        assert initial is not None
        states = fresh_states(store, initial)
        if place:
            return float(
                all(
                    contained(state.entities["block"], state.regions["tray/interior"])
                    and not touching(state, "block", "arm/")
                    and sum(v * v for v in state.entities["block"].velocity) < 0.0025
                    for state in states
                )
            )
        return float(
            all(
                state.entities["block"].pose.position.z > object_pose.position.z + 0.10
                and touching(state, "block", "arm/left_finger")
                and touching(state, "block", "arm/right_finger")
                for state in states
            )
        )

    return InteractiveEval(
        id="sim2_xarm_place" if place else "sim2_xarm_lift",
        inputs="Place the block in the tray." if place else "Lift and hold the block.",
        blueprint="xarm7-planner-coordinator",
        simulator="mujoco",
        scene="kitchen",
        setup=setup,
        action=action,
        score=score,
        timeout_s=8,
        tags=frozenset({"sim2", "manipulation", "place" if place else "lift"}),
    )


SUITE: Suite = (navigation_case(), manipulation_case(place=False), manipulation_case(place=True))
