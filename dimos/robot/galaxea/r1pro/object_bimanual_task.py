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

"""Demonstration task that retains per-arm grasp evidence across sequential primitives."""

from pathlib import Path
from typing import Any

import mujoco

from dimos.robot.galaxea.r1pro.home_kinematics import HomeKinematics
from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState
from dimos.robot.galaxea.r1pro.object_primitive_task import ObjectPrimitiveTask
from dimos.robot.galaxea.r1pro.object_primitives import Arm, active_indices


class BimanualPrimitiveTask(ObjectPrimitiveTask):
    def __init__(self, scene: Path, layout: ObjectLayout, *, arm: Arm, images: bool = True) -> None:
        super().__init__(scene, layout, arm=arm, images=images)
        self.scene_state = PrimitiveSceneState(self.model, self.data, layout, self.home)
        self.switch_arm(arm)

    def switch_arm(self, arm: Arm) -> None:
        """Change the next primitive's owner without moving or resetting either arm."""
        self.arm = arm
        self.active = list(active_indices(arm))
        self._state = self.scene_state.arms[arm]
        self.pad_ids = self.state.pad_ids
        self.tcp_id = self.model.site(f"{arm}_tcp").id
        self.bottle_id, self.bottle_geoms = self.state.bottle_id, self.state.bottle_geoms
        self.initial_height = self.state.initial_height
        # mj_step leaves transforms at the preceding integration point. Match
        # them to current joints before the SDK's strict FK consistency check.
        mujoco.mj_forward(self.model, self.data)
        self._kinematics = HomeKinematics(self.model, self.data, lock_lower_torso=True)
        self.probe.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.probe)

    def select(self, index: int) -> None:
        self._state = self.scene_state.select_pick(self.arm, index)
        self.bottle_id, self.bottle_geoms = self.state.bottle_id, self.state.bottle_geoms
        self.initial_height = self.state.initial_height
        self.peak_lift, self.bilateral_grasp = 0.0, False

    def inventory(self) -> list[dict[str, Any]]:
        return self.scene_state.inventory()

    def validate(self, initial: list[dict[str, Any]]) -> None:
        self.scene_state.observe()
        self.scene_state.validate(initial, arm=self.arm, selected=self.selected)
