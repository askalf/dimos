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

"""Independent arm-only primitive teacher, used for offline demonstrations only."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.grasping_task import HOME_TCP
from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport
from dimos.robot.galaxea.r1pro.home_kinematics import HomeKinematics
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_FPS as FPS
from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout
from dimos.robot.galaxea.r1pro.object_packing_state import ObjectPackingState
from dimos.robot.galaxea.r1pro.object_packing_task import ObjectPackingTask
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState
from dimos.robot.galaxea.r1pro.object_primitives import (
    MIRROR_ARM_SIGNS,
    Arm,
    Primitive,
    active_indices,
    apply_primitive_action,
    primitive_observation,
)
from dimos.robot.galaxea.r1pro.placement_regions import PlacementRegion


class ObjectPrimitiveTask(ObjectPackingTask):
    """ACT executor and measured state; the SDK teacher is an explicit separate API.

    Home posture is installed only on reset. During a primitive, neither the SDK
    teacher nor the action projection can command the torso or the other hand.
    """

    def __init__(self, scene: Path, layout: ObjectLayout, *, arm: Arm, images: bool = True) -> None:
        self.arm = arm
        self.policy_neighbor_distance: float | None = None
        self.active = list(active_indices(arm))
        super().__init__(scene, layout, images=False)
        self.home[2] = -0.2
        self.home[4:11] = self.home[11:18] * MIRROR_ARM_SIGNS[:7]
        self.data.qpos[self.qids] = self.home
        mujoco.mj_forward(self.model, self.data)
        self._kinematics = HomeKinematics(self.model, self.data, lock_lower_torso=True)
        self.home = self._kinematics.solve(
            self.data,
            {"right": np.array(HOME_TCP), "left": np.array(HOME_TCP) * [1, -1, 1]},
            allow_torso=False,
            position_tolerance=0.002,
        )
        self.reset(layout.seed)
        self._state = ObjectPackingState(self.model, self.data, layout, self.home, arm=arm)
        self.pad_ids = self.state.pad_ids
        self.tcp_id = self.model.site(f"{arm}_tcp").id
        self.renderer = mujoco.Renderer(self.model, 160, 160) if images else None
        self.probe = mujoco.MjData(self.model)
        self.probe.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.probe)

    def teacher_preposition(
        self, target: NDArray[Any], *, preferred_reach: float | None = None
    ) -> None:
        """Physically move the base before a primitive, keeping joint commands held.

        This deterministic bench teacher is not the house navigation adapter.
        Deployment will request the corresponding SDK whole-body plan instead.
        """
        initial = self.inventory()
        scene = PrimitiveSceneState(self.model, self.data, self.layout, self.home)
        path = scene.preposition_path(self.arm, target, preferred_reach=preferred_reach)
        command = self.data.ctrl[self.aids].copy()
        for pose in PlanarTransport.targets(path, FPS, speed=0.08):
            super().step(command, base_target=pose)
            self.state.observe()
            self.validate(initial)
        for _ in range(FPS):
            super().step(command)
            self.state.observe()
            self.validate(initial)
        mujoco.mj_forward(self.model, self.data)
        self._kinematics = HomeKinematics(self.model, self.data, lock_lower_torso=True)
        self.probe.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.probe)

    def select(self, index: int) -> None:
        if not 0 <= index < len(self.layout.objects):
            raise ValueError("Unknown object index")
        row = self.geometry(index)
        if not row["upright"] or not row["released"] or not row["support_geoms"]:
            raise RuntimeError("Pick must start with an upright supported released object")
        state = self.state
        state.selected = index
        state.bottle_id = self.model.body(self.layout.objects[index].name).id
        state.bottle_geoms = set(
            map(int, np.flatnonzero(self.model.geom_bodyid == state.bottle_id))
        )
        state.initial_height = float(self.data.body(state.bottle_id).xpos[2])
        state.peak_lift, state.bilateral_grasp = 0.0, False
        self.bottle_id, self.bottle_geoms = state.bottle_id, state.bottle_geoms
        self.initial_height = state.initial_height
        self.peak_lift, self.bilateral_grasp = 0.0, False

    def primitive_observation(
        self, primitive: Primitive, *, render_images: bool = True
    ) -> dict[str, NDArray[Any]]:
        values: dict[str, NDArray[Any]] = primitive_observation(
            primitive,
            self.arm,
            self.data.qpos[self.qids],
            self.state.goal(neighbor_distance=self.policy_neighbor_distance),
        )
        if self.renderer is not None and render_images:
            for key, camera in (("head", "head"), ("wrist", f"{self.arm}_wrist")):
                self.renderer.update_scene(self.data, camera=camera)
                values[f"observation.images.{key}"] = self.renderer.render().copy()
        return values

    def primitive_step(self, action: NDArray[Any]) -> None:
        super().step(apply_primitive_action(self.data.ctrl[self.aids], action, self.arm))
        self.state.observe()

    def _move(
        self, phase: str, target: NDArray[Any], opening: float, seconds: float
    ) -> Iterator[tuple[str, NDArray[np.float32]]]:
        assert self._kinematics is not None
        start = self.data.site(f"{self.arm}_tcp").xpos.copy()
        start_opening = float(self.data.ctrl[self.aids[self.active[-1]]])
        critical = phase in ("approach", "grasp", "place", "seek_support", "release")
        for frame in range(round(seconds * FPS)):
            t = (frame + 1) / (seconds * FPS)
            u = t * t * (3 - 2 * t)
            goal = self._kinematics.solve(
                self.probe,
                {self.arm: start + (target - start) * u},
                allow_torso=False,
                position_tolerance=0.002 if critical else 0.006,
                orientation_tolerance=0.005,
            )
            action = goal[self.active]
            action[-1] = start_opening + (opening - start_opening) * u
            previous = self.data.ctrl[self.aids][self.active]
            if np.max(np.abs(action[:-1] - previous[:-1])) > 0.14:
                raise RuntimeError("SDK teacher produced a discontinuous arm command")
            command = apply_primitive_action(self.data.ctrl[self.aids], action, self.arm)
            self.probe.qpos[self.qids] = command
            mujoco.mj_forward(self.model, self.probe)
            yield phase, action.astype(np.float32)
        for _ in range(FPS // 2):
            yield phase, action.astype(np.float32)

    def teacher_pick(
        self,
        *,
        from_approach: bool = False,
        clearance_z: float = 0.94,
        already_staged: bool = False,
    ) -> Iterator[tuple[str, NDArray[np.float32]]]:
        """Demonstrate a pick, optionally correcting a still-open ACT approach."""
        source = self.data.body(self.bottle_id).xpos.copy()
        obj = self.layout.objects[self.selected]
        sign = -1 if self.arm == "right" else 1
        grasp = source + np.array([0, 0, min(0.03, obj.half_size[2] * 0.45)])
        if from_approach:
            row = self.geometry(self.selected)
            offset = self.data.site(f"{self.arm}_tcp").xpos - grasp
            opening = self.data.qpos[self.qids[self.active[-1]]]
            if not (
                row["upright"]
                and row["released"]
                and row["support_geoms"]
                and row["settled"]
                and opening >= 0.04
                and np.linalg.norm(offset[:2]) <= 0.05
                and abs(offset[2]) <= 0.04
            ):
                raise RuntimeError("Approach correction requires an open hand near a stable source")
            self.probe.qpos[:] = self.data.qpos
            mujoco.mj_forward(self.model, self.probe)
        elif not already_staged:
            stage = self.data.body("base_link").xpos + self.data.body("base_link").xmat.reshape(
                3, 3
            ) @ np.array([0.42, sign * 0.28, 0])
            stage[2] = clearance_z
            yield from self._move(
                "stage",
                stage,
                0.05,
                2.0,
            )
            yield from self._move("above", np.r_[source[:2], clearance_z], 0.05, 2.0)
        yield from self._move("approach", grasp, 0.05, 2.0)
        yield from self._move("grasp", grasp, 0.0, 1.0)
        yield from self._move("lift", np.r_[source[:2], clearance_z], 0.0, 2.0)
        if not self.state.holding():
            raise RuntimeError("Pick did not establish a current stable 10 cm lift")
        # Train an explicit held endpoint; no place action follows this boundary.
        for _ in range(FPS):
            yield "hold", self.data.ctrl[self.aids][self.active].astype(np.float32)
        if not self.state.holding():
            raise RuntimeError("Object slipped during the held endpoint")

    def teacher_place(
        self, target: NDArray[Any], region: PlacementRegion, *, clearance_z: float = 0.94
    ) -> Iterator[tuple[str, NDArray[np.float32]]]:
        if not self.state.carrying():
            raise RuntimeError("Place must start with a measured upright two-pad grasp")
        self.state.target = target.copy()
        self.probe.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.probe)
        offset = self.data.site(f"{self.arm}_tcp").xpos - self.data.body(self.bottle_id).xpos
        destination = target + offset
        yield from self._move("transfer", np.r_[destination[:2], clearance_z], 0.0, 2.0)
        yield from self._move("place", destination, 0.0, 2.0)
        for _ in range(8):
            if set(self.geometry(self.selected)["support_geoms"]) & set(region.support_geoms):
                break
            destination = destination - [0, 0, 0.002]
            yield from self._move("seek_support", destination, 0.0, 0.3)
        if not set(self.geometry(self.selected)["support_geoms"]) & set(region.support_geoms):
            raise RuntimeError("No requested physical support before release")
        yield from self._move("release", destination, 0.05, 1.0)
        yield from self._move("retreat", np.r_[destination[:2], clearance_z], 0.05, 2.0)
        # A primitive can end at a clear retreat without an unnecessary home tour.
        row = self.geometry(self.selected)
        obj = self.layout.objects[self.selected]
        if not (
            row["released"]
            and row["upright"]
            and row["settled"]
            and set(row["support_geoms"]) & set(region.support_geoms)
            and region.contains(tuple(row["position"]), obj.radius, obj.half_size[2])
        ):
            raise RuntimeError(
                "Place did not finish with a supported released object in the region"
            )
