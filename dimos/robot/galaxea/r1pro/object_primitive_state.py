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

"""Measured per-arm object ownership for composable pick and place actions."""

import time
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.grasping_transport import PlanarTransport
from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout
from dimos.robot.galaxea.r1pro.object_packing_state import ObjectPackingState
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, Arm


class PrimitiveSceneState:
    """Two contact observers share physics, without inventing held-object ownership."""

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, layout: ObjectLayout, home: NDArray[Any]
    ) -> None:
        self.model, self.data, self.layout = model, data, layout
        self.arms = {arm: ObjectPackingState(model, data, layout, home, arm=arm) for arm in ARMS}

    def inventory(self) -> list[dict[str, Any]]:
        rows = []
        for i, obj in enumerate(self.layout.objects):
            evidence = {arm: state.geometry(i) for arm, state in self.arms.items()}
            row = evidence["right"].copy()
            contacts = [arm for arm in ARMS if not evidence[arm]["released"]]
            holders = [
                arm
                for arm in ARMS
                if evidence[arm]["grasped"]
                and evidence[arm]["upright"]
                and not evidence[arm]["support_geoms"]
            ]
            if len(holders) > 1:
                raise RuntimeError(f"Both grippers claim {obj.name}; handoff is not supported")
            held_by = holders[0] if holders else None
            row.update(
                released=not contacts,
                supported=bool(row["support_geoms"]),
                contacting_arms=contacts,
                held_by=held_by,
                tcp_offset=(
                    self.data.site(f"{held_by}_tcp").xpos - self.data.body(obj.name).xpos
                ).tolist()
                if held_by
                else None,
            )
            rows.append(row)
        for arm in ARMS:
            if sum(row["held_by"] == arm for row in rows) > 1:
                raise RuntimeError(f"Multiple objects held by {arm}; resolve ambiguous contact")
        return rows

    def held_objects(self) -> dict[Arm, int | None]:
        rows = self.inventory()
        return {
            arm: next((i for i, row in enumerate(rows) if row["held_by"] == arm), None)
            for arm in ARMS
        }

    def transport_planner(self) -> PlanarTransport:
        """Include both hands' measured cargo; leave the tray and unheld objects stationary."""
        return PlanarTransport(
            self.model,
            self.data,
            cargo_bodies=tuple(row["object"] for row in self.inventory() if row["held_by"]),
            carry_tray=False,
            sweep_spacing=0.005,
        )

    @staticmethod
    def preposition_pose(arm: Arm, target: NDArray[Any]) -> NDArray[np.float64]:
        """Put the target in the arm's demonstrated workspace, including forward base motion."""
        return np.array(
            [
                float(target[0]) - 0.4,
                float(target[1]) + (0.32 if arm == "right" else -0.32),
                0.0,
            ]
        )

    @classmethod
    def preposition_poses(cls, arm: Arm, target: NDArray[Any]) -> list[NDArray[np.float64]]:
        """Prefer the nominal workspace, then clear poses within demonstrated reach."""
        nominal = cls.preposition_pose(arm, target)
        return [
            nominal + np.array([x, y, 0])
            for x, y in (
                (0, 0),
                (-0.04, 0),
                (0.04, 0),
                (0, -0.04),
                (0, 0.04),
                (-0.04, -0.04),
                (-0.04, 0.04),
                (0.04, -0.04),
                (0.04, 0.04),
                # The 40 cm nominal reach can put the parked hands into the
                # bench. Retain the demonstrated 48--52 cm forward reach as
                # alternatives, without forcing far-table goals back to x=0.
                (-0.08, 0),
                (-0.12, 0),
                (-0.08, -0.04),
                (-0.08, 0.04),
                (-0.12, -0.04),
                (-0.12, 0.04),
            )
        ]

    def preposition_path(self, arm: Arm, target: NDArray[Any]) -> list[list[float]]:
        """Route the held posture into the learned workspace without sweeping cargo into clutter."""
        planner = self.transport_planner()
        deadline = time.monotonic() + 10
        for desired in self.preposition_poses(arm, target):
            if time.monotonic() >= deadline:
                raise RuntimeError("Prepositioning planning exceeded ten seconds")
            if not planner.clear_pose_segment(desired, desired):
                continue
            try:
                path = planner.plan(
                    tuple(desired[:2]),
                    resolution=0.025,
                    max_distance=1.5,
                    timeout=max(0, deadline - time.monotonic()),
                )
                return planner.shorten_path([*path, desired.tolist()])
            except RuntimeError:
                continue
        raise RuntimeError("No collision-free base route into the requested arm's workspace")

    def select_pick(self, arm: Arm, index: int) -> ObjectPackingState:
        rows = self.inventory()
        if arm not in self.arms or not 0 <= index < len(rows):
            raise ValueError("Unknown arm or object")
        if any(arm in row["contacting_arms"] for row in rows):
            raise RuntimeError(f"The {arm} hand already contacts an object")
        row = rows[index]
        if (
            not row["released"]
            or not row["upright"]
            or not row["support_geoms"]
            or not row["settled"]
        ):
            raise RuntimeError("Pick source must be upright, settled, supported and unheld")
        state = self.arms[arm]
        state.selected = index
        state.bottle_id = self.model.body(self.layout.objects[index].name).id
        state.bottle_geoms = set(
            map(int, np.flatnonzero(self.model.geom_bodyid == state.bottle_id))
        )
        state.initial_height = float(self.data.body(state.bottle_id).xpos[2])
        state.peak_lift, state.bilateral_grasp = 0.0, False
        state.target = np.zeros(3)
        return state

    def select_place(self, arm: Arm, target: NDArray[Any]) -> ObjectPackingState:
        held = self.held_objects()
        if arm not in held or (index := held[arm]) is None:
            raise RuntimeError(f"No confirmed object held by {arm}")
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Place requires a finite supported object goal")
        state = self.arms[arm]
        if index != state.selected:
            raise RuntimeError("Measured hold differs from the selected object; recover explicitly")
        state.target = target.copy()
        return state

    def observe(self) -> None:
        for state in self.arms.values():
            state.observe()

    def validate(self, initial: list[dict[str, Any]], *, arm: Arm, selected: int) -> None:
        """Protect unrequested objects, including another hand's moving cargo.

        Held cargo may move with its TCP during prepositioning but cannot slip,
        tip, be released, or change hands. All other objects stay at their support.
        """
        for i, (before, after) in enumerate(zip(initial, self.inventory(), strict=True)):
            if i == selected:
                continue
            if before["held_by"] is not None:
                if (
                    after["held_by"] != before["held_by"]
                    or not after["upright"]
                    or np.linalg.norm(np.asarray(after["tcp_offset"]) - before["tcp_offset"])
                    > 0.015
                ):
                    raise RuntimeError(f"Lost or disturbed the other hand's {after['object']}")
            elif (
                not after["upright"]
                or not after["released"]
                or not after["support_geoms"]
                or (before["inside"] and not after["inside"])
                or np.linalg.norm(np.asarray(after["position"]) - before["position"]) > 0.015
            ):
                raise RuntimeError(f"Disturbed unrequested {after['object']}")
        collisions = self.arms[arm].guard.collisions(self.data, ignore_cargo=True)
        if collisions:
            raise RuntimeError(f"Robot collided with environment: {collisions}")
