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

"""Interactive ACT manipulation and explicit object carrying between apartment supports."""

import json
import time
from typing import Any, cast

import numpy as np

from dimos.agents.annotation import skill
from dimos.manipulation.manipulation_spec import ExecutionStatus
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.galaxea.r1pro.apartment_navigation import (
    APARTMENT_NAV_TASK,
    ApartmentNavigationSpec,
    ApartmentSimSpec,
)
from dimos.robot.galaxea.r1pro.home_surfaces import station_name
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.navigation_sim import pose_message
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, Arm, Primitive
from dimos.robot.galaxea.r1pro.primitive_skills import (
    R1ProPrimitiveSkills,
    resolve_primitive_object,
)


class R1ProApartmentSkills(R1ProPrimitiveSkills):
    _apartment: ApartmentSimSpec
    _navigation: ApartmentNavigationSpec

    def _stop_control(self) -> None:
        try:
            self._control.task_invoke(APARTMENT_NAV_TASK, "cancel", {})
        finally:
            super()._stop_control()

    def _follow(self, path: list[list[float]], report: dict[str, Any]) -> None:
        if len(path) < 2:
            return
        path = self._apartment.validate_object_navigation(path)
        before = self._sim.primitive_state()
        self._control.task_invoke(APARTMENT_NAV_TASK, "reset", {})
        accepted = self._control.task_invoke(
            APARTMENT_NAV_TASK,
            "start_path",
            {
                "path": Path(poses=[pose_message(p) for p in path], frame_id="world"),
                "current_odom": pose_message(before["base_pose"]),
            },
        )
        if not accepted:
            raise RuntimeError("Holonomic task rejected the apartment path")
        report["motion_started"] = True
        length = float(np.linalg.norm(np.diff(np.asarray(path)[:, :2], axis=0), axis=1).sum())
        deadline = time.monotonic() + 90 + length / 0.05
        last_pose = np.asarray(before["base_pose"])
        progressed = time.monotonic()
        last_sim_time = before["sim_time"]
        updated = time.monotonic()
        while True:
            self._pause(0.05)
            state = self._sim.primitive_state()
            if state["error"]:
                raise RuntimeError(state["error"])
            if state["sim_time"] > last_sim_time:
                updated, last_sim_time = time.monotonic(), state["sim_time"]
            if time.monotonic() - updated > 2:
                raise RuntimeError("Apartment simulation stopped updating during navigation")
            task = self._control.task_invoke(APARTMENT_NAV_TASK, "get_state", {})
            if task == "arrived":
                break
            if task in ("aborted", "idle"):
                raise RuntimeError(f"Navigation task stopped: {task}")
            pose = np.asarray(state["base_pose"])
            if np.linalg.norm(pose - last_pose) > 0.002:
                last_pose, progressed = pose, time.monotonic()
            if time.monotonic() - progressed > 12 or time.monotonic() > deadline:
                raise RuntimeError("Navigation made no progress or exceeded its deadline")
        self._sim.stop_primitive_base()
        report.setdefault("paths", []).append(path)
        self._pause(0.5)
        final = np.asarray(self._sim.primitive_state()["base_pose"])
        error = final - path[-1]
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        if np.linalg.norm(error[:2]) > 0.025 or abs(error[2]) > 0.03:
            raise RuntimeError("Measured navigation endpoint is outside the docking tolerance")

    def _navigate(
        self, destination: str, arm: str, report: dict[str, Any], stance: list[float] | None = None
    ) -> None:
        self._stop_control()
        plan = self._apartment.prepare_object_navigation(destination, arm, stance)
        report["navigation"] = plan
        try:
            self._follow(plan["departure"], report)
            self._navigation.request_object_route(
                plan["goal"], plan["footprint_offset"], plan["cloud"]
            )
            deadline = time.monotonic() + 90
            while True:
                self._pause(0.1)
                route = self._navigation.object_route_status()
                if route.get("error"):
                    raise RuntimeError(route["error"])
                if route["ready"]:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("KronkNav did not produce an apartment route")
            report["native_route"] = route["path"]
            self._follow(route["path"], report)
            self._follow(plan["arrival"], report)
            self._apartment.finish_object_navigation()
        finally:
            self._control.task_invoke(APARTMENT_NAV_TASK, "cancel", {})
            self._sim.stop_primitive_base()

    def _execute(
        self, primitive: Primitive, arm: Arm, index: int, region: str, report: dict[str, Any]
    ) -> None:
        state = self._sim.primitive_state()
        if primitive == "place":
            region = station_name(region)
            held = state["held_objects"][arm]
            if held is None:
                raise RuntimeError(f"The {arm} hand holds no object")
            index = next(row["index"] for row in state["objects"] if row["id"] == held)
        assessment = report.get("reachability")
        if assessment is None:
            assessment = self._apartment.assess_object_reachability(primitive, index, arm, region)
            report["reachability"] = assessment
        if not assessment["candidates"]:
            raise RuntimeError(
                "No feasible stance found for the requested hand and supported target in the bounded search"
            )
        stance = assessment["candidates"][0]
        if stance["arm"] != arm:
            raise RuntimeError("Reachability cannot substitute the requested hand")
        target = np.asarray(stance["base_pose"])
        current = np.asarray(state["base_pose"])
        angle = abs(np.arctan2(np.sin(target[2] - current[2]), np.cos(target[2] - current[2])))
        if np.linalg.norm(target[:2] - current[:2]) > 0.5 or angle > 0.02:
            destination = state["objects"][index]["id"] if primitive == "pick" else region
            self._navigate(destination, arm, report, stance["base_pose"])
        super()._execute(primitive, arm, index, region, report)

    def _prepare_primitive(
        self, primitive: Primitive, arm: Arm, index: int, region: str, report: dict[str, Any]
    ) -> dict[str, Any]:
        return self._apartment.prepare_reachable_primitive(
            primitive, arm, index, region, report["reachability"]["candidates"][0]
        )

    def _prepare_posture(self, selection: dict[str, Any], report: dict[str, Any]) -> None:
        stance = selection["reachability"]
        if not stance["torso_changed"]:
            return
        positions = stance["ready_joints"]
        groups = {"torso": (0, 4), "left_arm": (4, 11), "right_arm": (11, 18)}
        plan = self._manipulation.plan_to_joints(
            {
                name: JointState(
                    name=list(R1PRO_PICK_PLACE_JOINTS[first:last]), position=positions[first:last]
                )
                for name, (first, last) in groups.items()
            },
            speed_scale=0.3,
        )
        if not plan.succeeded or plan.plan is None:
            raise RuntimeError(f"DimOS whole-body posture planning failed: {plan.message}")
        index = int(selection["object"].removeprefix("object_")) - 1
        self._apartment.validate_apartment_posture(plan.plan.trajectory, index, stance["arm"])
        report["posture_plan_id"] = plan.plan.plan_id
        report["motion_started"] = True
        execution = self._manipulation.execute(blocking=False, plan_id=plan.plan.plan_id)
        if execution.status is not ExecutionStatus.ACCEPTED:
            raise RuntimeError(f"DimOS rejected the posture trajectory: {execution}")
        deadline = time.monotonic() + 90
        while True:
            self._pause(0.05)
            state = self._sim.primitive_state()
            if state["error"]:
                raise RuntimeError(state["error"])
            execution = self._manipulation.wait_for_execution(timeout=0.01)
            if execution.succeeded:
                break
            if (
                execution.status
                not in (
                    ExecutionStatus.ACCEPTED,
                    ExecutionStatus.EXECUTING,
                    ExecutionStatus.TIMED_OUT,
                )
                or time.monotonic() >= deadline
            ):
                raise RuntimeError(f"DimOS posture execution failed: {execution}")
        actual = self._sim.primitive_state()["joint_positions"]
        if (
            max(
                abs(actual[name] - positions[i])
                for i, name in enumerate(R1PRO_PICK_PLACE_JOINTS[:18])
            )
            > 0.02
        ):
            raise RuntimeError("Measured posture did not reach the assessed ACT starting pose")

    @skill
    def pick_object(self, object: str = "nearest", arm: str = "auto") -> str:
        """ACT grasp, lift and HOLD a selected item, positioning the robot first if needed.

        Args:
            object: Object ID, unique kind/shape, nearest/furthest or leftmost/rightmost.
            arm: auto compares free hands using reachability; left/right preserves your choice.
        """
        if arm not in (*ARMS, "auto"):
            return json.dumps(dict(accepted=False, reason="Use auto, left or right"))
        state = self._sim.primitive_state()
        if arm in ARMS and state["held_objects"][arm]:
            return json.dumps(dict(accepted=False, reason=f"{arm} hand is occupied"))
        try:
            index = resolve_primitive_object(state["objects"], object)
        except ValueError as exc:
            return json.dumps(dict(accepted=False, reason=str(exc)))

        def operation(report: dict[str, Any]) -> None:
            assessment = self._apartment.assess_object_reachability("pick", index, arm)
            report["reachability"] = assessment
            if not assessment["candidates"]:
                raise RuntimeError(
                    "No feasible stance found for the selected item and requested hand"
                )
            selected_arm = cast("Arm", assessment["candidates"][0]["arm"])
            if arm != "auto" and selected_arm != arm:
                raise RuntimeError("Reachability cannot substitute the requested hand")
            self._execute("pick", selected_arm, index, "", report)

        return self._start(f"pick/{arm}/{index}", operation)

    @skill
    def place_object(self, region: str, arm: str = "auto") -> str:
        """ACT place a held item in an empty reachable part of the requested surface.

        Args:
            region: A support region from get_surfaces, or tray/table.
            arm: left/right chooses a held item. auto requires exactly one occupied hand.
        """
        if arm == "auto":
            held = self._sim.primitive_state()["held_objects"]
            hands = [side for side in ARMS if held[side] is not None]
            if len(hands) != 1:
                return json.dumps(
                    dict(
                        accepted=False,
                        reason="Choose the occupied left or right hand; auto requires exactly one held item",
                    )
                )
            arm = hands[0]
        return super().place_object(station_name(region), arm)

    @skill
    def go_to(self, destination: str) -> str:
        """Navigate to a measured apartment support while retaining every held object.

        Args:
            destination: dining_table, kitchen, or worktable. Arrival does not place or release.
        """

        def operation(report: dict[str, Any]) -> None:
            held = self._sim.primitive_state()["held_objects"]
            arm = "left" if held["left"] and not held["right"] else "right"
            self._navigate(destination, arm, report)

        return self._start(f"navigate/{destination}", operation)

    @skill
    def get_surfaces(self) -> str:
        """List measured apartment placement regions; free-space checks happen before placement."""
        return json.dumps(self._sim.primitive_state()["defined_regions"])
