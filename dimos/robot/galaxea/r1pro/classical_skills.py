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

"""Explicit classical pick/hold/place state machine for the apartment demo."""

from collections.abc import Callable
from concurrent.futures import CancelledError
import json
from pathlib import Path as FilePath
import threading
import time
from typing import Any

import numpy as np

from dimos.agents.annotation import skill
from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.manipulation.grasping.grasp_gen_spec import GraspGenSpec
from dimos.manipulation.manipulation_spec import ExecutionStatus, ManipulationSpec
from dimos.manipulation.planning.trajectory_generator.joint_trajectory_generator import (
    JointTrajectoryGenerator,
)
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.msgs.trajectory_msgs.TrajectoryStatus import TrajectoryState
from dimos.robot.galaxea.r1pro.apartment_navigation import (
    APARTMENT_NAV_TASK,
    ApartmentNavigationSpec,
)
from dimos.robot.galaxea.r1pro.classical_selection import color_name, resolve_classical_object
from dimos.robot.galaxea.r1pro.classical_sim import ClassicalSimSpec
from dimos.robot.galaxea.r1pro.config import R1PRO_PLANAR_BASE
from dimos.robot.galaxea.r1pro.home_spec import HomeControlSpec
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.navigation_sim import pose_message
from dimos.robot.galaxea.r1pro.object_primitives import ARMS
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class R1ProClassicalSkills(Module):
    _sim: ClassicalSimSpec
    _control: HomeControlSpec
    _manipulation: ManipulationSpec
    _grasp_generator: GraspGenSpec
    _navigation: ApartmentNavigationSpec

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._done.set()
        self._thread: threading.Thread | None = None
        self._closing = False
        self._counter = 0
        self._action: dict[str, Any] = dict(state="idle", recovery_required=False)

    def _status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._action)

    def _pause(self, seconds: float) -> None:
        if self._cancel.wait(seconds):
            raise CancelledError("Cancelled; hands hold their current positions")

    def _start(
        self, name: str, operation: Callable[[dict[str, Any]], None], *, recovery: bool = False
    ) -> str:
        with self._lock:
            if self._closing or not self._done.is_set():
                return json.dumps(dict(accepted=False, reason="busy_or_stopping"))
            if self._action.get("recovery_required") and not recovery:
                return json.dumps(
                    dict(accepted=False, reason="recovery_required", action=self._action)
                )
            prior_recovery = bool(self._action.get("recovery_required"))
            self._counter += 1
            self._action = dict(
                id=self._counter,
                name=name,
                state="running",
                success=False,
                recovery_required=prior_recovery,
            )
            self._cancel.clear()
            self._done.clear()
            self._thread = threading.Thread(
                target=self._run, args=(operation,), daemon=True, name="r1pro-classical-action"
            )
            self._thread.start()
            return json.dumps(dict(accepted=True, **self._action))

    def _run(self, operation: Callable[[dict[str, Any]], None]) -> None:
        report: dict[str, Any] = {}
        prior_recovery = bool(self._status().get("recovery_required"))
        terminal: dict[str, Any] = dict(state="failed", success=False, recovery_required=False)
        try:
            operation(report)
            terminal.update(state="completed", success=True, recovery_required=False)
        except CancelledError as exc:
            terminal.update(
                state="cancelled",
                error=str(exc),
                recovery_required=prior_recovery or bool(report.get("motion_started")),
            )
        except Exception as exc:
            logger.exception("Classical action failed")
            terminal.update(
                error=str(exc),
                recovery_required=prior_recovery or bool(report.get("motion_started")),
            )
        finally:
            try:
                self._stop_control()
            except Exception as exc:
                terminal.update(
                    state="failed", success=False, recovery_required=True, cleanup_error=str(exc)
                )
            try:
                report["final"] = self._sim.primitive_state()
                report["snapshot"] = self._sim.save_classical_state()
                output = (
                    FilePath(self._sim.prepare_primitive_session()["output"])
                    / f"action-{self._counter:03d}.json"
                )
                output.write_text(
                    json.dumps({**self._status(), **report, **terminal}, indent=2) + "\n"
                )
                terminal["evidence"] = str(output)
            except Exception as exc:
                terminal["report_error"] = str(exc)
            with self._lock:
                self._action.update(terminal)
                self._done.set()

    @skill
    def wait_for_action(self, seconds: float = 10.0) -> str:
        """Wait up to 20 seconds for the current action; repeat while state is running."""
        self._done.wait(min(max(seconds, 0.0), 20.0))
        return json.dumps(self._status())

    @skill
    def stop_action(self) -> str:
        """Cancel current motion and preserve each hand's position; never release or reset."""
        self._cancel.set()
        return json.dumps(self._status())

    @skill
    def recover_action(self) -> str:
        """Preserve a confirmed hold, or open supported contacts and restore only the failed arm.

        Recovery never retries an grasp and never resets scene progress or the other hand.
        """

        def operation(report: dict[str, Any]) -> None:
            self._stop_control()
            recovery = self._sim.primitive_recovery()
            report["recovery"] = recovery
            if recovery["mode"] == "empty_hand":
                arm = recovery["arm"]
                joints = recovery["joints"]
                measured = self._control.get_joint_positions()
                start = [measured[n] for n in joints]
                opened = [*start[:-1], 0.05]
                trajectory = JointTrajectory(
                    joint_names=joints,
                    points=[
                        TrajectoryPoint(positions=start, time_from_start=0.0),
                        TrajectoryPoint(positions=opened, time_from_start=0.8),
                        TrajectoryPoint(positions=opened, time_from_start=1.2),
                    ],
                )
                report["motion_started"] = True
                opening_result = self._control.execute_trajectory(trajectory, f"primitive_{arm}")
                if opening_result.status is not TrajectoryExecutionStatus.ACCEPTED:
                    raise RuntimeError(f"Recovery gripper opening was rejected: {opening_result}")
                self._pause(1.3)
                stopped = self._control.cancel_trajectory(f"primitive_{arm}")
                if not stopped.safe:
                    raise RuntimeError("Cannot confirm recovery gripper stopped")
                plan = self._manipulation.plan_to_joints(
                    {f"{arm}_arm": JointState(name=joints[:-1], position=recovery["home"][:-1])},
                    speed_scale=0.25,
                )
                if not plan.succeeded or plan.plan is None:
                    raise RuntimeError(f"Recovery planning failed: {plan.message}")
                self._sim.validate_primitive_recovery_plan(plan.plan.trajectory)
                self._pause(0)
                result = self._manipulation.execute(blocking=False, plan_id=plan.plan.plan_id)
                if result.status is not ExecutionStatus.ACCEPTED:
                    raise RuntimeError(f"Recovery was rejected: {result}")
                deadline = time.monotonic() + 60
                while True:
                    self._pause(0.05)
                    result = self._manipulation.wait_for_execution(timeout=0.01)
                    if result.succeeded:
                        break
                    if (
                        result.status
                        not in (
                            ExecutionStatus.EXECUTING,
                            ExecutionStatus.ACCEPTED,
                            ExecutionStatus.TIMED_OUT,
                        )
                        or time.monotonic() > deadline
                    ):
                        raise RuntimeError(f"Recovery failed: {result}")
                self._pause(0.5)
            report["recovered"] = self._sim.finish_primitive_recovery()

        return self._start("recover", operation, recovery=True)

    @skill
    def reset_scene(self) -> str:
        """Reset objects and both hands only on an explicit user reset request; requires idle actions."""

        def operation(report: dict[str, Any]) -> None:
            self._stop_control()
            if not self._sim.reset():
                raise RuntimeError("Simulator rejected reset")
            self._pause(0.8)

        return self._start("reset", operation, recovery=True)

    @rpc
    def stop(self) -> None:
        with self._lock:
            self._closing = True
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        super().stop()

    def _follow(self, path: list[list[float]], report: dict[str, Any]) -> None:
        if len(path) < 2:
            return
        path = self._sim.validate_object_navigation(path)
        before = self._sim.primitive_state()
        self._pause(0)
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
        self._phase("prepare_carry", report)
        carry = self._sim.classical_carry_posture()
        if carry:
            self._drive(carry, report)
        self._phase("navigate", report)
        plan = self._sim.prepare_object_navigation(destination, arm, stance)
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
            self._sim.finish_object_navigation()
        finally:
            self._control.task_invoke(APARTMENT_NAV_TASK, "cancel", {})
            self._sim.stop_primitive_base()

    def _position_base(self, selection: dict[str, Any], report: dict[str, Any]) -> None:
        report["base_plan_ids"] = []
        for waypoint in selection["base_waypoints"][1:]:
            target = np.asarray(waypoint)
            if np.max(np.abs(target - self._sim.primitive_state()["base_pose"])) <= 0.004:
                continue
            plan = self._manipulation.plan_to_joints(
                {
                    "moving_base": JointState(
                        name=list(R1PRO_PLANAR_BASE.joint_names), position=target.tolist()
                    )
                },
                speed_scale=0.7,
            )
            if not plan.succeeded or plan.plan is None:
                raise RuntimeError(f"SDK prepositioning plan failed: {plan.message}")
            self._sim.validate_primitive_base_plan(plan.plan.trajectory)
            report["base_plan_ids"].append(plan.plan.plan_id)
            self._pause(0)
            report["motion_started"] = True
            execution = self._manipulation.execute(blocking=False, plan_id=plan.plan.plan_id)
            if execution.status is not ExecutionStatus.ACCEPTED:
                raise RuntimeError(f"SDK prepositioning was rejected: {execution}")
            deadline = time.monotonic() + 90
            while True:
                self._pause(0.05)
                state = self._sim.primitive_state()
                if state["error"]:
                    raise RuntimeError(state["error"])
                execution = self._manipulation.wait_for_execution(timeout=0.01)
                if execution.succeeded:
                    break
                if execution.status not in (
                    ExecutionStatus.EXECUTING,
                    ExecutionStatus.TIMED_OUT,
                    ExecutionStatus.ACCEPTED,
                ):
                    raise RuntimeError(f"SDK prepositioning failed: {execution}")
                if time.monotonic() > deadline:
                    raise RuntimeError("SDK prepositioning timed out")
            self._sim.stop_primitive_base()
            self._pause(1.0)
            if np.max(np.abs(target - self._sim.primitive_state()["base_pose"])) > 0.01:
                raise RuntimeError(
                    "Measured base is outside the classical prepositioning tolerance"
                )

    def _prepare_posture(self, selection: dict[str, Any], report: dict[str, Any]) -> None:
        stance = selection["reachability"]
        positions = stance["ready_joints"]
        actual = self._sim.primitive_state()["joint_positions"]
        error = max(
            abs(actual[name] - positions[i]) for i, name in enumerate(R1PRO_PICK_PLACE_JOINTS[:18])
        )
        report["initial_posture_error_rad"] = error
        index = int(selection["object"].removeprefix("object_")) - 1
        target = np.asarray(stance.get("pregrasp", stance.get("preplace")))
        if error > 0.02:
            points = self._sim.classical_posture(
                index,
                stance["arm"],
                positions,
                target.tolist() if report.get("phase") == "preplace" else None,
            )
            report["posture_waypoints"] = len(points)
            self._drive(points, report)
        for attempt in range(3):
            actual_pose = np.asarray(self._sim.primitive_state()["tcp_poses"][stance["arm"]])
            position_error = float(np.linalg.norm(actual_pose[:3, 3] - target[:3, 3]))
            orientation_error = float(np.linalg.norm(actual_pose[:3, :3] - target[:3, :3]))
            report["staging_error"] = dict(
                position_m=position_error, rotation_matrix=orientation_error
            )
            if position_error <= 0.004 and orientation_error <= 0.015:
                break
            if attempt == 2:
                raise RuntimeError("Measured TCP did not reach the staged pose")
            self._drive(self._sim.classical_align(index, stance["arm"], target.tolist()), report)
        actual = self._sim.primitive_state()["joint_positions"]
        report["final_posture_error_rad"] = max(
            abs(actual[name] - positions[i]) for i, name in enumerate(R1PRO_PICK_PLACE_JOINTS[:18])
        )

    def _phase(self, phase: str, report: dict[str, Any]) -> None:
        self._pause(0)
        report["phase"] = phase
        with self._lock:
            self._action["phase"] = phase

    def _stop_control(self) -> None:
        self._control.task_invoke(APARTMENT_NAV_TASK, "cancel", {})
        result = self._manipulation.cancel()
        if result.status in (ExecutionStatus.UNCERTAIN, ExecutionStatus.FAULT):
            raise RuntimeError(result.message)
        for task in ("joint_trajectory", "primitive_left", "primitive_right"):
            if not self._control.cancel_trajectory(task).safe:
                raise RuntimeError(f"Cannot confirm {task} stopped")
        self._sim.stop_primitive_base()

    def _drive(self, points: list[list[float]], report: dict[str, Any]) -> None:
        self._pause(0)
        if not points:
            raise RuntimeError("Planner returned no executable waypoints")
        before = self._sim.primitive_state()
        if before.get("error"):
            raise RuntimeError(before["error"])
        if len(points) == 1:
            # The SDK may return a single pose when start and goal coincide.
            # Represent its hold explicitly; the caller still verifies TCP
            # staging and contacts before proceeding to the next phase.
            points = [points[0], points[0]]
        carrying = any(row.get("held_by") for row in before.get("objects", []))
        generator = JointTrajectoryGenerator(
            num_joints=20,
            max_velocity=([0.08] * 4 + [0.25] * 14 if carrying else [0.6] * 18) + [0.05, 0.05],
            max_acceleration=([0.15] * 4 + [0.5] * 14 if carrying else [1.5] * 18) + [0.2, 0.2],
            points_per_segment=8,
        )
        trajectory = generator.generate(points)
        trajectory.joint_names = list(R1PRO_PICK_PLACE_JOINTS)
        self._pause(0)
        accepted = self._control.execute_trajectory(trajectory, "joint_trajectory")
        if accepted.status is not TrajectoryExecutionStatus.ACCEPTED:
            raise RuntimeError(f"Cartesian trajectory rejected: {accepted}")
        report["motion_started"] = True
        duration = trajectory.points[-1].time_from_start
        deadline = time.monotonic() + duration + 10
        active = before.get("active")
        protected_grasp = None
        if active and report.get("phase") in ("lift", "preplace", "lower_to_support"):
            side = active[1]
            protected_grasp = next(
                (
                    row["index"]
                    for row in before["objects"]
                    if row["grasped"] and side in row["contacting_arms"]
                ),
                None,
            )
        fresh, stamp, stable = time.monotonic(), -1.0, 0
        settle_velocity = (
            0.0025
            if report.get("phase") in ("stage_pregrasp", "approach", "preplace", "lower_to_support")
            else 0.01
        )
        while time.monotonic() < deadline:
            self._pause(0.05)
            state = self._sim.primitive_state()
            if state["error"]:
                raise RuntimeError(state["error"])
            if state["sim_time"] > stamp:
                fresh, stamp = time.monotonic(), state["sim_time"]
                error = max(
                    abs(state["joint_positions"][n] - points[-1][i])
                    for i, n in enumerate(R1PRO_PICK_PLACE_JOINTS[:18])
                )
                velocities = state.get("joint_velocities", {})
                moving = max(abs(velocities.get(n, 0.0)) for n in R1PRO_PICK_PLACE_JOINTS[:18])
                commands = state.get("joint_commands")
                delivered = (
                    commands is None
                    or max(
                        abs(commands[n] - points[-1][i])
                        for i, n in enumerate(R1PRO_PICK_PLACE_JOINTS)
                    )
                    <= 1e-5
                )
                # Millimetre contact moves need a settled mechanism. A 0.03
                # rad/s endpoint can still move the TCP several mm per second
                # while the next IK query assumes its measured pose is static.
                stable = (
                    stable + 1 if delivered and error <= 0.02 and moving <= settle_velocity else 0
                )
            if protected_grasp is not None and not state["objects"][protected_grasp]["grasped"]:
                raise RuntimeError("Selected object lost two-finger contact during transfer")
            if time.monotonic() - fresh > 2:
                raise RuntimeError("Simulation stopped updating during Cartesian execution")
            task_state = self._control.task_invoke("joint_trajectory", "get_state", {})
            if task_state not in (TrajectoryState.EXECUTING, TrajectoryState.COMPLETED):
                raise RuntimeError("Cartesian task stopped before completing")
            if task_state == TrajectoryState.COMPLETED and stable >= 3:
                # Gripper commands terminate against the object; joint error is
                # not a grasp test. The caller verifies actual finger contacts.
                report.setdefault("trajectory_checks", []).append(
                    dict(
                        phase=report.get("phase"),
                        target=points[-1],
                        commanded=state.get("joint_commands"),
                        measured=state["joint_positions"],
                        tcp=state.get("tcp_poses"),
                    )
                )
                return
        raise RuntimeError("Measured Cartesian trajectory did not finish")

    def _line(
        self, index: int, arm: str, target: list[list[float]], report: dict[str, Any]
    ) -> None:
        self._drive(self._sim.classical_line(index, arm, target), report)

    def _gripper(self, arm: str, opening: float, report: dict[str, Any]) -> None:
        state = self._sim.primitive_state()
        start = [state["joint_commands"][n] for n in R1PRO_PICK_PLACE_JOINTS]
        target = list(start)
        target[18 if arm == "left" else 19] = opening
        self._drive([start, target], report)
        self._pause(0.3)

    def _seek_support(self, chosen: dict[str, Any], report: dict[str, Any]) -> None:
        """Confirm the intended physical support before releasing, within a 10 mm descent."""
        expected = set(chosen["region"]["support_geoms"])
        target = None
        initial_z = None
        for step in range(21):
            self._pause(0.2)
            state = self._sim.primitive_state()
            row = state["objects"][chosen["index"]]
            actual = np.asarray(state["tcp_poses"][chosen["arm"]], dtype=float)
            if initial_z is None:
                initial_z = float(actual[2, 3])
            report.setdefault("support_samples", []).append(
                dict(
                    step=step,
                    tcp=state.get("tcp_poses", {}).get(chosen["arm"]),
                    object_position=row.get("position"),
                    support_geoms=row["support_geoms"],
                    commanded_target=None if target is None else target.tolist(),
                )
            )
            supports = set(row["support_geoms"])
            if supports & expected:
                if not row["upright"]:
                    raise RuntimeError("Object tipped before release")
                report["support_before_release"] = sorted(supports & expected)
                report["support_descent_m"] = initial_z - float(actual[2, 3])
                return
            if supports:
                raise RuntimeError("Object contacted a different surface before release")
            if step < 20:
                target = actual.copy()
                target[2, 3] -= 0.0005
                if initial_z - float(target[2, 3]) > 0.01:
                    break
                self._line(chosen["index"], chosen["arm"], target.tolist(), report)
        raise RuntimeError("No intended support contact within the bounded descent; holding grip")

    @skill
    def get_scene(self) -> str:
        """Inspect fresh object IDs, types, colors, robot-relative sides and held objects.

        This demo uses simulation instance labels and virtual depth scans.
        """
        state = self._sim.primitive_state()
        for row in state["objects"]:
            row["color"] = color_name(row["rgba"])
        return json.dumps({**state, "action": self._status(), "controller": "classical_graspgenx"})

    @skill
    def get_surfaces(self) -> str:
        """Inspect measured support regions before choosing a placement."""
        return json.dumps(self._sim.primitive_state()["defined_regions"])

    @skill
    def pick_object(self, object: str = "nearest", arm: str = "auto") -> str:
        """Approach, grasp, lift and HOLD the requested item with GraspGenX and DimOS.

        Args:
            object: Exact ID or combined attributes, e.g. blue carton on the left.
            arm: auto compares free hands; left/right strictly preserves that hand.
        """
        state = self._sim.primitive_state()
        try:
            if arm not in (*ARMS, "auto"):
                raise ValueError("Choose auto, left or right")
            if arm in ARMS and state["held_objects"][arm]:
                raise ValueError(f"The {arm} hand is occupied")
            index = resolve_classical_object(state["objects"], object)
        except ValueError as exc:
            return json.dumps(dict(accepted=False, reason=str(exc)))

        def operation(report: dict[str, Any]) -> None:
            self._stop_control()
            report["requested_object"] = object
            report["selected_object"] = f"object_{index + 1}"
            self._phase("perceive", report)
            cloud = self._sim.classical_object_cloud(index)
            self._phase("generate_grasps", report)
            candidates = self._grasp_generator.propose_grasps(cloud)
            report["grasp_candidates"] = len(candidates)
            self._phase("assess_reachability", report)
            options = self._sim.assess_classical_pick(index, candidates, arm)
            if not options:
                raise RuntimeError(
                    "No GraspGenX candidate has a clear approach and lift with the requested hand"
                )
            chosen = options[0]
            side = chosen["arm"]
            if arm not in ("auto", side):
                raise RuntimeError("Reachability attempted to substitute the requested arm")
            report["grasp"] = chosen
            current = np.asarray(self._sim.primitive_state()["base_pose"])
            desired = np.asarray(chosen["base_pose"])
            if np.linalg.norm(current[:2] - desired[:2]) > 0.7:
                self._phase("navigate", report)
                self._navigate(f"object_{index + 1}", side, report, chosen["base_pose"])
            stance = dict(chosen, target=chosen["source_position"])
            selection = self._sim.prepare_reachable_primitive("pick", side, index, "", stance)
            report["selection"] = selection
            self._phase("position_body", report)
            self._position_base(selection, report)
            self._phase("stage_pregrasp", report)
            self._prepare_posture(selection, report)
            self._phase("approach", report)
            self._line(index, side, chosen["tcp"], report)
            self._phase("close_gripper", report)
            self._gripper(side, 0.0, report)
            closed = self._sim.primitive_state()["objects"][index]
            if side not in closed["grasping_arms"]:
                raise RuntimeError("The requested hand did not establish two-pad object contact")
            self._phase("lift", report)
            target = np.asarray(chosen["tcp"])
            target[2, 3] += 0.12
            self._line(index, side, target.tolist(), report)
            self._pause(0.5)
            final = self._sim.primitive_state()
            if final["held_objects"][side] != f"object_{index + 1}" or not final["complete"]:
                raise RuntimeError("The selected object did not survive the verified lift")
            self._phase("holding", report)

        return self._start("pick", operation)

    @skill
    def place_object(self, region: str, arm: str = "auto") -> str:
        """Place the held item on an empty supported region, then release and retreat.

        Args:
            region: A region returned by get_surfaces, or tray/table.
            arm: Holding hand; auto requires exactly one occupied hand.
        """
        held = self._sim.primitive_state()["held_objects"]
        choices = [side for side in ARMS if held[side] and arm in ("auto", side)]
        if len(choices) != 1:
            return json.dumps(dict(accepted=False, reason="Specify one occupied hand"))
        side = choices[0]

        def operation(report: dict[str, Any]) -> None:
            self._stop_control()
            self._phase("find_empty_support", report)
            goals = self._sim.classical_place_goals(side, region)
            if not goals:
                raise RuntimeError(
                    "No empty spot has a clear placement and release corridor with this hand"
                )
            chosen = goals[0]
            report["placement"] = chosen
            current = np.asarray(self._sim.primitive_state()["base_pose"])
            desired = np.asarray(chosen["base_pose"])
            if np.linalg.norm(current[:2] - desired[:2]) > 0.7:
                self._phase("navigate", report)
                destination = "worktable" if region in ("tray", "table") else region
                self._navigate(destination, side, report, chosen["base_pose"])
                # Carrying and turning change the measured grasp transform.
                # Choose the final support corridor from the arrived state.
                goals = self._sim.classical_place_goals(side, region)
                if not goals:
                    raise RuntimeError("No supported placement corridor after arrival")
                chosen = goals[0]
                report["placement"] = chosen
            selection = self._sim.prepare_reachable_primitive(
                "place", side, chosen["index"], region, chosen
            )
            self._phase("position_body", report)
            self._position_base(selection, report)
            self._phase("preplace", report)
            self._prepare_posture(selection, report)
            self._phase("lower_to_support", report)
            self._line(chosen["index"], side, chosen["tcp"], report)
            self._seek_support(chosen, report)
            self._phase("release", report)
            self._gripper(side, 0.05, report)
            self._phase("retreat", report)
            self._line(chosen["index"], side, chosen["preplace"], report)
            self._pause(0.5)
            if not self._sim.primitive_state()["complete"]:
                raise RuntimeError("The released item did not settle inside the requested support")
            self._phase("placed", report)

        return self._start("place", operation)

    @skill
    def go_to(self, destination: str) -> str:
        """Navigate to kitchen, dining_table or worktable while keeping held objects. Never release."""

        def operation(report: dict[str, Any]) -> None:
            self._phase("navigate", report)
            held = self._sim.primitive_state()["held_objects"]
            arm = next((side for side in ARMS if held[side]), "right")
            self._navigate(destination, arm, report)
            self._phase("arrived", report)

        return self._start("navigate", operation)
