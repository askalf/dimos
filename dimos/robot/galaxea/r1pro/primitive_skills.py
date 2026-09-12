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

"""Explicit pick/hold/place commands with per-arm ACT ownership and SDK prepositioning."""

from collections.abc import Callable
from concurrent.futures import CancelledError
import json
from pathlib import Path
import threading
import time
from typing import Any, cast

import numpy as np
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.manipulation.manipulation_spec import ExecutionStatus, ManipulationSpec
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.robot.galaxea.r1pro.config import R1PRO_PLANAR_BASE
from dimos.robot.galaxea.r1pro.home_spec import HomeControlSpec, PackingPolicySpec
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, Arm, Primitive
from dimos.robot.galaxea.r1pro.primitive_spec import PrimitiveSimSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def resolve_primitive_object(rows: list[dict[str, Any]], selector: str) -> int:
    """Include supported tray contents in selection; never substitute an ineligible ID."""
    available = [
        r for r in rows if r["upright"] and r["released"] and r["settled"] and r["support_geoms"]
    ]
    value = selector.strip().lower().replace("-", "_").replace(" ", "_")
    for row in rows:
        if value in (row["id"], row["object"]):
            if row not in available:
                raise ValueError(f"{value} is not a supported, settled, unheld pick source")
            return int(row["index"])
    if not available:
        raise ValueError("No supported unheld objects remain")
    keys = {
        "nearest": ("distance_m", 1),
        "furthest": ("distance_m", -1),
        "farthest": ("distance_m", -1),
        "rightmost": ("left_m", 1),
        "leftmost": ("left_m", -1),
    }
    if value in keys:
        key, sign = keys[value]
        return int(min(available, key=lambda r: (sign * r[key], r["index"]))["index"])
    shapes = [r for r in available if r["shape"] == value]
    if len(shapes) == 1:
        return int(shapes[0]["index"])
    raise ValueError("Use an object ID, unique shape, nearest/furthest or leftmost/rightmost")


class R1ProPrimitiveSkillsConfig(ModuleConfig):
    action_timeout: float = Field(default=40.0, gt=0, le=120)
    hold_seconds: float = Field(default=5.0, ge=1)


class R1ProPrimitiveSkills(Module):
    config: R1ProPrimitiveSkillsConfig
    _sim: PrimitiveSimSpec
    _control: HomeControlSpec
    _manipulation: ManipulationSpec
    _pick_right: PackingPolicySpec
    _place_right: PackingPolicySpec
    _pick_left: PackingPolicySpec
    _place_left: PackingPolicySpec

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
                target=self._run, args=(operation,), daemon=True, name="r1pro-primitive-action"
            )
            self._thread.start()
            return json.dumps(dict(accepted=True, **self._action))

    def _stop_control(self) -> None:
        errors = []
        for arm in ARMS:
            for primitive in ("pick", "place"):
                policy = cast("PackingPolicySpec", getattr(self, f"_{primitive}_{arm}"))
                try:
                    if policy.stop_rollout()["active"]:
                        errors.append(f"{primitive}/{arm} remains active")
                except Exception as exc:
                    errors.append(str(exc))
        try:
            result = self._manipulation.cancel()
            if result.status in (ExecutionStatus.UNCERTAIN, ExecutionStatus.FAULT):
                errors.append(result.message)
        finally:
            self._sim.stop_primitive_base()
        if errors:
            raise RuntimeError("; ".join(errors))

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
            logger.exception("Primitive action failed")
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
                output = (
                    Path(self._sim.prepare_primitive_session()["output"])
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

    def _execute(
        self, primitive: Primitive, arm: Arm, index: int, region: str, report: dict[str, Any]
    ) -> None:
        self._stop_control()
        selection = self._sim.prepare_primitive(primitive, arm, index, region)
        report["selection"] = selection
        target = np.asarray(selection["base_target"])
        if np.max(np.abs(target - self._sim.primitive_state()["base_pose"])) > 0.004:
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
            report["base_plan_id"] = plan.plan.plan_id
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
                raise RuntimeError("Measured base is outside the policy prepositioning tolerance")
        policy = cast("PackingPolicySpec", getattr(self, f"_{primitive}_{arm}"))
        policy.clear_rollout_observations()
        deadline = time.monotonic() + 120
        while True:
            self._pause(0.1)
            status = policy.preflight_rollout()
            if status["policy_ready"] and status["observations_ready"] and not status["last_error"]:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"Policy preflight failed: {status}")
        report["motion_started"] = True
        status = policy.start_rollout()
        if not status["active"]:
            raise RuntimeError(f"ACT did not start: {status}")
        report["history"] = []
        try:
            deadline = time.monotonic() + self.config.action_timeout
            stable = 0
            while True:
                self._pause(0.05)
                if not self._sim.is_simulation_running():
                    raise RuntimeError("Simulation stopped")
                status = policy.rollout_status()
                if status["last_error"] or not status["active"]:
                    raise RuntimeError(f"ACT stopped: {status}")
                state = self._sim.primitive_state()
                report["history"].append(state)
                if state["error"]:
                    raise RuntimeError(state["error"])
                stable = stable + 1 if state["complete"] else 0
                if stable >= 3:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"ACT {primitive}/{arm} timed out")
        finally:
            report["stopped"] = policy.stop_rollout()
        until = time.monotonic() + (self.config.hold_seconds if primitive == "pick" else 1.0)
        while time.monotonic() < until:
            self._pause(0.05)
            state = self._sim.primitive_state()
            if state["error"] or not state["complete"]:
                raise RuntimeError(
                    f"Physical {primitive} outcome did not persist after ACT stopped"
                )

    @skill
    def get_scene(self) -> str:
        """Read object IDs, measured positions, support regions and held objects for each hand."""
        return json.dumps(dict(**self._sim.primitive_state(), action=self._status()))

    @skill
    def pick_object(self, object: str = "nearest", arm: str = "right") -> str:
        """ACT grasp and lift with the requested hand, then STOP holding. Never place or release.

        Args:
            object: Object ID, unique shape, nearest/furthest, or leftmost/rightmost; includes tray objects.
            arm: left or right. The other hand preserves its held object.
        """
        if arm not in ARMS:
            return json.dumps(dict(accepted=False, reason="arm must be left or right"))
        state = self._sim.primitive_state()
        if state["held_objects"][arm] is not None:
            return json.dumps(dict(accepted=False, reason=f"{arm} hand is occupied"))
        try:
            index = resolve_primitive_object(state["objects"], object)
        except ValueError as exc:
            return json.dumps(dict(accepted=False, reason=str(exc)))
        return self._start(
            f"pick/{arm}/{index}",
            lambda report: self._execute("pick", cast("Arm", arm), index, "", report),
        )

    @skill
    def place_object(self, region: str, arm: str = "right") -> str:
        """Use the independent ACT placement policy for an already held object and explicit support region.

        Args:
            region: tray or table. A full region refuses placement and preserves the hold.
            arm: left or right, matching the hand holding the requested object.
        """
        if arm not in ARMS or region not in self._sim.primitive_state()["regions"]:
            return json.dumps(
                dict(
                    accepted=False,
                    reason="Use left/right and an available support region from get_scene",
                )
            )
        return self._start(
            f"place/{arm}/{region}",
            lambda report: self._execute("place", cast("Arm", arm), -1, region, report),
        )

    @skill
    def define_placement_region(
        self, name: str, x: float, y: float, width: float, depth: float
    ) -> str:
        """Name an explicit rectangular region on the physical worktable; no motion or training.

        Args:
            name: New name for later place_object calls.
            x: World-frame rectangle center X in meters.
            y: World-frame rectangle center Y in meters.
            width: Rectangle width along X in meters.
            depth: Rectangle depth along Y in meters.
        """
        with self._lock:
            if not self._done.is_set():
                return json.dumps(dict(accepted=False, reason="Wait for the active action"))
            try:
                return json.dumps(
                    dict(
                        accepted=True,
                        region=self._sim.define_placement_region(name, x, y, width, depth),
                    )
                )
            except ValueError as exc:
                return json.dumps(dict(accepted=False, reason=str(exc)))

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

        Recovery never retries an ACT grasp and never resets scene progress or the other hand.
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
