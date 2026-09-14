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

"""Characterize Alfred's FlowBase and write the artifact the follower reads.

    dimos --rerun-host 0.0.0.0 run alfred-autotune

One command. It drives the excitation battery, fits a FOPDT model per axis,
tunes the gains and writes ``alfred_posedomain.json`` next to the Go2's, plus a
characterization report. Nothing offline to run afterwards.

THE OPERATOR PACES IT. Nothing moves until you advance the gate, and the gate
comes back between every run - 54 of them. Click anywhere in the rerun viewer to
play the next run; drive the base with the viewer's keyboard in between to
reposition it, which is most of what the run actually involves. The Go2's
benchmark gates the same way, off its KeyboardTeleop; Alfred is headless over
ssh, so the gate and the driving both come from the viewer instead of a pygame
window that has no display to open on.

    app.AlfredAutotuneDriver.skip()    drop a run the robot is badly placed for
    app.AlfredAutotuneDriver.abort()   end the battery, fit what was measured

The holonomic pose follower self-calibrates from a pose-domain artifact and the
only one vendored is the Go2's. Pointing Alfred's follower at a quadruped's plant
model means commanded and achieved speeds disagree, so the base runs hot and
glides past its goal. This produces Alfred's own.

Feedback is world pose from Point-LIO, not wheel odometry: the fitter identifies
what the robot actually did in the world, and a caster base's own encoders slip
exactly when a step excitation is most informative. The driver projects that
world pose onto the axis under test - forward travel for vx, lateral for vy,
unwrapped yaw for wz - because the pose fitter identifies on a scalar per axis,
not on a world pose.

The battery commands motion at up to Alfred's declared vmax on all three axes
with no obstacle checking of any kind: this is the plant test, not navigation.
Clear a few metres and watch it. Nothing repositions the base automatically -
that is what the gate is for.

The raw streams are recorded too, so a collection can be re-fitted later without
re-driving the robot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
from typing import Any

import numpy as np
from pydantic import Field

from dimos.control.autotune.drive import play_run
from dimos.control.autotune.excitation import ExcitationRun, step_battery
from dimos.control.autotune.live import make_sinks
from dimos.control.autotune.profile import BatteryConfig, Channel, RobotProfile
from dimos.control.autotune.report import write_tuned_artifact
from dimos.control.autotune.runner import autotune_offline
from dimos.control.benchmarking.gate import GATE_ADVANCE, GATE_QUIT, GATE_SKIP
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.global_config import global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.imitation.collection.episode_monitor import EpisodeStatus
from dimos.memory.module import Recorder, RecorderConfig
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.navigation.nav_3d.mls_planner.start_relay import StartRelay
from dimos.robot.diy.alfred.alfred_model import (
    ALFRED_BASE_VELOCITY_LIMITS,
    ALFRED_CHARACTERIZATION_REPORT,
    ALFRED_FOLLOWER_ARTIFACT,
)
from dimos.robot.diy.alfred.config import ALFRED
from dimos.robot.diy.alfred.effector_high_level import AlfredHighLevel
from dimos.robot.diy.alfred.mount_tf import AlfredMountTf
from dimos.utils.logging_config import setup_logger
from dimos.visualization.vis_module import vis_module

logger = setup_logger()

ODOM_FRAME = "odom"
LIDAR_FRAME = "mid360_link"

_VX_MAX, _VY_MAX, _WZ_MAX = ALFRED_BASE_VELOCITY_LIMITS

# alfred_model owns where these live, so alfred-nav resolves the same paths.
ALFRED_ARTIFACT_PATH = ALFRED_FOLLOWER_ARTIFACT
ALFRED_REPORT_PATH = ALFRED_CHARACTERIZATION_REPORT


def alfred_autotune_profile() -> RobotProfile:
    """What Alfred is, declared for autotune.

    Amplitudes are fractions of each channel's vmax, so the battery scales with
    whatever limits Alfred is commissioned at rather than a fixed m/s grid.

    The pose fitter is chosen over the velocity fitter deliberately. Point-LIO
    publishes world pose; differentiating it to recover body velocity would smear
    the step edge that carries the time constant, and the pose fitter
    forward-models instead of differentiating.
    """
    return RobotProfile(
        name="alfred",
        command_interface="twist",
        odom_type="pose",
        channels=[
            Channel(name="vx", vmax=_VX_MAX),
            Channel(name="vy", vmax=_VY_MAX),
            Channel(name="wz", vmax=_WZ_MAX),
        ],
        fitter="pose",
        controller_form="velocity_pi",
        command_stream="cmd_vel",
        feedback_stream="start_pose",
        battery=BatteryConfig(amplitude_fractions=(0.25, 0.5, 0.75), repeats=3),
    )


def project_to_axis(
    samples: list[tuple[float, float, float, float]], channel: str
) -> tuple[np.ndarray, np.ndarray]:
    """World pose samples ``(t, x, y, yaw)`` to the scalar the pose fitter reads.

    The fitter identifies one axis at a time against a single measured signal, so
    a world pose has to be resolved onto the axis under test. Translation is taken
    in the body frame the run STARTED in: a step in vy moves the robot sideways in
    the world, and only the initial heading says which world direction that was.

    Yaw is unwrapped before differencing so a run that crosses +/-pi does not read
    as a 2*pi jump - which would otherwise fit as an enormous gain.
    """
    t0, x0, y0, yaw0 = samples[0]
    t = np.array([s[0] - t0 for s in samples], dtype=float)

    if channel == "wz":
        yaw = np.unwrap(np.array([s[3] for s in samples], dtype=float))
        return t, yaw - yaw[0]

    dx = np.array([s[1] - x0 for s in samples], dtype=float)
    dy = np.array([s[2] - y0 for s in samples], dtype=float)
    cos0, sin0 = math.cos(yaw0), math.sin(yaw0)
    if channel == "vx":
        return t, dx * cos0 + dy * sin0
    if channel == "vy":
        return t, -dx * sin0 + dy * cos0
    raise ValueError(f"unknown channel {channel!r}; expected vx, vy or wz")


class AlfredAutotuneRecorderConfig(RecorderConfig):
    # There is no `world` frame in this blueprint - Point-LIO publishes `odom`
    # and the mount tree roots at the lidar - so the default root_frame makes
    # every pose lookup fail, twice per message, at odometry rate. That flood
    # buries the gate prompts the operator is meant to be reading.
    root_frame: str = ODOM_FRAME
    # A twist and an episode marker have no pose to anchor, by nature.
    poseless_streams: list[str] = Field(default_factory=lambda: ["cmd_vel", "status"])


class AlfredAutotuneRecorder(Recorder):
    """Captures the two streams the fitters read, segmented by episode.

    The driver fits in-process, so this is not on the critical path; it exists so
    a collection can be re-fitted later without re-driving the robot. Command and
    pose only: the imitation collector records a camera and the coordinator's
    joints, neither of which identifies a base plant.
    """

    config: AlfredAutotuneRecorderConfig

    cmd_vel: In[Twist]  # what was asked of the base
    start_pose: In[PoseStamped]  # what the world says it did
    status: In[EpisodeStatus]  # run boundaries


class AlfredAutotuneDriverConfig(ModuleConfig):
    # 50 Hz matches AlfredHighLevel's odometry poll: commanding faster than the
    # feedback arrives buys resolution the fit cannot use.
    tick_hz: float = 50.0
    step_duration_s: float = 4.0
    # Held after a run so the base is at rest before the operator takes over. It
    # is not the repositioning window - the gate is, and it waits as long as you
    # need.
    settle_s: float = 1.0
    # 0 waits forever. The operator paces this run; a battery that marched on by
    # itself would be driving an unattended robot.
    gate_timeout_s: float = 0.0
    artifact_path: str = ALFRED_ARTIFACT_PATH
    report_path: str = ALFRED_REPORT_PATH
    # A run this short cannot carry a time constant; fitting it would pollute the
    # pool rather than fail loudly.
    min_samples_per_run: int = 8


class AlfredAutotuneDriver(Module):
    """Plays the battery, captures pose per run, fits and writes the artifact."""

    config: AlfredAutotuneDriverConfig

    cmd_vel: Out[Twist]
    status: Out[EpisodeStatus]
    start_pose: In[PoseStamped]
    # The operator's gate and their hands on the base, both from the rerun
    # viewer: this runs on a headless robot over ssh, so a pygame window is not
    # available the way it is for the Go2's KeyboardTeleop.
    clicked_point: In[PointStamped]
    tele_cmd_vel: In[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buf_lock = threading.Lock()
        self._buf: list[tuple[float, float, float, float]] = []
        self._capturing = False
        self._gate = threading.Event()
        self._gate_action = GATE_ADVANCE
        # Teleop only reaches the base between runs. The driver is the sole
        # writer of cmd_vel, so a stray key during an excitation cannot corrupt
        # the step it is trying to identify.
        self._relaying = False

    @rpc
    def start(self) -> None:
        super().start()
        self.start_pose.subscribe(self._on_pose)
        self.clicked_point.subscribe(self._on_click)
        self.tele_cmd_vel.subscribe(self._on_teleop)
        logger.warning(
            "Alfred autotune is waiting at the gate. Nothing moves until you advance it: "
            "click anywhere in the rerun viewer to play each run, and drive the base with "
            "the viewer's keyboard between runs to reposition it. "
            "app.AlfredAutotuneDriver.skip() drops a run, .abort() ends the battery."
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alfred-autotune", daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop.set()
        # Whatever else is unwinding, the base must not be left with a standing
        # command from a half-played run.
        try:
            self.cmd_vel.publish(Twist())
        except Exception:
            logger.exception("Failed to zero Alfred's base on autotune shutdown")
        super().stop()

    def _on_pose(self, msg: PoseStamped) -> None:
        with self._buf_lock:
            if self._capturing:
                self._buf.append((float(msg.ts), float(msg.x), float(msg.y), float(msg.yaw)))

    def _capture(self, on: bool) -> list[tuple[float, float, float, float]]:
        """Arm or disarm pose capture; returns what was captured."""
        with self._buf_lock:
            captured = list(self._buf)
            self._buf = []
            self._capturing = on
        return captured

    def _on_click(self, msg: PointStamped) -> None:
        """A click anywhere in the viewer advances the battery by one run."""
        self._gate_action = GATE_ADVANCE
        self._gate.set()

    def _on_teleop(self, msg: Twist) -> None:
        """Operator driving between runs. Ignored while a run is playing."""
        if self._relaying:
            self.cmd_vel.publish(msg)

    @rpc
    def advance(self) -> None:
        """Play the next run. The same thing a viewer click does."""
        self._on_click(PointStamped())

    @rpc
    def skip(self) -> None:
        """Drop the next run and move on - a run the robot was badly placed for."""
        self._gate_action = GATE_SKIP
        self._gate.set()

    @rpc
    def abort(self) -> None:
        """Stop the battery and fit whatever has been measured so far."""
        self._gate_action = GATE_QUIT
        self._gate.set()

    def _wait_for_gate(self, run: ExcitationRun, index: int, total: int) -> int:
        """Block until the operator advances, skips or aborts.

        The base is the operator's while this waits: teleop is relayed through so
        they can reposition, which on a 54-run battery is most of what the run
        actually involves.
        """
        self._gate.clear()
        self._relaying = True
        logger.info(
            "Waiting at the gate - click the viewer to play, or reposition with its keyboard",
            next_run=run.label,
            index=index,
            of=total,
        )
        try:
            timeout = self.config.gate_timeout_s or None
            while not self._gate.wait(timeout=0.2):
                if self._stop.is_set():
                    return GATE_QUIT
                if timeout is not None:
                    timeout -= 0.2
                    if timeout <= 0:
                        logger.warning("Gate timed out; ending the battery", run=run.label)
                        return GATE_QUIT
        finally:
            self._relaying = False
            # Whatever the operator left on the stick, the run starts from rest.
            self.cmd_vel.publish(Twist())
        return self._gate_action

    def _segment(self, run: ExcitationRun) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Close out one run's capture as a fitter segment, or None if too thin."""
        samples = self._capture(False)
        if len(samples) < self.config.min_samples_per_run:
            logger.warning(
                "Alfred autotune run produced too little pose to fit; dropping it",
                run=run.label,
                samples=len(samples),
                needed=self.config.min_samples_per_run,
            )
            return None
        t, measured = project_to_axis(samples, run.channel)
        return t, measured, float(run.amplitude)

    def _run(self) -> None:
        profile = alfred_autotune_profile()
        runs = step_battery(profile, duration_s=self.config.step_duration_s)
        logger.info(
            "Alfred autotune battery starting",
            runs=len(runs),
            channels=profile.channel_names,
            tick_hz=self.config.tick_hz,
        )
        sink, episodes, clock = make_sinks(
            profile,
            publish_twist=self.cmd_vel.publish,
            publish_status=self.status.publish,
        )

        segments: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {
            channel: [] for channel in profile.channel_names
        }
        played = 0
        try:
            for index, run in enumerate(runs, start=1):
                if self._stop.is_set():
                    logger.warning("Alfred autotune stopped early", played=played, of=len(runs))
                    break

                action = self._wait_for_gate(run, index, len(runs))
                if action == GATE_QUIT:
                    logger.warning(
                        "Battery aborted at the gate; fitting what was measured",
                        played=played,
                        of=len(runs),
                    )
                    break
                if action == GATE_SKIP:
                    logger.info("Run skipped at the gate", run=run.label, index=index)
                    continue

                self._capture(True)
                play_run(run, sink, episodes, clock, tick_hz=self.config.tick_hz)
                segment = self._segment(run)
                if segment is not None:
                    segments[run.channel].append(segment)
                played += 1
                logger.info("Alfred autotune run done", run=run.label, index=index, of=len(runs))
                # Let transients die before the operator takes the base back; the
                # gate, not this dwell, is where repositioning happens.
                sink.stop()
                clock.sleep(self.config.settle_s)
        except Exception:
            logger.exception("Alfred autotune battery failed; zeroing the base")
            self.cmd_vel.publish(Twist())
            return
        finally:
            self._capture(False)
            self.cmd_vel.publish(Twist())

        self._emit(profile, segments, played=played, total=len(runs))

    def _emit(
        self,
        profile: RobotProfile,
        segments: dict[str, list[tuple[np.ndarray, np.ndarray, float]]],
        *,
        played: int,
        total: int,
    ) -> None:
        """Fit, tune and write. A partial battery still emits what it measured."""
        fitted = {channel: len(segs) for channel, segs in segments.items()}
        if not any(fitted.values()):
            logger.error(
                "Alfred autotune captured no usable segments; nothing to fit. Check that "
                "Point-LIO is publishing and that start_pose reaches the driver.",
                played=played,
            )
            return

        try:
            outputs = autotune_offline(profile, segments, robot_id="alfred", sim_or_hw="hw")
        except Exception:
            logger.exception("Alfred autotune fit failed", segments=fitted)
            return

        try:
            # The package's own writer, so the artifact round-trips in the key
            # order the follower's from_artifact expects.
            artifact_path = write_tuned_artifact(self.config.artifact_path, outputs.artifact)
            report_path = Path(self.config.report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(outputs.report, indent=2))
        except Exception:
            logger.exception("Alfred autotune could not write its artifact")
            return

        logger.info(
            "Alfred autotune complete",
            artifact=str(artifact_path),
            report=str(report_path),
            runs_played=f"{played}/{total}",
            segments_fitted=fitted,
            valid_for_tuning=outputs.artifact.get("valid_for_tuning"),
        )
        logger.info(
            "Point the follower at it: "
            f'holonomic_pose_follower params {{"artifact_path": "{artifact_path}"}}'
        )


alfred_autotune = autoconnect(
    # The operator's console: their gate (a click) and their hands on the base
    # (the viewer's keyboard) both arrive here. The robot is headless over ssh,
    # so this is the console - there is no pygame window to open on it.
    # rerun_open="none": the robot is headless, so trying to open a native
    # window here only produces a winit error about DISPLAY not being set. The
    # operator connects a viewer from their own machine.
    vis_module(viewer_backend=global_config.viewer, rerun_config={"rerun_open": "none"}),
    AlfredHighLevel.blueprint().remappings([(AlfredHighLevel, "wheel_odometry", "odom_sources")]),
    AlfredMountTf.blueprint(root_frame=LIDAR_FRAME),
    # Pose feedback. Point-LIO owns odom -> mid360_link, so the mount tree roots
    # at the lidar here for the same reason it does in alfred-nav.
    PointLio.blueprint(
        frame_id=ODOM_FRAME,
        sensor_frame_id=LIDAR_FRAME,
        lidar_ip=ALFRED.mid360_ip,
    ),
    StartRelay.blueprint(world_frame=ODOM_FRAME, base_frame="base_link"),
    AlfredAutotuneDriver.blueprint(),
    AlfredAutotuneRecorder.blueprint(),
    # Point-LIO is a C++ native and speaks LCM only.
).global_config(robot_model="alfred", transport="lcm")
