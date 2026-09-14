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

    dimos run alfred-autotune --alfredautotunedriver.armed true

One command. It drives the excitation battery, fits a FOPDT model per axis,
tunes the gains and writes ``alfred_posedomain.json`` next to the Go2's, plus a
characterization report. Nothing offline to run afterwards.

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

SUPERVISE THE RUN. The battery commands motion at up to Alfred's declared vmax
on all three axes with no obstacle checking of any kind: this is the plant test,
not navigation. Clear a few metres, arm it deliberately, and reposition the base
by hand during the settle dwells. It refuses to move unless armed.

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

from dimos.control.autotune.drive import play_run
from dimos.control.autotune.excitation import ExcitationRun, step_battery
from dimos.control.autotune.live import make_sinks
from dimos.control.autotune.profile import BatteryConfig, Channel, RobotProfile
from dimos.control.autotune.report import write_tuned_artifact
from dimos.control.autotune.runner import autotune_offline
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.imitation.collection.episode_monitor import EpisodeStatus
from dimos.memory.module import Recorder, RecorderConfig
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
    pass


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
    # Long enough for the base to come to rest, and for an operator to reposition
    # it by hand between runs.
    settle_s: float = 3.0
    # Off by default. Arming is a deliberate act: this commands real motion on
    # all three axes with no obstacle checking of any kind.
    armed: bool = False
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buf_lock = threading.Lock()
        self._buf: list[tuple[float, float, float, float]] = []
        self._capturing = False

    @rpc
    def start(self) -> None:
        super().start()
        self.start_pose.subscribe(self._on_pose)
        if not self.config.armed:
            logger.warning(
                "Alfred autotune is not armed; no motion will be commanded. Re-run with "
                "--alfredautotunedriver.armed true once the area is clear and supervised."
            )
            return
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
                self._capture(True)
                play_run(run, sink, episodes, clock, tick_hz=self.config.tick_hz)
                segment = self._segment(run)
                if segment is not None:
                    segments[run.channel].append(segment)
                played += 1
                logger.info("Alfred autotune run done", run=run.label, index=index, of=len(runs))
                # Inter-run settle: hold zero so transients die before the next
                # excitation, and so the operator can reposition the base.
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
