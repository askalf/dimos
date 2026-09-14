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

"""Characterize Alfred's FlowBase and emit the artifact the follower reads.

    dimos --rerun-host 0.0.0.0 run alfred-autotune

The holonomic pose follower self-calibrates from a pose-domain artifact, and the
only one vendored is the Go2's. Running Alfred's follower against a Go2 plant
model means commanded and achieved speeds disagree, so the base runs hot and
glides past its goal. This blueprint produces Alfred's own.

It drives a step battery through AlfredHighLevel while Point-LIO supplies world
pose, and records each excitation run as one episode. Feedback is pose rather
than wheel odometry on purpose: the fitter identifies what the robot actually
did in the world, and a caster base's own encoders slip exactly when a step
excitation is most informative.

The run is the operator's to supervise. The battery commands motion at up to
Alfred's declared vmax on all three axes, and nothing repositions the base
between runs - clear a few metres, watch it, and reposition during the settle
dwells. There is no obstacle checking here; this blueprint is the plant test,
not navigation.

Afterwards the fit, tune, and emit steps need no robot:

    from dimos.control.autotune.runner import autotune_offline
    outputs = autotune_offline(profile, segments_by_channel, robot_id="alfred", sim_or_hw="hw")

Write the resulting artifact next to the Go2's and point the follower's
``artifact_path`` at it.
"""

from __future__ import annotations

import threading
from typing import Any

from dimos.control.autotune.drive import run_battery
from dimos.control.autotune.excitation import step_battery
from dimos.control.autotune.live import make_sinks
from dimos.control.autotune.profile import BatteryConfig, Channel, RobotProfile
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
from dimos.robot.diy.alfred.alfred_model import ALFRED_BASE_VELOCITY_LIMITS
from dimos.robot.diy.alfred.config import ALFRED
from dimos.robot.diy.alfred.effector_high_level import AlfredHighLevel
from dimos.robot.diy.alfred.mount_tf import AlfredMountTf
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

ODOM_FRAME = "odom"
LIDAR_FRAME = "mid360_link"

_VX_MAX, _VY_MAX, _WZ_MAX = ALFRED_BASE_VELOCITY_LIMITS


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


class AlfredAutotuneRecorderConfig(RecorderConfig):
    pass


class AlfredAutotuneRecorder(Recorder):
    """Captures the two streams the fitters read, segmented by episode.

    Command and pose only. The imitation collector records a camera and the
    coordinator's joints; neither identifies a base plant, and both would bloat
    a session that is already one long motion test.
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


class AlfredAutotuneDriver(Module):
    """Plays the excitation battery and marks each run as an episode."""

    config: AlfredAutotuneDriverConfig

    cmd_vel: Out[Twist]
    status: Out[EpisodeStatus]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @rpc
    def start(self) -> None:
        super().start()
        if not self.config.armed:
            logger.warning(
                "Alfred autotune is not armed; no motion will be commanded. "
                "Re-run with --alfredautotunedriver.armed true once the area is clear."
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
        try:
            played = run_battery(
                runs,
                sink,
                episodes,
                clock,
                tick_hz=self.config.tick_hz,
                settle_s=self.config.settle_s,
            )
        except Exception:
            logger.exception("Alfred autotune battery failed; zeroing the base")
            self.cmd_vel.publish(Twist())
            return
        self.cmd_vel.publish(Twist())
        logger.info(
            "Alfred autotune battery complete. Fit offline with "
            "dimos.control.autotune.runner.autotune_offline",
            episodes=played,
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
