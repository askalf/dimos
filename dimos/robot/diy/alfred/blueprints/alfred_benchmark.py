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

"""Drive Alfred's follower through a path battery and score it offline.

    dimos --rerun-host 0.0.0.0 run alfred-benchmark      # drive, records runs
    python -m dimos.control.benchmarking.score <out_dir> # score, writes plots

Two blueprints:

``alfred-holonomic-controller`` is Alfred's base control on its own - the
coordinator, the follower, and the odometry it closes the loop on. No planner and
no manipulation, so a Path on ``/path`` is the only thing steering. It is the
Go2's ``unitree-go2-holonomic-controller`` shape, and useful by itself for
driving paths from any source.

``alfred-benchmark`` composes that with the Benchmarker, which publishes the path
battery, records what the robot did, and paces runs at an operator gate. Scoring
is offline and separate: it reads the recordings and emits cross-track error
against speed plus the executed-versus-reference overlays, which is how you see
whether the follower is actually tracking.

The gate is a viewer click, for the same reason autotune's is: the Go2 gates off
KeyboardTeleop, which is pygame, and Alfred is headless over ssh. Click to run
the next path; reposition with the viewer's keyboard in between.

The follower reads Alfred's tuned artifact if autotune has written one and the
Go2's otherwise, which is worth knowing before reading a bad score as a controller
problem - it may just be calibrated for a quadruped.
"""

from __future__ import annotations

from typing import Any

from dimos.control.benchmarking.benchmark import Benchmarker
from dimos.control.benchmarking.gate import GATE_ADVANCE
from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.control.coordinator import TaskConfig
from dimos.control.path_following_coordinator import PathFollowingCoordinator
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.global_config import global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.std_msgs.Int8 import Int8
from dimos.navigation.nav_3d.mls_planner.start_relay import StartRelay
from dimos.robot.diy.alfred.alfred_model import (
    ALFRED_BASE_VELOCITY_LIMITS,
    alfred_follower_artifact,
)
from dimos.robot.diy.alfred.config import ALFRED
from dimos.robot.diy.alfred.effector_high_level import AlfredHighLevel
from dimos.robot.diy.alfred.mount_tf import AlfredMountTf
from dimos.utils.logging_config import setup_logger
from dimos.visualization.vis_module import vis_module

logger = setup_logger()

ODOM_FRAME = "odom"
LIDAR_FRAME = "mid360_link"
ALFRED_BASE_HARDWARE_ID = "flowbase"
BASE_VELOCITY_TASK_NAME = "vel_flowbase"
NAV_FOLLOWER_TASK_NAME = "holonomic_follower"
# Fraction of vmax the corner regulator may not throttle below.
_CORNER_FLOOR = 0.25 * ALFRED_BASE_VELOCITY_LIMITS[0]

_VX_MAX = ALFRED_BASE_VELOCITY_LIMITS[0]
# Fractions of Alfred's own envelope, so the ladder scales if it is recommissioned.
ALFRED_BENCHMARK_SPEEDS = ",".join(f"{frac * _VX_MAX:.2f}" for frac in (0.3, 0.5, 0.7, 0.9))

_flowbase_hardware = HardwareComponent(
    hardware_id=ALFRED_BASE_HARDWARE_ID,
    hardware_type=HardwareType.BASE,
    joints=make_twist_base_joints(ALFRED_BASE_HARDWARE_ID),
    adapter_type="transport_lcm",
    auto_enable=True,
)


class AlfredBenchmarkCoordinator(PathFollowingCoordinator):
    """Carries the ``path`` and ``speed`` ports the follower task binds to."""


class AlfredViewerGateConfig(ModuleConfig):
    pass


class AlfredViewerGate(Module):
    """Turns a viewer click into the Benchmarker's operator gate.

    The Benchmarker paces each run on an ``Int8`` carrying the gate codes, and the
    Go2 sources those from KeyboardTeleop's ENTER/skip/quit. That is pygame, and
    this robot is headless over ssh, so the click stands in for ENTER.

    Skip and quit have no click to express them, so they are rpcs - the same split
    autotune's gate makes, for the same reason.
    """

    config: AlfredViewerGateConfig

    clicked_point: In[PointStamped]
    operator_command: Out[Int8]

    @rpc
    def start(self) -> None:
        super().start()
        self.clicked_point.subscribe(self._on_click)
        logger.warning(
            "Alfred benchmark is waiting at the gate. Click anywhere in the rerun viewer to "
            "run the next path, and drive the base with the viewer's keyboard between runs. "
            "app.AlfredViewerGate.skip() drops a path, .quit() ends the battery."
        )

    def _on_click(self, msg: PointStamped) -> None:
        self.operator_command.publish(Int8(data=GATE_ADVANCE))

    @rpc
    def advance(self) -> None:
        """Run the next path. The same thing a viewer click does."""
        self.operator_command.publish(Int8(data=GATE_ADVANCE))

    @rpc
    def skip(self) -> None:
        """Drop the next path - one the robot is badly placed for."""
        from dimos.control.benchmarking.gate import GATE_SKIP

        self.operator_command.publish(Int8(data=GATE_SKIP))

    @rpc
    def quit(self) -> None:
        """End the battery and keep what has been recorded."""
        from dimos.control.benchmarking.gate import GATE_QUIT

        self.operator_command.publish(Int8(data=GATE_QUIT))


def _base_tasks() -> list[TaskConfig]:
    """Teleop over the follower, the same order alfred-nav gives them."""
    return [
        # Priority 20: a held key preempts the follower for repositioning between
        # runs, by claim rather than by a timer.
        TaskConfig(
            name=BASE_VELOCITY_TASK_NAME,
            type="velocity",
            joint_names=list(_flowbase_hardware.joints),
            priority=20,
            params={"zero_on_timeout": False},
        ),
        TaskConfig(
            name=NAV_FOLLOWER_TASK_NAME,
            type="holonomic_pose_follower",
            joint_names=list(_flowbase_hardware.joints),
            priority=10,
            params={
                "speed": 0.4,
                "goal_tolerance": 0.20,
                "orientation_tolerance": 0.25,
                # Alfred's own if autotune has run, the Go2's otherwise. A bad
                # score on a quadruped's gains is not a controller problem.
                "artifact_path": alfred_follower_artifact(),
                # A sharp vertex makes the nearest point on the path flip between
                # the incoming and outgoing legs, and the reference yaw flips
                # with it - the robot sits on the corner oscillating. Monotonic
                # progress cannot flip back.
                "progress_back_m": 0.0,
                # And do not let the vertex's discretized dyaw/ds throttle the
                # approach to a standstill. 25% of vmax still slows hard for a
                # corner; it just arrives at one.
                "min_corner_speed": _CORNER_FLOOR,
            },
        ),
    ]


_ALFRED_BASE_TRANSPORTS: dict[Any, Any] = {
    # The base adapter owns two raw topics derived from its hardware_id. cmd_vel
    # is the coordinator's single arbitrated base command, and AlfredHighLevel is
    # what turns it into wheels; the Benchmarker records the same topic, so what
    # it scores is what the robot was actually told.
    ("cmd_vel", Twist): LCMTransport.spec(f"/{ALFRED_BASE_HARDWARE_ID}/cmd_vel", Twist),
    ("start_pose", PoseStamped): LCMTransport.spec(f"/{ALFRED_BASE_HARDWARE_ID}/odom", PoseStamped),
}


_alfred_base_control = autoconnect(
    # Headless: the viewer is the operator's console, opened from their machine.
    vis_module(viewer_backend=global_config.viewer, rerun_config={"rerun_open": "none"}),
    AlfredHighLevel.blueprint().remappings([(AlfredHighLevel, "wheel_odometry", "odom_sources")]),
    AlfredMountTf.blueprint(root_frame=LIDAR_FRAME),
    PointLio.blueprint(
        frame_id=ODOM_FRAME,
        sensor_frame_id=LIDAR_FRAME,
        lidar_ip=ALFRED.mid360_ip,
    ),
    StartRelay.blueprint(world_frame=ODOM_FRAME, base_frame="base_link"),
    AlfredBenchmarkCoordinator.blueprint(
        instance_name="ControlCoordinator",
        hardware=[_flowbase_hardware],
        tasks=_base_tasks(),
        # The viewer's keyboard arrives as tele_cmd_vel; the coordinator hears
        # twist_command and maps it onto the base's virtual joints, where
        # vel_flowbase claims it at priority 20. Without this the gate's
        # instruction to reposition between runs is a lie.
    ).remappings([(AlfredBenchmarkCoordinator, "twist_command", "tele_cmd_vel")]),
)


alfred_holonomic_controller = (
    _alfred_base_control.transports(dict(_ALFRED_BASE_TRANSPORTS))
    # Point-LIO is a C++ native and speaks LCM only.
    .global_config(n_workers=6, robot_model="alfred", transport="lcm")
)


alfred_benchmark = (
    autoconnect(
        _alfred_base_control,
        # "all" is the tangent-heading geometry plus the decoupled-yaw full-pose
        # cases. gate_source="stream" waits for the operator between every run,
        # which is what AlfredViewerGate feeds.
        Benchmarker.blueprint(
            robot="alfred",
            battery="all",
            gate_source="stream",
            # The Benchmarker's default ladder is the Go2's and tops out at
            # 1.0 m/s. Alfred's declared vmax is 0.5, so three of those five
            # speeds are outside its envelope and would score saturation rather
            # than tracking. Ladder to 90% of vmax instead.
            speeds=ALFRED_BENCHMARK_SPEEDS,
        ),
        AlfredViewerGate.blueprint(),
    )
    # The Benchmarker calls the robot's pose `odom`; StartRelay publishes it as
    # `start_pose`, which is also what the base adapter reads back.
    .remappings([(Benchmarker, "odom", "start_pose")])
    .transports(dict(_ALFRED_BASE_TRANSPORTS))
    .global_config(n_workers=8, robot_model="alfred", transport="lcm")
)
