# Copyright 2025-2026 Dimensional Inc.
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

from contextlib import ExitStack
import functools
import threading
from typing import Any

from reactivex import Observable, Subject

from dimos.core.global_config import GlobalConfig
from dimos.core.transport import LCMTransport, PubSubTransport
from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.simulation.dimsim.dimsim_process import DimSimProcess
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_WIDTH = 640
_HEIGHT = 288
_FOV_DEG = 46


class DimSimConnection:
    """DimSim speaks LCM on the wire, independent of the module transport.

    LCM runs consume simulator topics directly. Other backends receive a
    one-way relay of those same topics, preserving their world-frame data.
    The observables remain silent so GO2Connection does not publish them twice.
    """

    camera_info_static: CameraInfo = CameraInfo.from_fov(
        fov_deg=_FOV_DEG,
        width=_WIDTH,
        height=_HEIGHT,
        axis="horizontal",
        frame_id="camera_optical",
    )

    def __init__(self, global_config: GlobalConfig) -> None:
        self._dimsim_process: DimSimProcess = DimSimProcess(global_config)
        self._odom_transport: PubSubTransport[PoseStamped] = LCMTransport("/odom", PoseStamped)
        self._tf_transport: PubSubTransport[TFMessage] = make_transport(
            "/tf", TFMessage, g=global_config
        )
        self._odom_output: PubSubTransport[PoseStamped] | None = None
        self._cmd_transport: PubSubTransport[Twist] | None = None
        self._relays: list[tuple[PubSubTransport[Any], PubSubTransport[Any]]] = []
        if global_config.transport != "lcm":
            self._odom_output = make_transport("/odom", PoseStamped, g=global_config)
            self._cmd_transport = LCMTransport("/cmd_vel", Twist)
            for topic, message_type in (
                ("/color_image", Image),
                ("/lidar", PointCloud2),
                ("/camera_info", CameraInfo),
            ):
                self._relays.append(
                    (
                        LCMTransport(topic, message_type),
                        make_transport(topic, message_type, g=global_config),
                    )
                )
        self._resources: ExitStack | None = None
        self._command_lock = threading.Lock()
        self._command_ready = False

    def start(self) -> None:
        if self._resources is not None:
            return
        resources = ExitStack()
        self._resources = resources
        try:
            # Register cleanup before startup so partial failures roll back too.
            resources.callback(self._dimsim_process.stop)
            transports: list[PubSubTransport[Any]] = [self._tf_transport, self._odom_transport]
            if self._odom_output is not None:
                transports.append(self._odom_output)
            if self._cmd_transport is not None:
                transports.append(self._cmd_transport)
            for source, target in self._relays:
                transports.extend((target, source))
            for transport in transports:
                resources.callback(transport.stop)
                transport.start()
            resources.callback(self._odom_transport.subscribe(self._handle_odom))
            for source, target in self._relays:
                resources.callback(source.subscribe(target.publish))
            self._dimsim_process.start()
            with self._command_lock:
                self._command_ready = True
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        with self._command_lock:
            self._command_ready = False
        resources, self._resources = self._resources, None
        if resources is not None:
            resources.close()

    @functools.cache
    def lidar_stream(self) -> Observable[PointCloud2]:
        return Subject()

    @functools.cache
    def odom_stream(self) -> Observable[PoseStamped]:
        return Subject()

    @functools.cache
    def video_stream(self) -> Observable[Image]:
        return Subject()

    @functools.cache
    def lowstate_stream(self) -> Observable[Any]:
        return Subject()

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        # In LCM mode the simulator already receives the original /cmd_vel.
        # Republishing it here would feed GO2Connection's subscriber forever.
        if self._cmd_transport is not None:
            with self._command_lock:
                if not self._command_ready:
                    return False
                self._cmd_transport.publish(twist)
        return True

    def standup(self) -> bool:
        return True

    def liedown(self) -> bool:
        return True

    def balance_stand(self) -> bool:
        return True

    def sport_command(self, api_id: int) -> bool:
        return True

    def stop_movement(self) -> None:
        if self._cmd_transport is not None:
            self.move(Twist())

    def set_obstacle_avoidance(self, enabled: bool = True) -> bool:
        return True

    def set_rage_mode(self, enable: bool) -> bool:
        return True

    def set_light(self, level: int) -> bool:
        return True

    def switch_joystick(self, enable: bool = True) -> bool:
        return True

    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        return {}

    def _handle_odom(self, msg: PoseStamped) -> None:
        if self._odom_output is not None:
            self._odom_output.publish(msg)
        self._tf_transport.publish(TFMessage(*_odom_to_tf(msg)))


def _odom_to_tf(odom: PoseStamped) -> list[Transform]:
    """Build transform chain from odometry pose.

    Transform tree: world -> base_link -> {camera_link -> camera_optical, lidar_link}
    """
    camera_link = Transform(
        translation=Vector3(0.3, 0.0, 0.0),  # camera 30cm forward
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        frame_id="base_link",
        child_frame_id="camera_link",
        ts=odom.ts,
    )

    camera_optical = Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        frame_id="camera_link",
        child_frame_id="camera_optical",
        ts=odom.ts,
    )

    lidar_link = Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        frame_id="base_link",
        child_frame_id="lidar_link",
        ts=odom.ts,
    )

    return [
        Transform.from_pose("base_link", odom),
        camera_link,
        camera_optical,
        lidar_link,
    ]
