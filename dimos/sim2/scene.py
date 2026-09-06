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

"""Compose native scene and robot MJCFs once, with named device attachments."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dimos.sim2.sensors.spec import Camera, Imu
from dimos.sim2.spec import WorldConfig
from dimos.utils.data import LfsPath


def quaternion(rpy: tuple[float, float, float]) -> NDArray[np.float64]:
    return np.asarray(Rotation.from_euler("xyz", rpy).as_quat(scalar_first=True), dtype=np.float64)


def load_scene(config: WorldConfig) -> mujoco.MjModel:
    world = mujoco.MjSpec.from_file(str(config.scene))
    world.option.timestep = config.timestep
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for robot_id, instance in config.robots.items():
        robot = mujoco.MjSpec.from_file(str(instance.config.model))
        robot.option.integrator = world.option.integrator
        if instance.config.meshdir is not None:
            robot.meshdir = str(instance.config.meshdir)
        root = robot.body(instance.config.root_body)
        if root is None:
            raise ValueError(f"{robot_id}: unknown robot root {instance.config.root_body!r}")
        # Root placement belongs to the instance, not to the source asset.
        root.pos = (0.0, 0.0, 0.0)
        root.quat = (1.0, 0.0, 0.0, 0.0)
        if not instance.config.floating:
            root.mocap = True
        for key in list(robot.keys):
            robot.delete(key)
        for sensor in instance.config.sensors:
            body = robot.body(sensor.mount.link)
            if body is None:
                raise ValueError(f"{robot_id}/{sensor.name}: unknown mount {sensor.mount.link!r}")
            quat = quaternion(sensor.mount.rpy)
            if isinstance(sensor, Camera):
                body.add_camera(
                    name="sensor/" + sensor.name,
                    pos=sensor.mount.xyz,
                    quat=quat,
                    fovy=sensor.fovy,
                )
            else:
                body.add_site(
                    name="sensor/" + sensor.name,
                    pos=sensor.mount.xyz,
                    quat=quat,
                    size=(0.001, 0.001, 0.001),
                    rgba=(0, 0, 0, 0),
                )
                if isinstance(sensor, Imu):
                    for suffix, kind in (
                        ("gyro", mujoco.mjtSensor.mjSENS_GYRO),
                        ("accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
                        ("quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
                    ):
                        robot.add_sensor(
                            name=f"sensor/{sensor.name}/{suffix}",
                            type=kind,
                            objtype=mujoco.mjtObj.mjOBJ_SITE,
                            objname="sensor/" + sensor.name,
                        )
        frame = world.worldbody.add_frame(pos=instance.xyz, quat=quaternion(instance.rpy))
        world.attach(robot, prefix=robot_id + "/", frame=frame)
    world.visual.global_.offwidth = max(
        [
            640,
            *[
                s.width
                for r in config.robots.values()
                for s in r.config.sensors
                if isinstance(s, Camera)
            ],
        ]
    )
    world.visual.global_.offheight = max(
        [
            480,
            *[
                s.height
                for r in config.robots.values()
                for s in r.config.sensors
                if isinstance(s, Camera)
            ],
        ]
    )
    return world.compile()


def scene_path(value: str | None, default: str) -> Path:
    if value is None or value == "none":
        return LfsPath("sim2/scenes") / default
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "scene.xml"
    if not path.is_file():
        raise FileNotFoundError(f"sim2 scene must be an MJCF file or scene directory: {path}")
    return path
