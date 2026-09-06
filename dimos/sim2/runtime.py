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

"""Single mutable MuJoCo world; control and snapshots never wait for rendering."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dimos.sim2.control.interface import descriptor
from dimos.sim2.ipc.abi import (
    ABI_VERSION,
    ChannelDescriptor,
    FrameField,
    FrameLayout,
)
from dimos.sim2.ipc.channel import FrameMetadata, RobotChannel
from dimos.sim2.scene import load_scene, quaternion
from dimos.sim2.sensors.spec import Imu
from dimos.sim2.spec import ControlInterface, RobotConfig, WorldConfig

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass
class RobotBinding:
    config: RobotConfig
    channel: RobotChannel
    qpos: NDArray[np.int32]
    dofs: NDArray[np.int32]
    actuators: NDArray[np.int64]
    root: int
    scale: NDArray[np.float64]
    offset: NDArray[np.float64]
    imu: tuple[int, int, int] | None
    enabled: bool = True


class SimulationRuntime:
    def __init__(self, config: WorldConfig, sim_id: str) -> None:
        self.config = config
        self.model = load_scene(config)
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.episode = 0
        self.tick = 0
        self._closed = False
        self.robots: dict[str, RobotBinding] = {}
        nstate = mujoco.mj_stateSize(self.model, STATE)
        self.state = np.empty(nstate)
        self.snapshot_descriptor = ChannelDescriptor(
            abi_version=ABI_VERSION,
            sim_id=sim_id,
            robot_id="world",
            generation=sim_id,
            shm_name=descriptor(sim_id + "/world", ControlInterface.WHOLE_BODY, 1).shm_name,
            control_interface=ControlInterface.WHOLE_BODY,
            dof=0,
            capabilities=(),
            physics_dt=config.timestep,
            control_decimation=1,
            action_layout=FrameLayout((), 64),
            observation_layout=FrameLayout(
                (
                    FrameField("state", "<f8", (nstate,), 48),
                    FrameField("wall_time", "<f8", (1,), 48 + nstate * 8),
                ),
                ((56 + nstate * 8 + 63) // 64) * 64,
            ),
        )
        self.snapshots = RobotChannel.create(self.snapshot_descriptor)
        try:
            for robot_id, instance in config.robots.items():
                definition = instance.config
                joints = [
                    self.model.joint(robot_id + "/" + j.model_name).id for j in definition.joints
                ]
                actuators = np.array(
                    [self.model.actuator(robot_id + "/" + j.actuator).id for j in definition.joints]
                )
                channel = RobotChannel.create(
                    descriptor(sim_id + "/" + robot_id, definition.control, len(joints))
                )
                imus = [s for s in definition.sensors if isinstance(s, Imu)]
                imu = None
                if imus:
                    g, a, q = (
                        int(self.model.sensor(f"{robot_id}/sensor/{imus[0].name}/{s}").adr[0])
                        for s in ("gyro", "accel", "quat")
                    )
                    imu = (g, a, q)
                self.robots[robot_id] = RobotBinding(
                    definition,
                    channel,
                    self.model.jnt_qposadr[joints],
                    self.model.jnt_dofadr[joints],
                    actuators,
                    self.model.body(robot_id + "/" + definition.root_body).id,
                    np.array([j.scale for j in definition.joints]),
                    np.array([j.offset for j in definition.joints]),
                    imu,
                )
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        with self.lock:
            self.episode += 1
            mujoco.mj_resetData(self.model, self.data)
            for binding in self.robots.values():
                home = np.array([j.home for j in binding.config.joints])
                self.data.qpos[binding.qpos] = home * binding.scale + binding.offset
                for joint, aid in zip(binding.config.joints, binding.actuators, strict=True):
                    self.data.ctrl[aid] = (
                        joint.home * joint.ctrl_scale + joint.ctrl_offset
                        if joint.mode == "position"
                        else 0.0
                    )
                binding.channel.set_episode(self.episode)
                binding.enabled = True
            mujoco.mj_forward(self.model, self.data)
            self._publish(force_snapshot=True)
            for binding in self.robots.values():
                binding.channel.set_lifecycle("ready")
            self.snapshots.set_lifecycle("ready")

    def set_spawn(
        self, robot_id: str, xyz: tuple[float, float, float], rpy: tuple[float, float, float]
    ) -> None:
        with self.lock:
            binding = self.robots[robot_id]
            if binding.config.floating:
                jid = self.model.body_jntadr[binding.root]
                q = self.model.jnt_qposadr[jid]
                v = self.model.jnt_dofadr[jid]
                self.data.qpos[q : q + 3] = xyz
                self.data.qpos[q + 3 : q + 7] = quaternion(rpy)
                self.data.qvel[v : v + 6] = 0
            else:
                mid = self.model.body_mocapid[binding.root]
                self.data.mocap_pos[mid] = xyz
                self.data.mocap_quat[mid] = quaternion(rpy)
            mujoco.mj_forward(self.model, self.data)
            self.episode += 1
            for robot in self.robots.values():
                robot.channel.set_episode(self.episode)
            self._publish(force_snapshot=True)

    def step(self) -> None:
        with self.lock:
            for binding in self.robots.values():
                self._apply(binding)
            mujoco.mj_step(self.model, self.data)
            self.tick += 1
            self._publish()

    def _apply(self, b: RobotBinding) -> None:
        action = b.channel.read_action()
        q = self.data.qpos[b.qpos]
        dq = self.data.qvel[b.dofs]
        if action is not None and action.metadata.episode_id == self.episode:
            values = action.values
            enabled = bool(values["enabled"][0])
            target = values["position"] * b.scale + b.offset
            velocity = values["velocity"] * b.scale
            if b.config.control == ControlInterface.WHOLE_BODY:
                kp, kd = values["kp"], values["kd"]
                ff = values["effort"]
            else:
                kp = np.array([j.kp for j in b.config.joints])
                kd = np.array([j.kd for j in b.config.joints])
                ff = values["effort"]
        else:
            enabled = True
            target = np.array([j.home for j in b.config.joints]) * b.scale + b.offset
            velocity = np.zeros_like(dq)
            kp = np.array([j.kp for j in b.config.joints])
            kd = np.array([j.kd for j in b.config.joints])
            ff = np.zeros_like(dq)
        torque = kp * (target - q) + kd * (velocity - dq) + ff
        b.enabled = enabled
        for i, (joint, aid) in enumerate(zip(b.config.joints, b.actuators, strict=True)):
            if joint.mode == "position":
                public_target = (
                    (target[i] - b.offset[i]) / b.scale[i]
                    if enabled
                    else (q[i] - b.offset[i]) / b.scale[i]
                )
                value = public_target * joint.ctrl_scale + joint.ctrl_offset
            else:
                value = torque[i] if enabled else 0.0
            if self.model.actuator_ctrllimited[aid]:
                value = np.clip(value, *self.model.actuator_ctrlrange[aid])
            self.data.ctrl[aid] = value

    def _publish(self, *, force_snapshot: bool = False) -> None:
        meta = FrameMetadata(0, self.episode, self.tick, 0, float(self.data.time))
        timestamp = time.time()
        for b in self.robots.values():
            values: dict[str, Any] = {
                "position": (self.data.qpos[b.qpos] - b.offset) / b.scale,
                "velocity": self.data.qvel[b.dofs] / b.scale,
                "effort": self.data.qfrc_actuator[b.dofs],
                "enabled": [int(b.enabled)],
                "wall_time": [timestamp],
            }
            if b.config.control == ControlInterface.WHOLE_BODY:
                assert b.imu is not None, "whole-body configuration requires an IMU"
                g, a, r = b.imu
                gyro, accel, quat = (
                    self.data.sensordata[g : g + 3],
                    self.data.sensordata[a : a + 3],
                    self.data.sensordata[r : r + 4],
                )
                values.update(
                    imu_quaternion=quat,
                    imu_gyroscope=gyro,
                    imu_accelerometer=accel,
                    imu_rpy=Rotation.from_quat(quat, scalar_first=True).as_euler("xyz"),
                    root_position=self.data.xpos[b.root],
                    root_quaternion=self.data.xquat[b.root],
                    root_linear_velocity=self.data.cvel[b.root, 3:],
                    root_angular_velocity=self.data.cvel[b.root, :3],
                )
            else:
                values.update(gripper=[0.0], error_code=[0])
            b.channel.publish_observation(values, meta)
        stride = max(1, round(1.0 / self.config.snapshot_hz / self.config.timestep))
        if force_snapshot or self.tick % stride == 0:
            mujoco.mj_getState(self.model, self.data, self.state, STATE)
            self.snapshots.publish_observation(
                {"state": self.state, "wall_time": [timestamp]}, meta
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for channel in [*(b.channel for b in self.robots.values()), self.snapshots]:
            channel.set_lifecycle("closed")
            channel.unlink()
            channel.close()
