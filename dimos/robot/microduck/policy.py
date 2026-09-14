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

"""CPU inference for Pollen's exported Microduck walking policies.

Observation order and position targets follow microduck_rl/scripts/infer_policy.py.
Joint order and the reference pose come from the ONNX export metadata.
"""

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

from dimos.simulation.engines.robot_sim_binding import RobotSimBinding, RobotSimSpec

PHYSICS_DT = 0.005
CONTROL_DT = 0.02
SPAWN_HEIGHT = 0.125


class MicroduckPolicy:
    def __init__(self, path: Path) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        metadata = self._session.get_modelmeta().custom_metadata_map
        self.joint_names = tuple(metadata["joint_names"].split(","))
        self.default_positions = np.array(
            metadata["default_joint_pos"].split(","), dtype=np.float32
        )
        self._action_scale = float(metadata["action_scale"])
        observation_names = metadata["observation_names"]
        if observation_names not in {
            "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command",
            "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command",
        }:
            raise ValueError(f"Unsupported Microduck observation layout: {observation_names}")
        input_info = self._session.get_inputs()[0]
        output_info = self._session.get_outputs()[0]
        self._input_name = input_info.name
        self._output_name = output_info.name
        dof = len(self.joint_names)
        if dof != 14 or len(set(self.joint_names)) != dof:
            raise ValueError("Microduck walking policies must describe 14 distinct joints")
        if self.default_positions.shape != (dof,) or not np.isfinite(self.default_positions).all():
            raise ValueError("Invalid Microduck reference joint positions")
        if not np.isfinite(self._action_scale) or self._action_scale <= 0:
            raise ValueError("Microduck action scale must be finite and positive")
        if input_info.shape not in ([1, 51], [1, 61]) or output_info.shape != [1, dof]:
            raise ValueError("Expected Microduck policy input [1, 51/61] and output [1, 14]")
        self._command_size = 3 if input_info.shape == [1, 51] else 13
        self._last_action = np.zeros(dof, dtype=np.float32)

    def sim_spec(self) -> RobotSimSpec:
        return RobotSimSpec(
            robot_id="microduck",
            hardware_joints=self.joint_names,
            model_joint_names=self.joint_names,
            model_actuator_names=self.joint_names,
            root_body_names=("trunk_base",),
            root_joint_names=("trunk_base_freejoint",),
            require_floating_base=True,
            imu_gyro_names=("imu_ang_vel",),
            imu_accel_names=("imu_accel",),
            require_imu=True,
        )

    def reset(self) -> None:
        self._last_action.fill(0)

    def targets(
        self,
        data: mujoco.MjData,
        binding: RobotSimBinding,
        velocity: tuple[float, float, float],
    ) -> NDArray[np.float32]:
        """Read one physics snapshot and return joint targets in policy order."""
        assert binding.root_body_id is not None
        assert binding.imu_gyro_slice is not None
        command = np.zeros(self._command_size, dtype=np.float32)
        command[:3] = velocity
        gravity = -data.xmat[binding.root_body_id].reshape(3, 3)[2, :]
        obs = np.concatenate(
            (
                data.sensordata[binding.imu_gyro_slice],
                gravity,
                data.qpos[list(binding.joint_qpos_adrs)] - self.default_positions,
                data.qvel[list(binding.joint_qvel_adrs)],
                self._last_action,
                command,
            )
        ).astype(np.float32)
        action = np.asarray(
            self._session.run([self._output_name], {self._input_name: obs[None, :]})[0][0],
            dtype=np.float32,
        )
        if action.shape != self._last_action.shape or not np.isfinite(action).all():
            raise ValueError("Microduck policy returned invalid joint actions")
        self._last_action = action.copy()
        return self.default_positions + self._action_scale * action
