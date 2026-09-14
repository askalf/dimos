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

"""Microduck locomotion in the shared DimOS MuJoCo engine."""

from functools import partial
import math
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.robot.assets.source import RobotDescriptionSource
from dimos.robot.microduck.policy import CONTROL_DT, PHYSICS_DT, SPAWN_HEIGHT, MicroduckPolicy
from dimos.simulation.engines.mujoco_engine import MujocoEngine


class MicroduckSimConfig(ModuleConfig):
    scene_path: Path | None = None
    policy_path: Path | None = None
    headless: bool = False
    command_timeout: float = Field(default=0.5, gt=0, allow_inf_nan=False)


class MicroduckSim(Module):
    """Simulated Microduck with cmd_vel, joint state, odometry and reset RPCs."""

    config: MicroduckSimConfig
    cmd_vel: In[Twist]
    joint_state: Out[JointState]
    odom: Out[PoseStamped]
    tf: Out[TFMessage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._engine: MujocoEngine | None = None
        self._state_lock = threading.Lock()
        self._velocity = (0.0, 0.0, 0.0)
        self._command_deadline = 0.0
        self._ready = threading.Event()
        self._state: dict[str, Any] = {}
        # Only accessed by the engine's simulation thread.
        self._next_control_time = 0.0
        self._previous_sim_time = -1.0

    @rpc
    def start(self) -> None:
        super().start()
        scene_path = self.config.scene_path
        if scene_path is None:
            source = RobotDescriptionSource(
                "https://github.com/pollen-robotics/microduck_rl.git",
                "53b8971b61baf5b7f3c16d135dd7cac37623de4b",
            )
            scene_path = source / "src/mjlab_microduck/robot/microduck/scene.xml"
        policy_path = self.config.policy_path
        if policy_path is None:
            policies = RobotDescriptionSource(
                "https://huggingface.co/pollen-robotics/microduck-policies", "088524a"
            )
            policy_path = policies / "alpha_walking.onnx"

        policy = MicroduckPolicy(policy_path)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        model.opt.timestep = PHYSICS_DT
        model.stat.extent = 0.6
        model.stat.center[:] = (0, 0, SPAWN_HEIGHT)
        engine = MujocoEngine(
            config_path=scene_path,
            headless=self.config.headless,
            model=model,
            robot_sim_spec=policy.sim_spec(),
            spawn_z=SPAWN_HEIGHT,
            reset_joint_positions=policy.default_positions.tolist(),
            on_before_step=partial(self._control, policy),
            on_after_step=self._publish_state,
        )
        binding = engine.robot_binding
        assert binding is not None
        # Validate inference before the engine starts its background loop.
        policy.targets(engine.data, binding, (0.0, 0.0, 0.0))
        policy.reset()
        with self._state_lock:
            self._engine = engine
        self._ready.clear()
        self._next_control_time = 0.0
        self._previous_sim_time = -1.0
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self.move)))
        if not engine.connect() or not self._ready.wait(timeout=10.0):
            self.stop()
            raise RuntimeError("Microduck simulation did not produce its first physics state")

    @rpc
    def stop(self) -> None:
        with self._state_lock:
            engine = self._engine
            self._engine = None
            self._velocity = (0.0, 0.0, 0.0)
            self._command_deadline = 0.0
        try:
            if engine is not None:
                engine.disconnect()
        finally:
            super().stop()

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> None:
        """Command simulated body velocity; expire after duration or command_timeout."""
        velocity = (float(twist.linear.x), float(twist.linear.y), float(twist.angular.z))
        if not all(math.isfinite(value) for value in (*velocity, duration)) or duration < 0:
            raise ValueError("Velocity and duration must be finite; duration must be nonnegative")
        with self._state_lock:
            self._velocity = velocity
            self._command_deadline = time.monotonic() + (duration or self.config.command_timeout)

    @rpc
    def reset(self) -> bool:
        """Return the simulated duck to its initial standing pose and clear motion."""
        with self._state_lock:
            engine = self._engine
            self._velocity = (0.0, 0.0, 0.0)
            self._command_deadline = 0.0
        if engine is None:
            return False
        return engine.request_reset(wait=True, timeout=2.0)

    @rpc
    def get_state(self) -> dict[str, Any]:
        """Return the latest physics time, root pose and commanded velocity."""
        with self._state_lock:
            return dict(self._state)

    def _control(self, policy: MicroduckPolicy, engine: MujocoEngine) -> None:
        sim_time = float(engine.data.time)
        if sim_time < self._previous_sim_time:
            policy.reset()
            self._next_control_time = 0.0
        self._previous_sim_time = sim_time
        if sim_time + PHYSICS_DT / 2 < self._next_control_time:
            return
        self._next_control_time = sim_time + CONTROL_DT
        with self._state_lock:
            velocity = self._velocity if time.monotonic() < self._command_deadline else (0, 0, 0)
        binding = engine.robot_binding
        assert binding is not None
        targets = policy.targets(engine.data, binding, velocity)
        engine.write_joint_command(JointState(position=targets.tolist()))

    def _publish_state(self, engine: MujocoEngine) -> None:
        # Publish at policy frequency, not every 200 Hz physics substep.
        if float(engine.data.time) + PHYSICS_DT / 2 < self._next_control_time:
            return
        root_pose = engine.get_root_pose()
        assert root_pose is not None
        position, xyzw = root_pose
        ts = time.time()
        pose = PoseStamped(
            ts=ts,
            frame_id="world",
            position=Vector3(*position),
            orientation=Quaternion(*xyzw),
        )
        self.odom.publish(pose)
        self.joint_state.publish(
            JointState(
                ts=ts,
                frame_id="base_link",
                name=engine.joint_names,
                position=engine.joint_positions,
                velocity=engine.joint_velocities,
                effort=engine.joint_efforts,
            )
        )
        self.tf.publish(TFMessage(Transform.from_pose("base_link", pose)))
        with self._state_lock:
            self._state = {
                "sim_time": float(engine.data.time),
                "position": tuple(float(x) for x in position),
                "orientation_wxyz": tuple(float(x) for x in xyzw[[3, 0, 1, 2]]),
                "command": self._velocity
                if time.monotonic() < self._command_deadline
                else (0, 0, 0),
            }
        self._ready.set()
