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

"""Native PICO / XRoboToolkit full-body teleoperation of SONIC G1.

    dimos --transport zenoh --simulation mujoco run unitree-g1-sonic-pico-teleop

Setup: dimos/teleop/pico/README.md. A+X toggles full-body POSE. The existing
SONIC task owns retargeting, planner fallback and hardware control lifecycle.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_sonic_wbc import (
    _g1_sonic_control_blueprint,
    _g1_sonic_visualization,
)
from dimos.teleop.pico.module import PicoTeleopModule
from dimos.teleop.pico.video import PicoVideoModule

unitree_g1_sonic_pico_teleop = autoconnect(
    PicoTeleopModule.blueprint(
        linear_scale=0.6, linear_min_speed=0.1, yaw_scale=1.5, deadzone=0.15
    ),
    *((PicoVideoModule.blueprint(),) if global_config.simulation == "mujoco" else ()),
    _g1_sonic_control_blueprint(
        task_type="g1_sonic_teleop",
        task_name="sonic_teleop",
        enable_sim_camera=True,
    ),
    _g1_sonic_visualization(),
).global_config(robot_model="unitree_g1", n_workers=3, listen_host="0.0.0.0")
