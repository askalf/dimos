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

"""Microduck CPU simulation with Pollen's pretrained walking policy."""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.microduck.simulation import MicroduckSim
from dimos.robot.unitree.keyboard_teleop import KeyboardTeleop

microduck_sim = MicroduckSim.blueprint().global_config(simulation="mujoco", n_workers=1)

microduck_sim_keyboard = autoconnect(
    microduck_sim,
    KeyboardTeleop.blueprint(
        linear_speed=0.3,
        boost_multiplier=1.5,
        publish_only_when_active=True,
    ),
).global_config(n_workers=2)
