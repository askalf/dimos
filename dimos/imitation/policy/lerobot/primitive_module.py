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

"""Native policy bindings with disjoint left/right arm coordinator resources."""

from dimos.imitation.policy.lerobot.module import LeRobotPolicyConfig
from dimos.imitation.policy.module import declare_policy_module
from dimos.robot.galaxea.r1pro.object_primitives import primitive_profile

R1ProRightPickPolicy = declare_policy_module(
    "R1ProRightPickPolicy",
    __name__,
    primitive_profile("pick", "right"),
    LeRobotPolicyConfig,
    "dimos_lerobot.primitive_runtime:R1ProRightPickPolicyRuntime",
)

R1ProRightPlacePolicy = declare_policy_module(
    "R1ProRightPlacePolicy",
    __name__,
    primitive_profile("place", "right"),
    LeRobotPolicyConfig,
    "dimos_lerobot.primitive_runtime:R1ProRightPlacePolicyRuntime",
)

R1ProLeftPickPolicy = declare_policy_module(
    "R1ProLeftPickPolicy",
    __name__,
    primitive_profile("pick", "left"),
    LeRobotPolicyConfig,
    "dimos_lerobot.primitive_runtime:R1ProLeftPickPolicyRuntime",
)

R1ProLeftPlacePolicy = declare_policy_module(
    "R1ProLeftPlacePolicy",
    __name__,
    primitive_profile("place", "left"),
    LeRobotPolicyConfig,
    "dimos_lerobot.primitive_runtime:R1ProLeftPlacePolicyRuntime",
)
