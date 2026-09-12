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

"""Standard ACT rollout runtimes for independently trained arm primitives."""

from dimos_lerobot.runtime import LeRobotBackend

from dimos.imitation.policy.runtime import declare_policy_runtime
from dimos.robot.galaxea.r1pro.primitive_policies import (
    R1ProLeftPickPolicy,
    R1ProLeftPlacePolicy,
    R1ProRightPickPolicy,
    R1ProRightPlacePolicy,
)

R1ProRightPickPolicyRuntime = declare_policy_runtime(
    "R1ProRightPickPolicyRuntime",
    __name__,
    R1ProRightPickPolicy,
    LeRobotBackend,
)

R1ProRightPlacePolicyRuntime = declare_policy_runtime(
    "R1ProRightPlacePolicyRuntime",
    __name__,
    R1ProRightPlacePolicy,
    LeRobotBackend,
)

R1ProLeftPickPolicyRuntime = declare_policy_runtime(
    "R1ProLeftPickPolicyRuntime",
    __name__,
    R1ProLeftPickPolicy,
    LeRobotBackend,
)

R1ProLeftPlacePolicyRuntime = declare_policy_runtime(
    "R1ProLeftPlacePolicyRuntime",
    __name__,
    R1ProLeftPlacePolicy,
    LeRobotBackend,
)
