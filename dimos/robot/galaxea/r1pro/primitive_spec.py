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

"""RPC contract shared by the native primitive simulator and its clients."""

from typing import Any, Protocol

from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.spec.utils import Spec


class PrimitiveSimSpec(Spec, Protocol):
    def prepare_primitive_session(self) -> dict[str, Any]: ...
    def prepare_primitive(
        self, primitive: str, arm: str, index: int = -1, region: str = "tray"
    ) -> dict[str, Any]: ...
    def define_placement_region(
        self, name: str, x: float, y: float, width: float, depth: float
    ) -> dict[str, Any]: ...
    def primitive_state(self) -> dict[str, Any]: ...
    def stop_primitive_base(self) -> None: ...
    def validate_primitive_base_plan(self, trajectory: JointTrajectory) -> None: ...
    def is_simulation_running(self) -> bool: ...
    def primitive_recovery(self) -> dict[str, Any]: ...
    def validate_primitive_recovery_plan(self, trajectory: JointTrajectory) -> None: ...
    def finish_primitive_recovery(self) -> dict[str, Any]: ...
    def reset(self) -> bool: ...
