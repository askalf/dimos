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

"""Launch and lifecycle contract used by Sim's Habitat backend.

Implementation seam: Sim keeps process ownership, MCP connection, recording,
agent-module composition, and cleanup. A backend supplies launch configuration,
observation readiness, and a normalized pose for settling. Backend selection
must not silently replace a suite's scene while retaining its reference answers.

DimSim retains its existing launch/setup path. Habitat starts a fresh process
per case. Scene configuration is supplied by suites, not global CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dimos.memory.store.base import Store
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped


@dataclass(frozen=True)
class SimLaunchSpec:
    """Arguments to DimosCliCall, without starting a process.

    Sim appends agent modules and disabled-module flags to blueprint names and
    owns --record. Global arguments precede `run`; module arguments follow the
    composed blueprint names. Environment pairs become proc.extra_env.
    `simulation_flag=None` omits --simulation for standalone Habitat blueprints.
    """

    blueprint: tuple[str, ...]
    simulation_flag: str | None
    global_args: tuple[str, ...] = ()
    module_args: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()


class SimBackendSpec(Protocol):
    """Backend hooks called by Sim under its existing launch deadline.

    Acceptance: existing DimSim cases retain launch behavior; Habitat cases
    launch without --dimsim-scene/--simulation habitat; failed readiness still
    tears down the entire owned process group. Skill selection and navigation
    behavior belong to the caller's dimos composition, not the simulator backend.
    """

    def preflight(self) -> None:
        """Validate scene/config inputs and prerequisites before starting an agent."""
        ...

    def launch_spec(self) -> SimLaunchSpec:
        """Provide simulator launch settings without imposing agent tools."""
        ...

    def wait_ready(self, recording: Store, *, deadline: float) -> None:
        """Require fresh RGB and pose; Sim owns MCP endpoint readiness.

        deadline uses time.monotonic(). Propagate startup failure/timeout;
        the owning Sim environment performs cleanup.
        """
        ...

    def latest_pose(self, recording: Store) -> PoseStamped:
        """Normalize backend odometry for Sim.settle; raise LookupError if absent."""
        ...

    def episode_metadata(self) -> dict[str, object]:
        """Return JSON-serializable scene, version, seed, spawn and sensor settings.

        Sim persists this beside the recording as a grader artifact. References
        and privileged semantic data must not enter the agent tool context.
        """
        ...
