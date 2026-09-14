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

"""Open-space classical demo: full viewer, Zenoh and optional HumanCLI agent."""

from dataclasses import replace

from dimos.agents.mcp.mcp_client import McpClient
from dimos.core.coordination.blueprints import Blueprint
from dimos.robot.galaxea.r1pro.classical_blueprint import (
    CLASSICAL_PROMPT,
    build_classical_apartment,
)
from dimos.robot.galaxea.r1pro.open_space_scene import OPEN_PLATFORMS
from dimos.robot.galaxea.r1pro.open_space_sim import R1ProOpenSpaceSim


def build_classical_open_space(*, agent: bool = False) -> Blueprint:
    source = build_classical_apartment(agent=agent, simulator=R1ProOpenSpaceSim)
    prompt = CLASSICAL_PROMPT.replace("in an apartment", "in an open platform arena") + (
        "\nThe named platforms are "
        + ", ".join(f"{p.name} ({p.height * 100:.0f} cm high)" for p in OPEN_PLATFORMS)
        + ". There is also a tray on the worktable. Use these exact names for go_to and place_object. "
        "There is no kitchen or dining table in this scene. Read get_scene for current object positions."
    )
    return replace(
        source,
        blueprints=tuple(
            replace(
                atom,
                kwargs={
                    **atom.kwargs,
                    **(
                        dict(
                            viewer_lookat=(0.2, 0.5, 0.4),
                            viewer_distance=13.5,
                            viewer_azimuth=135,
                            viewer_elevation=-50,
                        )
                        if atom.module is R1ProOpenSpaceSim
                        else dict(system_prompt=prompt)
                        if atom.module is McpClient
                        else {}
                    ),
                },
            )
            for atom in source.blueprints
        ),
    )


r1pro_classical_open_space_sim = build_classical_open_space().global_config()
r1pro_classical_open_space_sim_agent = build_classical_open_space(agent=True).global_config()
