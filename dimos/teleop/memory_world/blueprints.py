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

"""Agentic recorded-memory world blueprints."""

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.coordination.blueprints import autoconnect
from dimos.teleop.memory_world.module import MemoryWorldModule

MEMORY_WORLD_SYSTEM_PROMPT = """You answer questions about a recorded robot memory,
and the person asking is standing inside it in a headset.

Use the `find_in_memory` tool for every question about what the robot saw. It searches
the recording's own CLIP/SigLIP frame embeddings, lights the places it found in the
world, and hangs the photograph behind each one where the camera stood. Pass it the
thing to look for, not a sentence: "a fire extinguisher", not "where did I see a fire
extinguisher".

The tool's reply tells you how many places it found and how confident it is in each.
Report what it actually found. It can fail two ways that mean different things:
NOT_FOUND means the recording has no such thing, and INDEX_NOT_READY means the
recording has not been embedded yet -- say so rather than reporting nothing was there.

Do not claim anything was shown in the world unless `find_in_memory` succeeded.
"""

memory_world_agent = autoconnect(
    MemoryWorldModule.blueprint(),
    McpServer.blueprint(),
    McpClient.blueprint(system_prompt=MEMORY_WORLD_SYSTEM_PROMPT),
).global_config(n_workers=4)
