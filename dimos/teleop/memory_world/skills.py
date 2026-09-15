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

"""The memory world's own skills: what an agent can ask it to do.

Lifted out of `module.py` because that file is at the repository's 75 KB per-file limit
and these two are the most self-contained thing in it -- the same reason `answers.py`,
`visual_answers.py`, `world_cache.py` and `replay_serving.py` exist. Mixed into
`MemoryWorldModule`, so `self` is the module and every caller is unaffected.

Note what is NOT here: on the hyperspace blueprint the SEARCH skills belong to the
Hyperspace module (`start_item_query` and its two siblings), and the agent calls those
directly. `find_in_memory` is the siglip path's own search, and `navigate_to_place`
routes to whichever engine's answer is currently on screen.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from dimos.agents.annotation import skill
from dimos.agents.skill_result import SkillResult
from dimos.teleop.memory_world.answers import NavigateRequest
from dimos.teleop.memory_world.visual_search import search_phrase
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class MemoryWorldSkills:
    """The two things an agent can ask the memory world itself to do."""

    config: Any

    if TYPE_CHECKING:

        def _find_with_siglip(self, phrase: str, started: float, span: Any) -> SkillResult: ...
        def _navigate_to(self, request: NavigateRequest) -> dict[str, Any]: ...

    @skill
    def find_in_memory(
        self,
        query: str,
        from_fraction: float = 0.0,
        to_fraction: float = 1.0,
    ) -> SkillResult:
        """Find where something was seen in the recording and highlight it in the world:
        "where did I see a car", answered by a vector-database lookup over the recording's
        CLIP/SigLIP image embeddings. Each result is a place the thing was seen FROM,
        with the photograph that matched.

        Args:
            query: What to look for, e.g. "a car" or "a whiteboard". Pass the thing, not
                the whole sentence.
            from_fraction: Where in the recording to start looking. 0.0 is the very
                beginning, 1.0 the very end. Use 0.0 and 0.5 for "the first half".
            to_fraction: Where to stop looking, on the same 0.0-1.0 scale.
        """
        started = time.monotonic()
        phrase = search_phrase(query)
        if not phrase:
            return SkillResult.fail("INVALID_QUERY", "The query text is empty")
        # Clamped and ordered rather than refused: an LLM that says (0.5, 0.0) means the
        # second half, and failing the whole question over the argument order teaches it
        # nothing it can act on.
        low, high = sorted((float(from_fraction), float(to_fraction)))
        low, high = max(0.0, min(1.0, low)), max(0.0, min(1.0, high))
        if high <= low:
            return SkillResult.fail(
                "INVALID_QUERY",
                f"{from_fraction} to {to_fraction} is not a stretch of the recording",
            )
        span = None if (low, high) == (0.0, 1.0) else (low, high)

        try:
            return self._find_with_siglip(phrase, started, span=span)
        except Exception as error:  # an index built for another model, camera or frame
            logger.exception("visual index query failed")
            return SkillResult.fail("QUERY_FAILED", f"The SigLIP index cannot answer: {error}")

    @skill
    def navigate_to_place(self, place: int = 1, start: str = "recording start") -> SkillResult:
        """Draw a walking route to one of the places the last `find_in_memory` answer
        found, and show it in the world.

        Args:
            place: Which place to walk to. 1 is the first one the answer listed.
            start: Where to walk FROM. "recording start" is where the robot was when the
                recording began -- what someone means by "the starting point". "viewer"
                is where the person asking is standing right now.
        """
        # "Where the person is" wins over "start", because one sentence can hold both:
        # "start from where I am standing now" contains the word `start` and means the
        # opposite of the recording's beginning.
        said = start.lower()
        here = any(word in said for word in ("view", "stand", "here", "current", "now", " me"))
        begins = any(word in said for word in ("record", "start", "begin", "first"))
        wanted = "recording_start" if begins and not here else "viewer"
        try:
            payload = self._navigate_to(
                NavigateRequest(cluster=max(0, int(place) - 1), start_at=wanted)
            )
        except HTTPException as refused:
            return SkillResult.fail("NO_ROUTE", str(refused.detail))
        except Exception as error:
            logger.exception("navigation failed")
            return SkillResult.fail("NO_ROUTE", f"Could not plan a route: {error}")
        return SkillResult(
            success=True,
            message=(
                f"Drew a {payload['length_m']} m route to place #{payload['cluster'] + 1}, "
                f"starting from "
                + (
                    "where the robot was at the start of the recording"
                    if wanted == "recording_start"
                    else "where you are standing"
                )
            ),
            metadata=payload,
        )
