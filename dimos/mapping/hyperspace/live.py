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

"""The thing that answers a question, with everything it needs already loaded.

This is the live query, and there is exactly one of it. The ``Hyperspace`` module holds
one and its RPC delegates to it; ``dimos map live`` holds one over a recording on disk
and asks it the same way. That is deliberate: an offline tool that reimplemented the
query would drift from the robot's, and then the pages it draws would be a picture of
something the robot does not do.

What "live" means here, concretely:

* The models are resident. OWLv2 is two and a half seconds to bring up and the text
  towers thirteen; paid once, at construction, so a question costs a question.
* The index is not finished. It starts at whatever the store holds -- which on a robot
  at power-on is nothing -- and every question takes in the patches written since the
  last one. A map that stops growing is a recording; this does not stop.
* The clock runs from the question. Every answer carries how long after the asking it
  arrived, because that is the number a person waiting actually experiences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import TYPE_CHECKING, Any

from dimos.mapping.hyperspace.detect import (
    DetectConfig,
    Owlv2Boxes,
    RecordingFrames,
    find,
    merge_duplicates,
)
from dimos.mapping.hyperspace.frames import TextTowers, member_streams
from dimos.mapping.hyperspace.msgs import FoundObject, FoundObjects
from dimos.mapping.hyperspace.resident import ResidentIndex
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dimos.mapping.hyperspace.detect import Detection

logger = setup_logger()


@dataclass
class LiveConfig:
    """What the query is allowed to do, and how sure the detector has to be."""

    detect: DetectConfig = field(default_factory=DetectConfig)
    # Models to search. Empty = every model the store holds, which is what the
    # three-way agreement wants.
    models: list[str] = field(default_factory=list)
    # How close two answers have to be to be one place (m).
    merge_m: float = 0.75
    # Answers to return. 0 = all of them.
    top: int = 0
    # The recording's stream names. Empty means the defaults `RecordingFrames` knows,
    # which are not what every recording calls them -- bike.db's are
    # `realsense_color_image_compressed` and `realsense_camera_info`, and a query that
    # guesses wrong finds the images but no intrinsics, so every answer dies as
    # "no camera_info for <frame>" with a warning and no exception.
    color_stream: str = ""
    depth_stream: str = ""
    color_info_stream: str = ""
    depth_info_stream: str = ""


class LiveQuery:
    """Everything a question needs, held ready, over a store that is still filling.

    Thread-safe in the only sense that matters here: one question at a time. The models
    are not re-entrant and the answers are ordered by arrival, so two questions running
    at once would interleave their timings into nonsense.
    """

    def __init__(self, store: Any, config: LiveConfig | None = None) -> None:
        self.store = store
        self.config = config or LiveConfig()
        named = {
            key: value
            for key, value in (
                ("color_stream", self.config.color_stream),
                ("depth_stream", self.config.depth_stream),
                ("color_info_stream", self.config.color_info_stream),
                ("depth_info_stream", self.config.depth_info_stream),
            )
            if value
        }
        self.frames = RecordingFrames(store, config=self.config.detect, **named)
        self.boxes = Owlv2Boxes(self.config.detect)
        self.towers = TextTowers(self.config.detect.device or "cpu")
        self.held = ResidentIndex()
        self._lock = threading.Lock()
        self.loaded: dict[str, float] = {}

    def warm(self, specs: Sequence[str] = ()) -> dict[str, float]:
        """Load the models and read the transforms, before anyone is waiting.

        Returns the seconds each part took, because "the first answer was slow" and
        "the detector is slow" are different problems and the split is what tells them
        apart.
        """
        at = time.monotonic()
        for spec in specs:
            self.towers.background(spec)
        self.loaded["towers"] = time.monotonic() - at
        self.loaded["detector"] = self.boxes.warm()
        self.loaded["recording"] = self.frames.warm()
        at = time.monotonic()
        self.loaded["index"] = float(self.catch_up())
        self.loaded["index_s"] = time.monotonic() - at
        return dict(self.loaded)

    def members(self) -> list[tuple[str, str]]:
        """(tag, stream) for every model this query may search."""
        wanted = set(self.config.models)
        return [
            (tag, stream)
            for tag, stream in member_streams(self.store)
            if not wanted or tag in wanted
        ]

    def catch_up(self) -> int:
        """Take in the patches written since the last question. Rows added."""
        members = self.members()
        return self.held.grow(self.store, members) if members else 0

    def patches(self) -> int:
        """How many patches the answer to the next question will be searched over."""
        return sum(held.rows for held in self.held._held.values())

    def ask(self, text: str) -> FoundObjects:
        """One question, against the map as it stands this instant."""
        text = text.strip()
        if not text:
            raise ValueError("a query needs something to look for")
        with self._lock:
            started = time.monotonic()
            timings: dict[str, float] = {}
            added = self.catch_up()
            timings["index"] = time.monotonic() - started
            tags = [tag for tag, _ in self.members() if tag in self.held]
            if not tags:
                tags = [tag for tag, _ in self.members()]
            answers = list(
                find(
                    self.store,
                    self.store,
                    text,
                    config=self.config.detect,
                    models=tags,
                    towers=self.towers,
                    frames=self.frames,
                    boxes=self.boxes,
                    keep_images=True,
                    resident=self.held,
                    timings=timings,
                )
            )
            merge_duplicates(answers, self.config.merge_m)
            result = FoundObjects(
                query=text,
                frame=self.config.detect.world_frame,
                objects=objects_of(answers, self.config.top),
                refused=sum(1 for answer in answers if not answer.found),
                ms=round((time.monotonic() - started) * 1000, 1),
                timings={name: round(value, 4) for name, value in timings.items()},
            )
            logger.info(
                f"hyperspace find {text!r}: {len(result)} place(s), {result.refused} refused, "
                f"{result.ms} ms over {self.patches()} patches (+{added} new)"
            )
            self.answers = answers
            return result

    def close(self) -> None:
        self.towers.close()


def objects_of(answers: Sequence[Detection], top: int = 0) -> list[FoundObject]:
    """One `FoundObject` per place, strongest first.

    One per place, not one per answer: a second look at a cone already found is a
    better box for that cone, not another cone. The look reported is the one the
    detector was surest of, and it brings its own frame with it.
    """

    def place_of(answer: Detection) -> int:
        return answer.place_id if answer.place_id is not None else -answer.rank

    best: dict[int, Detection] = {}
    seen: dict[int, int] = {}
    for answer in answers:
        if answer.box3d is None:
            continue
        place = place_of(answer)
        seen[place] = seen.get(place, 0) + 1
        if place not in best or answer.score > best[place].score:
            best[place] = answer

    found = []
    ordered = sorted(best.values(), key=lambda answer: -answer.score)
    for answer in ordered[:top] if top > 0 else ordered:
        # The refined box when there is one: a place looked at twice has a sharper box
        # than either look gave on its own.
        box = answer.refined or answer.box3d
        assert box is not None
        found.append(
            FoundObject(
                frame=box.frame,
                centre=tuple(float(v) for v in box.centre),
                extent=tuple(float(v) for v in box.extent),
                depth_m=float(box.depth_m),
                confidence=float(answer.score),
                image=answer.image,
                camera_frame=answer.camera_frame,
                stamp=float(answer.ts),
                box2d=tuple(float(v) for v in (answer.box2d or (0.0, 0.0, 0.0, 0.0))),
                place_id=place_of(answer),
                views=seen.get(place_of(answer), 1),
                models=list(answer.models),
            )
        )
    return found
