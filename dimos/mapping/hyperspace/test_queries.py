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

"""The query surface: three shapes, one handle, and answers you can come back for."""

from __future__ import annotations

from dimos.mapping.hyperspace.queries import (
    AREA_PROMPTS,
    Place,
    Query,
    QueryBook,
    near_enough,
)


def a_place(x: float, score: float = 1.0) -> Place:
    return Place(where=(x, 0.0, 0.0), frame="odom", kind="heatmap", score=score)


def test_coming_back_for_more_walks_the_list_instead_of_repeating_it() -> None:
    """Two calls for the rest must not hand the same answer twice.

    The whole point of the handle is that an agent takes the first place, tries it, and
    comes back. If the second call replayed the first answer it would drive to the same
    spot forever.
    """
    book = QueryBook()
    query = book.put(
        Query(query_id="item-1", text="a cone", kind="item", places=[a_place(i) for i in range(5)])
    )
    query.taken = 1

    first_rest = query.places[query.taken :][:2]
    query.taken += len(first_rest)
    second_rest = query.places[query.taken :]
    query.taken += len(second_rest)

    assert [place.where[0] for place in first_rest] == [1.0, 2.0]
    assert [place.where[0] for place in second_rest] == [3.0, 4.0]
    assert query.remaining == 0


def test_a_forgotten_query_says_so_rather_than_looking_empty() -> None:
    """ "There were no more" and "I no longer remember" are different facts.

    The book is bounded, so a long-running robot drops old questions. An agent told
    "no more results" would conclude it had seen everything.
    """
    book = QueryBook(keep=2)
    for number in range(3):
        book.put(Query(query_id=f"item-{number}", text="a cone", kind="item"))

    assert book.get("item-0") is None, "the oldest was dropped"
    assert book.get("item-2") is not None
    assert book.ids() == ["item-1", "item-2"]


def test_ids_do_not_collide_between_kinds_or_calls() -> None:
    book = QueryBook()
    made = {book.next_id("item") for _ in range(3)} | {book.next_id("area") for _ in range(3)}
    assert len(made) == 6, f"ids repeated: {made}"


def test_a_radius_keeps_what_is_near_and_an_unknown_position_keeps_everything() -> None:
    """A radius measured from a position nobody recorded would answer about the wrong
    part of the map, so no origin means no filtering rather than a filter from zero."""
    places = [a_place(0.0), a_place(5.0), a_place(50.0)]

    assert len(near_enough(places, (0.0, 0.0, 0.0), 10.0)) == 2
    assert len(near_enough(places, (0.0, 0.0, 0.0), 0.0)) == 3, "no radius keeps everything"
    assert len(near_enough(places, None, 10.0)) == 3, "no known position cannot filter"


def test_a_heatmap_place_does_not_claim_a_size_it_never_measured() -> None:
    """A box has an extent because a detector drew one. A voxel does not.

    Reporting a size for a heatmap answer would have an agent plan around a number that
    came from nowhere.
    """
    voxel = a_place(1.0)
    assert voxel.extent is None
    assert "extent" not in voxel.as_dict()

    box = Place(where=(1.0, 2.0, 3.0), frame="odom", kind="item", score=0.8, extent=(0.5, 0.5, 1.0))
    assert box.as_dict()["extent"] == [0.5, 0.5, 1.0]


def test_an_area_is_contrasted_against_objects_not_against_a_room() -> None:
    """Measured on "kitchen" over sf_office_drive1, top six frames judged by eye:
    this set gave 6 of 6 and favoured wide shots of the space, while NO contrast gave
    2 of 6 -- luggage and a pile of boxes. The floor/wall/ceiling set used for objects
    would subtract the room an area query is asking for."""
    assert AREA_PROMPTS, "an area query still needs something to contrast against"
    assert not any(
        word in prompt for prompt in AREA_PROMPTS for word in ("floor", "wall", "ceiling", "room")
    ), f"an area contrast must not subtract the room: {AREA_PROMPTS}"


def test_the_query_module_can_configure_everything_its_start_reads() -> None:
    """Every `self.config.X` on the start path has to exist on the config.

    `Hyperspace.start()` crashed on `max_depth_m` -- present on the ingest's config and
    missing from the query module's -- so the detector never loaded and the module was
    unusable, while `map live` and `map find` were fine because they build the query
    object directly and never go through the module. Nothing caught it because nothing
    started the module. A missing field is an AttributeError at start, which on a robot
    is a module that is simply not there.
    """
    from pathlib import Path
    import re

    from dimos.mapping.hyperspace.module import HyperspaceConfig

    source = Path(__file__).with_name("module.py").read_text()
    # The query module's half of the file: from its config to the end.
    start = source.index("class HyperspaceConfig(")
    read = set(re.findall(r"self\.config\.([a-z_0-9]+)", source[start:]))
    known = set(HyperspaceConfig.model_fields)
    missing = sorted(read - known)
    assert not missing, f"Hyperspace reads config fields it does not declare: {missing}"
