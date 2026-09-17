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

from __future__ import annotations

import json
from typing import Any

import pytest

from dimos.evals.suites.lib.dimsim_scene import SCENES_DIR, build_scene, three_to_ros_xy


def _entry(title: str, x: float, y: float, z: float) -> dict[str, Any]:
    return {
        "id": title.lower().replace(" ", "-"),
        "title": title,
        "transform": {"position": {"x": x, "y": y, "z": z}},
    }


MANIFEST = [
    _entry("Sectional", 4.0, 0.5, 1.0),  # floor furniture, the goal
    _entry("Framed painting", 4.0, 1.9, 0.1),  # wall-mounted: above height cut
    _entry("Wine glass", 1.0, 0.9, 0.4),  # tabletop clutter
    _entry("Queen size bed", -4.9, 0.7, -2.4),
]


def test_frame_swap_matches_dimsim_client() -> None:
    """DimSimClient sends ROS (x, y, z) as Three (y, z, x); invert that here."""
    assert three_to_ros_xy({"x": 1.0, "y": 0.5, "z": 2.0}) == (2.0, 1.0)


def test_build_scene_filters_and_converts() -> None:
    scene = build_scene(MANIFEST, "sectional", half_extent=0.5)
    labels = [o["label"] for o in scene["obstacles"]]
    assert labels == ["Sectional", "Queen size bed"]
    assert scene["goal"] == {"label": "Sectional", "xy": [1.0, 4.0]}
    sectional = scene["obstacles"][0]
    assert sectional["min_xy"] == [0.5, 3.5]
    assert sectional["max_xy"] == [1.5, 4.5]


def test_room_bounds_enclose_every_obstacle() -> None:
    scene = build_scene(MANIFEST, "bed")
    minx, miny, maxx, maxy = scene["room_bounds"]
    for o in scene["obstacles"]:
        assert minx <= o["min_xy"][0] and o["max_xy"][0] <= maxx
        assert miny <= o["min_xy"][1] and o["max_xy"][1] <= maxy


def test_unknown_goal_lists_titles() -> None:
    with pytest.raises(LookupError, match="Sectional"):
        build_scene(MANIFEST, "hot tub")


def test_shipped_apartment_scene_is_current() -> None:
    """The committed JSON must be what the generator produces from the manifest."""
    manifest = json.loads((SCENES_DIR / "apartment" / "objects" / "manifest.json").read_text())
    expected = build_scene(manifest, "sectional")
    shipped = json.loads(
        (SCENES_DIR.parents[2] / "dimos/evals/suites/scenes/apartment_couch.json").read_text()
    )
    assert shipped == expected
