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

"""Build a policy scene JSON from a DimSim ``objects/manifest.json``.

The manifest is the god view: every placed object with its Three.js world
transform. It carries positions but not footprints (extents live inside the
GLBs), so every obstacle gets the same square footprint. Real centres,
approximate sizes; tune ``half_extent`` per scene.

Frames: DimSim is Three.js Y-up. ``DimSimClient`` maps ROS ``(x, y, z)`` to
Three ``(y, z, x)``, so ROS ``x = three.z`` and ROS ``y = three.x``.

    python -m dimos.evals.suites.lib.dimsim_scene apartment sectional \
        > dimos/evals/suites/scenes/apartment_couch.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from dimos.constants import DIMOS_PROJECT_ROOT

SCENES_DIR = DIMOS_PROJECT_ROOT / "misc" / "DimSim" / "scenes"

# Above this centre height (Three.js y, metres) an object is wall-mounted or
# on a high shelf, not something a quadruped drives into.
MAX_CENTER_HEIGHT_M = 1.3

# Tabletop clutter: it sits on furniture that is already an obstacle, and
# Jev's accuracy falls with irrelevant state.
CLUTTER = (
    "plate",
    "cutlery",
    "placemat",
    "wine glass",
    "shaker",
    "books",
    "plant",
    "kettle",
    "toaster",
    "blender",
    "coffee maker",
    "microwave",
    "soap",
    "bookshelf decor",
    "shower head",
    "towel",
    "pillow",
    "blanket",
    "duvet",
    "teddy",
    "laptop",
    "mug",
    "smartphone",
    "bed tray",
    "book ",
    "wall shelves",
    "watering can",
    "garden pots",
)


def three_to_ros_xy(position: dict[str, float]) -> tuple[float, float]:
    return (float(position["z"]), float(position["x"]))


def is_floor_obstacle(entry: dict[str, Any]) -> bool:
    if float(entry["transform"]["position"]["y"]) > MAX_CENTER_HEIGHT_M:
        return False
    title = str(entry["title"]).lower()
    return not any(word in title for word in CLUTTER)


def build_scene(
    manifest: list[dict[str, Any]],
    goal_title: str,
    *,
    half_extent: float = 0.5,
    margin: float = 0.5,
) -> dict[str, Any]:
    goal = next((e for e in manifest if goal_title.lower() in str(e["title"]).lower()), None)
    if goal is None:
        titles = sorted({str(e["title"]) for e in manifest})
        raise LookupError(f"no object titled like {goal_title!r}; have {titles}")

    obstacles = []
    for entry in manifest:
        if not is_floor_obstacle(entry):
            continue
        x, y = three_to_ros_xy(entry["transform"]["position"])
        obstacles.append(
            {
                "label": str(entry["title"]),
                "min_xy": [round(x - half_extent, 3), round(y - half_extent, 3)],
                "max_xy": [round(x + half_extent, 3), round(y + half_extent, 3)],
            }
        )

    xs = [o["min_xy"][0] for o in obstacles] + [o["max_xy"][0] for o in obstacles]
    ys = [o["min_xy"][1] for o in obstacles] + [o["max_xy"][1] for o in obstacles]
    gx, gy = three_to_ros_xy(goal["transform"]["position"])
    return {
        "frame_id": "world",
        "source": f"misc/DimSim/scenes manifest, goal={goal['id']}, half_extent={half_extent}",
        "goal": {"label": str(goal["title"]), "xy": [round(gx, 3), round(gy, 3)]},
        "room_bounds": [
            round(min(xs) - margin, 3),
            round(min(ys) - margin, 3),
            round(max(xs) + margin, 3),
            round(max(ys) + margin, 3),
        ],
        "obstacles": obstacles,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="scene folder under misc/DimSim/scenes")
    parser.add_argument("goal", help="substring of the goal object's title")
    parser.add_argument("--half-extent", type=float, default=0.5)
    args = parser.parse_args(argv)
    manifest_path = SCENES_DIR / args.scene / "objects" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    json.dump(build_scene(manifest, args.goal, half_extent=args.half_extent), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
