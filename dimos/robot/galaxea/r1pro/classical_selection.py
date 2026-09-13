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

"""Conjunctive scene selection: descriptions never silently become another object."""

import colorsys
import re
from typing import Any


def color_name(rgba: list[float]) -> str:
    """Describe the sim material; color is metadata, not an image classifier."""
    h, s, v = colorsys.rgb_to_hsv(*rgba[:3])
    if v < 0.2:
        return "black"
    if s < 0.2:
        return "white" if v > 0.75 else "gray"
    return ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "red")[
        next(
            i
            for i, bound in enumerate((0.035, 0.10, 0.18, 0.44, 0.54, 0.70, 0.80, 0.96, 1.01))
            if h < bound
        )
    ]


def resolve_classical_object(rows: list[dict[str, Any]], selector: str) -> int:
    """Require every supplied attribute, then apply a requested spatial superlative.

    Left/right mean robot-relative position. The arm argument is separate and
    must never change which object this description selects.
    """
    words = re.sub(r"[^a-z0-9_ ]", " ", selector.lower()).split()
    text = " ".join(words)
    for row in rows:
        if text in (row["id"], row["object"]):
            if not _available(row):
                raise ValueError(f"{text} is not a supported, settled, unheld object")
            return int(row["index"])
    eligible = [row for row in rows if _available(row)]
    aliases = {
        "carton": "drink_carton",
        "drink": "drink_carton",
        "cup": "cup",
        "bottle": "bottle",
        "glue": "glue_stick",
        "stick": "glue_stick",
        "toy": "toy_block",
        "block": "toy_block",
        "box": "box",
        "cylinder": "cylinder",
        "drink_carton": "drink_carton",
        "glue_stick": "glue_stick",
        "toy_block": "toy_block",
    }
    colors = {
        "red",
        "orange",
        "yellow",
        "green",
        "cyan",
        "blue",
        "purple",
        "pink",
        "black",
        "white",
        "gray",
    }
    ignored = {"the", "a", "an", "on", "at", "to", "its", "your", "my", "side", "object", "item"}
    ranks = {
        "nearest": ("distance_m", 1),
        "closest": ("distance_m", 1),
        "furthest": ("distance_m", -1),
        "farthest": ("distance_m", -1),
        "leftmost": ("left_m", -1),
        "rightmost": ("left_m", 1),
    }
    rank = None
    constrained = False
    for word in words:
        if word in ignored:
            continue
        constrained = True
        if word in aliases:
            value = aliases[word]
            eligible = [r for r in eligible if value in (r.get("kind"), r["shape"])]
        elif word in colors:
            eligible = [
                r
                for r in eligible
                if color_name(r["rgba"]) == word
                or (word == "blue" and color_name(r["rgba"]) == "cyan")
            ]
        elif word == "light":
            eligible = [r for r in eligible if min(r["rgba"][:3]) >= 0.3]
        elif word == "dark":
            eligible = [r for r in eligible if max(r["rgba"][:3]) < 0.6]
        elif word in ("left", "right"):
            sign = 1 if word == "left" else -1
            eligible = [r for r in eligible if sign * r["left_m"] > 0.02]
        elif word in ranks:
            if rank is not None:
                raise ValueError("Use one spatial superlative")
            rank = ranks[word]
        else:
            raise ValueError(
                f"Unsupported attribute {word!r}; use the exact object ID from get_scene"
            )
    if not constrained or not eligible:
        raise ValueError(f"No available object matches all attributes in {selector!r}")
    if rank:
        key, sign = rank
        eligible.sort(key=lambda row: (sign * row[key], row["index"]))
        if len(eligible) > 1 and abs(eligible[0][key] - eligible[1][key]) < 0.01:
            raise ValueError("Spatial selection is tied; specify an object ID")
        return int(eligible[0]["index"])
    if len(eligible) != 1:
        raise ValueError(f"Ambiguous object description; choose from {[r['id'] for r in eligible]}")
    return int(eligible[0]["index"])


def _available(row: dict[str, Any]) -> bool:
    return bool(row["released"] and row["settled"] and row["upright"] and row["support_geoms"])
