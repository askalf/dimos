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

"""Display the applied validation export without rewriting its saved review history."""

from typing import Any

# Original review ID -> (current suite ID, applied exported revision).
# The bedroom-sofa answer was clarified as Left in the accompanying user review.
APPLIED_REVIEWS = {
    "hssd_104348463_171513588_fridge_exists": ("hssd_104348463_171513588_island_chairs", 9),
    "hssd_106366410_174226806_treadmill_location": (
        "hssd_106366410_174226806_red_trash_bin_location",
        11,
    ),
    "hssd_106366410_174226806_bedroom_sofa": ("hssd_106366410_174226806_bed_relative_to_sofa", 10),
    "new-46515ccf-ac03-4965-b9d3-105bc646d283": (
        "hssd_106878858_174886965_bathroom_floor_pattern_match",
        10,
    ),
    "new-c35abd86-10d8-49cb-9648-5af00a6a03d8": ("hssd_108736851_177263586_dining_chairs", 5),
    "new-8a2b1b5b-062f-4b26-8fc1-f870bd08728d": (
        "hssd_108736851_177263586_curved_sofa_table_shape",
        12,
    ),
    "new-3f34d633-2c44-40e0-a460-5455b7afb7d6": ("hssd_108736851_177263586_side_table_sides", 22),
    "hssd_108736884_177263634_laptop_location": (
        "hssd_108736884_177263634_red_potted_plant_location",
        9,
    ),
    "hssd_108736884_177263634_kitchen_perimeter": (
        "hssd_108736884_177263634_kitchen_counter_windows",
        6,
    ),
    "hssd_108736884_177263634_bathtubs": ("hssd_108736884_177263634_bathtub_shape_match", 7),
    "new-cd721aa6-11fc-47e4-a5c4-eefd45f80d9e": (
        "hssd_108736884_177263634_bathroom_plant_exists",
        11,
    ),
}


def current_reviews(catalog: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    """Hide retired questions and alias applied suggestions to their real suite IDs.

    This is a display projection only. Raw export and SQLite retain every old
    entry. A new edit writes the canonical ID at revision zero; it then takes
    precedence over the historical alias. Later edits to a historical record
    stay marked for changes rather than being silently treated as applied.
    """
    result = {}
    for slug, scene in catalog.items():
        questions = {q["id"]: q for q in scene["questions"]}
        rows = history.get(slug, {})
        visible = {
            qid: row
            for qid, row in rows.items()
            if qid in questions
            or (row.get("original", {}).get("new") and qid not in APPLIED_REVIEWS)
        }
        for old_id, (new_id, applied_revision) in APPLIED_REVIEWS.items():
            if old_id not in rows or new_id not in questions or new_id in visible:
                continue
            row = rows[old_id]
            applied = row["revision"] == applied_revision
            visible[new_id] = row | {
                "revision": 0,
                "status": "approved" if applied else "changes",
                "comment": "" if applied else row.get("comment", ""),
                "proposed_question": "" if applied else row.get("proposed_question", ""),
                "proposed_options": "" if applied else row.get("proposed_options", ""),
                "proposed_answer": "" if applied else row.get("proposed_answer", ""),
                "original": questions[new_id],
                "applied_from": old_id,
                "applied_comment": row.get("comment", "")
                if applied
                else "A later edit to the original review still needs attention.",
            }
        result[slug] = visible
    return result
