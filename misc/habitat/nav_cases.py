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

"""Navigation cases and a mapping tour for one Habitat scene, from its ground-truth boxes.

Runs under the Habitat native env (python 3.9, no dimos imports), like server.py::

    target/habitat/env/bin/python misc/habitat/nav_cases.py \\
        misc/habitat/ground_truth/hssd/104348463_171513588.json \\
        target/habitat/data/hssd-hab/hssd-hab.scene_dataset_config.json \\
        dimos/evals/suites/scenes/habitat/104348463_171513588.json

The scene file references the ground truth by path and adds ``cases`` (label, spawn,
end point beside the target, geodesic distance) and ``tour`` (navmesh waypoints
spawn -> every end point -> spawn), all in the ROS world frame.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import habitat_sim
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "frames", ROOT / "dimos/simulation/habitat/frames.py"
)
assert _spec is not None and _spec.loader is not None
frames = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frames)
_nspec = importlib.util.spec_from_file_location(
    "navmesh", ROOT / "dimos/simulation/habitat/navmesh.py"
)
assert _nspec is not None and _nspec.loader is not None
navmesh = importlib.util.module_from_spec(_nspec)
_nspec.loader.exec_module(navmesh)

# Target labels in priority order; one case per label that has a reachable object.
WANT = (
    "chair", "bed", "couch", "sofa", "table", "toilet", "tv", "refrigerator", "fridge", "sink",
    "desk", "cabinet", "bathtub", "washer", "stool", "plant", "shelves", "dresser",
    "nightstand", "piano", "fireplace", "oven", "stove", "bench", "counter", "shower",
)  # fmt: skip
MAX_CASES = 6
MIN_GEODESIC_M, MAX_GEODESIC_M = 3.0, 12.0
# Furniture-sized: something an agent can be sent to and would see in a 20-object list.
MIN_FOOTPRINT_M, MIN_HEIGHT_M = 0.4, 0.2
END_STANDOFF_M = 0.6  # ring radius beyond the box's half diagonal
END_CLEARANCE_M = 0.25  # from the nearest navmesh edge
SPAWN_CLEARANCE_M = 0.5
TOUR_STEP_M = 0.5


def main(ground_truth: Path, dataset: Path, out: Path) -> None:
    gt = json.loads(ground_truth.read_text())
    scene_id = gt["scene_id"]
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_dataset_config_file = str(dataset)
    backend.scene_id = scene_id
    backend.enable_physics = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = []
    agent.action_space = {}
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    print("navmesh:", navmesh.ensure_navmesh(sim, scene_id, ROOT / "target/habitat/navmesh"))
    pf = sim.pathfinder
    print("navigable area m2:", round(pf.navigable_area, 1), "islands:", pf.num_islands)

    def shortest(a: Any, b: Any) -> tuple[float, list[Any]]:
        p = habitat_sim.ShortestPath()
        p.requested_start = np.asarray(a, np.float32)
        p.requested_end = np.asarray(b, np.float32)
        return (
            (p.geodesic_distance, [np.asarray(q) for q in p.points])
            if pf.find_path(p)
            else (math.inf, [])
        )

    def end_point(d: dict[str, Any]) -> Any:
        """Navigable ring point beside the box with the most clearance, in the Habitat frame."""
        cx, cy, cz = d["center_xyz"]
        sx, sy, sz = d["size_xyz"]
        r = math.hypot(sx, sy) / 2 + END_STANDOFF_M
        best = None
        for k in range(24):
            a = 2 * math.pi * k / 24
            ros = np.array([cx + r * math.cos(a), cy + r * math.sin(a), cz - sz / 2])
            hab = frames.position_to_habitat(ros).astype(np.float32)
            snapped = pf.snap_point(hab)
            if not np.isfinite(snapped).all() or np.linalg.norm(snapped - hab) > 0.5:
                continue
            clear = pf.distance_to_closest_obstacle(snapped, 2.0)
            if clear >= END_CLEARANCE_M and (best is None or clear > best[0]):
                best = (clear, snapped)
        return None if best is None else best[1]

    furniture = [
        d for d in gt["detections"]
        if not d["label"].startswith("wall")
        and max(d["size_xyz"][:2]) >= MIN_FOOTPRINT_M and d["size_xyz"][2] >= MIN_HEIGHT_M
    ]  # fmt: skip
    targets = []
    for label in WANT:
        for d in (d for d in furniture if label in d["label"].lower()):
            e = end_point(d)
            if e is not None:
                targets.append((d, e))
                break

    best = None
    for seed in range(60):
        pf.seed(seed)
        spawn = pf.get_random_navigable_point()
        if pf.distance_to_closest_obstacle(spawn, 2.0) < SPAWN_CLEARANCE_M:
            continue
        picked = [(d, e, shortest(spawn, e)[0]) for d, e in targets]
        picked = [t for t in picked if MIN_GEODESIC_M <= t[2] <= MAX_GEODESIC_M]
        if best is None or len(picked) > len(best[1]):
            best = (spawn, picked)
    if best is None or not best[1]:
        raise SystemExit(f"no cases for {scene_id}: {len(targets)} targets")
    spawn, picked = best[0], best[1][:MAX_CASES]
    spawn_ros = frames.position_to_ros(spawn).round(3).tolist()

    stops = [spawn] + [e for _, e, _ in picked] + [spawn]
    tour = []
    for i in range(len(stops) - 1):
        pts = shortest(stops[i], stops[i + 1])[1]
        for j in range(len(pts) - 1):
            q0, q1 = pts[j], pts[j + 1]
            n = max(1, int(np.linalg.norm(q1 - q0) / TOUR_STEP_M))
            for k in range(n):
                tour.append(frames.position_to_ros(q0 + (q1 - q0) * k / n).round(2).tolist()[:2])
    tour.append(spawn_ros[:2])

    scene = {
        "scene_id": scene_id,
        # Not resolved: the data dir is usually a symlink out of the repo.
        "scene_dataset_config": os.path.relpath(dataset.absolute(), ROOT),
        "ground_truth": os.path.relpath(ground_truth.absolute(), ROOT),
        "cases": [
            {
                "label": d["label"],
                "object_id": d["id"],
                "spawn_xyz": spawn_ros,
                "spawn_yaw_deg": 0.0,
                "end_xy": frames.position_to_ros(e).round(2).tolist()[:2],
                "geodesic_m": round(float(g), 2),
            }
            for d, e, g in picked
        ],
        "tour": tour,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, indent=2) + "\n")
    print(scene_id, [(c["label"], c["geodesic_m"]) for c in scene["cases"]], "tour", len(tour))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
