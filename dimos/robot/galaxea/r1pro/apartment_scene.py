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

"""Measured apartment support patches and reproducible randomized object placement."""

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout
from dimos.robot.galaxea.r1pro.placement_regions import (
    PlacementObstacle,
    PlacementRegion,
    placement_candidates,
)
from dimos.robot.galaxea.r1pro.tray_task import laptop_destination


def apartment_regions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, PlacementRegion]:
    """Identify level table/counter patches in this house from actual collision geometry."""
    desktop = laptop_destination(model, data)
    regions = {
        "dining_table": PlacementRegion(
            "dining_table",
            (desktop.tray_position[0], desktop.tray_position[1], desktop.tray_position[2]),
            (0.15, 0.13),
            (desktop.support_geom,),
        )
    }
    point = np.array([-2.64, -1.10])
    candidates = []
    for gid in range(model.ngeom):
        geom = model.geom(gid)
        if geom.type[0] != mujoco.mjtGeom.mjGEOM_BOX or not (
            geom.contype[0] or geom.conaffinity[0]
        ):
            continue
        # Only static supports; a placed object cannot become the countertop.
        bid = int(geom.bodyid[0])
        if model.body_dofnum[bid] or model.body_rootid[bid] == model.body("base_link").id:
            continue
        rotation = data.geom_xmat[gid].reshape(3, 3)
        extent = np.abs(rotation) @ geom.size
        top = float(data.geom_xpos[gid, 2] + extent[2])
        if (
            abs(rotation[2, 2]) > 0.999
            and 0.8 < top < 0.95
            and np.all(
                np.abs(point - data.geom_xpos[gid, :2]) + np.array([0.14, 0.16]) < extent[:2]
            )
        ):
            candidates.append((top, geom.name))
    if not candidates:
        raise RuntimeError("No physical kitchen-counter patch with adequate clearance")
    top, name = max(candidates)
    regions["kitchen"] = PlacementRegion("kitchen", (*point, top), (0.13, 0.15), (name,))
    table = model.body("task_table").id
    geom = model.geom(int(model.body_geomadr[table]))
    top = float(data.geom_xpos[geom.id, 2] + geom.size[2])
    regions["worktable"] = PlacementRegion("worktable", (0.46, 0, top), (0.18, 0.57), (geom.name,))
    return regions


def distribute_apartment_objects(
    scene: Path, layout: ObjectLayout
) -> tuple[ObjectLayout, dict[str, PlacementRegion]]:
    """Randomize supported poses across source table, laptop desk and kitchen counter.

    This changes only the initial XML. Runtime movement always uses physical
    controls; there are no attachments or object-position resets during actions.
    """
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    regions = apartment_regions(model, data)
    rng = np.random.default_rng(layout.seed + 991)
    stations = list(
        rng.permutation(["worktable", "worktable", "dining_table", "kitchen", "kitchen"])
    )
    tree = ET.parse(scene)
    placed = []
    occupancy: dict[str, list[PlacementObstacle]] = {name: [] for name in regions}
    # Keep the tray and its handles clear on the source table.
    tray = data.body("task_bin").xpos
    occupancy["worktable"].append(
        PlacementObstacle(tuple(tray[:2]), (0.16, 0.235), float(tray[2]), float(tray[2] + 0.1))
    )
    for obj, station in zip(layout.objects, stations, strict=False):
        region = regions[station]
        candidates = placement_candidates(
            region,
            radius=obj.radius,
            half_height=obj.half_size[2],
            obstacles=tuple(occupancy[station]),
        )
        if not candidates:
            raise RuntimeError(f"No supported initial space remains on {station}")
        # At the initial worktable retain the existing randomized bench
        # proposals, whose approach side is compatible with the robot's reset
        # posture. Geometry still checks occupancy and finger clearance. Room
        # supports sample their full free patch; reachability is checked later.
        position = (
            min(
                candidates,
                key=lambda p: float(np.linalg.norm(np.asarray(p[:2]) - obj.position[:2])),
            )
            if station == "worktable"
            else candidates[int(rng.integers(len(candidates)))]
        )
        item = replace(obj, position=(position[0], position[1], position[2] + 0.001), in_tray=False)
        placed.append(item)
        occupancy[station].append(
            PlacementObstacle(
                position[:2],
                (obj.radius, obj.radius),
                region.center[2],
                region.center[2] + 2 * obj.half_size[2],
            )
        )
        body = tree.find(f'.//body[@name="{obj.name}"]')
        assert body is not None
        body.set("pos", " ".join(map(str, item.position)))
    tree.write(scene, encoding="unicode")
    return ObjectLayout(layout.seed, tuple(placed)), regions
