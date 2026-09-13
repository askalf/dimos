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

"""Physical bench scenes and empty placement regions shared by collection and execution."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Protocol
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout, prepare_object_scene
from dimos.robot.galaxea.r1pro.object_packing_state import object_extent
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState
from dimos.robot.galaxea.r1pro.object_primitives import Arm
from dimos.robot.galaxea.r1pro.placement_regions import (
    PlacementObstacle,
    PlacementRegion,
    placement_candidates,
)


class PrimitivePlacementContext(Protocol):
    model: mujoco.MjModel
    data: mujoco.MjData
    layout: ObjectLayout
    arm: Arm
    home: NDArray[np.float64]
    bottle_id: int

    @property
    def selected(self) -> int: ...


def bilateral_layout(layout: ObjectLayout) -> ObjectLayout:
    """Use the interactive bench arrangement, retaining every tray occupant unchanged."""
    return ObjectLayout(
        layout.seed,
        tuple(
            replace(o, position=(o.position[0], -o.position[1], o.position[2]), yaw=-o.yaw)
            if i % 2 == 0 and not o.in_tray
            else o
            for i, o in enumerate(layout.objects)
        ),
    )


def prepare_primitive_scene(
    output: Path, layout: ObjectLayout, arm: Arm, *, scene_package: Path | None = None
) -> tuple[Path, ObjectLayout]:
    """A symmetric physical bench overlay; only initial scene generation mirrors objects."""
    if arm == "left":
        layout = ObjectLayout(
            layout.seed,
            tuple(
                replace(o, position=(o.position[0], -o.position[1], o.position[2]), yaw=-o.yaw)
                for o in layout.objects
            ),
        )
    scene = prepare_object_scene(output, layout, scene_package=scene_package)
    tree = ET.parse(scene)
    root = tree.getroot()
    table = root.find('.//body[@name="task_table"]')
    tray = root.find('.//body[@name="task_bin"]')
    if table is None or tray is None or (top := table.find("geom")) is None:
        raise ValueError("Primitive bench requires a physical source table and tray")
    table.set("pos", "0.49 0 0.675")
    top.set("size", "0.31 0.65 0.025")
    if arm == "left":
        tray.set("pos", "0.34 0.04 0.701")
    tree.write(scene, encoding="unicode")
    scene.with_suffix(".objects.json").write_text(json.dumps(layout.to_dict()) + "\n")
    return scene, layout


def placement_options(
    task: PrimitivePlacementContext, destination: str | PlacementRegion, seed: int | None
) -> tuple[tuple[tuple[float, float, float], ...], PlacementRegion]:
    """Empty supported object goals, before any reachability or base-path search."""
    if isinstance(destination, PlacementRegion):
        region = destination
    elif destination == "tray":
        tray = task.data.body("task_bin")
        region = PlacementRegion(
            "tray", tuple(tray.xpos + np.array([0, 0, 0.015])), (0.145, 0.105), ("bin_floor",)
        )
    elif destination in ("table", "worktable"):
        sign = -1 if task.arm == "right" else 1
        top = next(
            task.model.geom(i).name
            for i in range(task.model.ngeom)
            if task.model.geom_bodyid[i] == task.model.body("task_table").id
        )
        region = (
            PlacementRegion("worktable", (0.46, 0.0, 0.7), (0.18, 0.57), (top,))
            if destination == "worktable"
            else PlacementRegion("table", (0.41, sign * 0.44, 0.7), (0.15, 0.16), (top,))
        )
    else:
        raise ValueError("Collection destination must be table, worktable or tray")
    obstacles = []
    for i, obj in enumerate(task.layout.objects):
        if i == task.selected:
            continue
        body = task.data.body(obj.name)
        extent = object_extent(obj, body.xmat.reshape(3, 3))
        obstacles.append(
            PlacementObstacle(
                tuple(body.xpos[:2]),
                tuple(extent[:2]),
                float(body.xpos[2] - extent[2]),
                float(body.xpos[2] + extent[2]),
            )
        )
    if "bin_floor" not in region.support_geoms:
        tray = task.data.body("task_bin")
        obstacles.append(
            PlacementObstacle(
                tuple(tray.xpos[:2]), (0.16, 0.235), float(tray.xpos[2]), float(tray.xpos[2] + 0.1)
            )
        )
    obj = task.layout.objects[task.selected]
    points = placement_candidates(
        region, radius=obj.radius, half_height=obj.half_size[2], obstacles=tuple(obstacles)
    )
    source = task.data.body(task.bottle_id).xpos
    if seed is not None:
        points = tuple(p for p in points if np.linalg.norm(np.asarray(p[:2]) - source[:2]) >= 0.045)
    if not points:
        raise RuntimeError("No empty placement region with object and open-finger clearance")
    return points, region


def choose_placement(
    task: PrimitivePlacementContext, destination: str | PlacementRegion, seed: int | None
) -> tuple[NDArray[np.float64], PlacementRegion]:
    """Geometric placement, then prepositioning; no extra ACT command is inferred."""
    points, region = placement_options(task, destination, seed)
    first = 0 if seed is None else int(np.random.default_rng(seed).integers(len(points)))
    scene = PrimitiveSceneState(task.model, task.data, task.layout, task.home)
    planner = scene.transport_planner()
    base = task.data.body("base_link")
    yaw = float(np.arctan2(base.xmat[3], base.xmat[0]))
    # An empty object footprint may still leave the carrying posture in a
    # collision. Try the other empty spots before rejecting the entire region.
    for offset in range(len(points)):
        target = np.asarray(points[(first + offset) % len(points)])
        if any(
            planner.clear_pose_segment(pose, pose)
            for pose in scene.preposition_poses(task.arm, target, yaw=yaw)
        ):
            return target, region
    raise RuntimeError("No empty placement spot has a clear carrying pose in this region")
