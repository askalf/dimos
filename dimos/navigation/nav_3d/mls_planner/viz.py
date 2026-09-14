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


"""Rendering for the nav_3d stack's maps, planner artifacts, path, goal and robot.

One set of rerun archetypes shared by the live bridge, through a blueprint's rerun
config, and by plan_rrd, so a replay and a live run draw the same data the same way.
The planner's ``viz_publish_hz`` decides whether it emits the surface, nodes and
edges at all (0.0 = not at all, the default).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.msgs.nav_msgs.LineSegments3D import LineSegments3D
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from rerun._baseclasses import Archetype

    from dimos.msgs.geometry_msgs.PointStamped import PointStamped
    from dimos.msgs.nav_msgs.Path import Path

GRAPH_Z_LIFT = 0.05
PATH_Z_LIFT = 0.05

TIGHT_COLOR = (4.0, 8.0, 48.0)
OPEN_COLOR = (150.0, 200.0, 255.0)
NODE_COLOR = (255, 200, 0)
PATH_COLOR = (0, 255, 0)
GOAL_COLOR = (255, 0, 0)
BODY_COLOR = (0, 255, 127)
CLEARANCE_COLOR = (255, 120, 120, 80)


def clearance_colors(clearance: NDArray[np.float32], clamp_m: float) -> NDArray[np.uint8]:
    """Blue ramp from tight to open, saturating at clamp_m of clearance."""
    norm = np.clip(np.nan_to_num(clearance / clamp_m, nan=1.0, posinf=1.0), 0.0, 1.0)
    tight, open_ = np.array(TIGHT_COLOR), np.array(OPEN_COLOR)
    return np.asarray(tight + norm[:, None] * (open_ - tight), dtype=np.uint8)


def voxel_map_points(pts: NDArray[np.float32], voxel_size: float) -> Archetype:
    """Voxel centers colored by height on the turbo ramp."""
    import rerun as rr

    if len(pts) == 0:
        return rr.Points3D([])
    z = pts[:, 2]
    class_ids = ((z - z.min()) / (z.max() - z.min() + 1e-8) * 255).astype(np.uint8)
    return rr.Points3D(pts, class_ids=class_ids, radii=voxel_size / 3)


def render_voxel_map(msg: PointCloud2, voxel_size: float) -> Archetype:
    return voxel_map_points(msg.points_f32(), voxel_size)


def surface_points(
    pts: NDArray[np.float32],
    clearance: NDArray[np.float32],
    voxel_size: float,
    wall_clearance_m: float,
    clearance_clamp_m: float,
) -> Archetype:
    """Floor cells colored by wall clearance, dropping the ones the robot cannot fit on."""
    import rerun as rr

    passable = clearance >= wall_clearance_m
    pts, clearance = pts[passable], clearance[passable]
    return rr.Points3D(
        positions=pts,
        colors=clearance_colors(clearance, clearance_clamp_m),
        radii=voxel_size * 0.5,
    )


def render_surface_map(
    msg: PointCloud2,
    voxel_size: float,
    wall_clearance_m: float,
    clearance_clamp_m: float,
) -> Archetype:
    """Surface cells with clearance on the intensity channel, flat blue without it."""
    pts = msg.points_f32()
    clearance = msg.intensities_f32()
    if clearance is None or len(clearance) != len(pts):
        return msg.to_rerun(voxel_size=voxel_size, colors=[40, 75, 130])
    return surface_points(pts, clearance, voxel_size, wall_clearance_m, clearance_clamp_m)


def graph_nodes(pts: NDArray[np.float32]) -> Archetype:
    import rerun as rr

    if len(pts) == 0:
        return rr.Points3D([])
    lifted = pts.copy()
    lifted[:, 2] += GRAPH_Z_LIFT
    return rr.Points3D(positions=lifted, colors=[NODE_COLOR], radii=0.05)


def render_nodes(msg: PointCloud2) -> Archetype:
    return graph_nodes(msg.points_f32())


def graph_edges(edges: NDArray[np.float32]) -> Archetype:
    """Edges as ``[x0, y0, z0, x1, y1, z1, cost]`` rows, colored green to red by cost."""
    segments = LineSegments3D(
        segments=edges[:, :6].reshape(-1, 2, 3) if len(edges) else None,
        weights=edges[:, 6] if len(edges) else None,
    )
    return render_node_edges(segments)


def render_node_edges(msg: LineSegments3D) -> Archetype:
    return msg.to_rerun(z_offset=GRAPH_Z_LIFT, radii=0.01)


def path_strip(
    waypoints: NDArray[np.float32] | None, color: tuple[int, int, int] = PATH_COLOR
) -> Archetype:
    import rerun as rr

    if waypoints is None or len(waypoints) == 0:
        return rr.LineStrips3D([])
    points = np.asarray(waypoints, dtype=np.float32).copy()
    points[:, 2] += PATH_Z_LIFT
    return rr.LineStrips3D([points], colors=[color], radii=0.05)


def render_path(msg: Path) -> Archetype | None:
    """The planned route. An empty path means no route, so the last one stays drawn."""
    if len(msg.poses) == 0:
        return None
    return path_strip(np.array([[p.x, p.y, p.z] for p in msg.poses], dtype=np.float32))


def goal_point(xyz: tuple[float, float, float]) -> Archetype:
    import rerun as rr

    return rr.Points3D([xyz], colors=[GOAL_COLOR], radii=0.1)


def render_goal(msg: PointStamped) -> Archetype | None:
    """The active goal. NaN is the movement manager's placeholder for none."""
    if any(math.isnan(v) for v in (msg.x, msg.y, msg.z)):
        return None
    return goal_point((msg.x, msg.y, msg.z))


def robot_body_box(length: float, width: float, height: float) -> Archetype:
    import rerun as rr

    return rr.Boxes3D(half_sizes=[length / 2, width / 2, height / 2], colors=[BODY_COLOR])


def robot_clearance(height: float, wall_clearance_m: float) -> Archetype:
    """The planner's wall clearance as a cylinder around the body."""
    import rerun as rr

    return rr.Cylinders3D(
        lengths=[height],
        radii=[wall_clearance_m],
        colors=[CLEARANCE_COLOR],
        fill_mode="solid",
    )


def nav_static(
    length: float, width: float, height: float, wall_clearance_m: float
) -> dict[str, Callable[[Any], list[Any]]]:
    """Bridge static entities: the body box and clearance cylinder riding on base_link."""
    return {
        "world/robot_body": lambda rr: [
            robot_body_box(length, width, height),
            rr.Transform3D(parent_frame="tf#/base_link"),
        ],
        "world/robot_body/clearance": lambda rr: [robot_clearance(height, wall_clearance_m)],
    }


def nav_visual_override(
    viz_publish_hz: float,
    voxel_size: float,
    wall_clearance_m: float,
    clearance_clamp_m: float = 1.0,
) -> dict[str, Any]:
    """Bridge overrides for the maps, path, goal and the planner's debug entities.

    Pass the same ``viz_publish_hz``, ``voxel_size`` and ``wall_clearance_m`` given to
    ``MLSPlannerNative.blueprint(...)``.
    """
    on = viz_publish_hz > 0.0
    voxels = partial(render_voxel_map, voxel_size=voxel_size)
    surface = partial(
        render_surface_map,
        voxel_size=voxel_size,
        wall_clearance_m=wall_clearance_m,
        clearance_clamp_m=clearance_clamp_m,
    )
    return {
        "world/global_map": voxels,
        "world/full_map": voxels,
        "world/local_map": voxels,
        "world/path": render_path,
        "world/goal": render_goal,
        "world/surface_map": surface if on else None,
        "world/nodes": render_nodes if on else None,
        "world/node_edges": render_node_edges if on else None,
    }
