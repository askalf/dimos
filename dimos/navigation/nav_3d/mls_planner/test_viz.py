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

import numpy as np
import pytest

from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.navigation.nav_3d.mls_planner import viz

pytest.importorskip("rerun")


def test_surface_drops_cells_below_the_wall_clearance():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    clearance = np.array([0.05, 0.2, 2.0], dtype=np.float32)
    arch = viz.surface_points(pts, clearance, 0.1, wall_clearance_m=0.1, clearance_clamp_m=1.0)
    assert len(arch.positions) == 2


def test_message_and_array_surface_renderers_agree():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    clearance = np.array([0.2, 0.9], dtype=np.float32)
    cloud = PointCloud2.from_numpy(pts, intensities=clearance, timestamp=0.0)
    a = viz.render_surface_map(cloud, 0.1, 0.1, 1.0)
    b = viz.surface_points(pts, clearance, 0.1, 0.1, 1.0)
    assert a.colors.as_arrow_array().to_pylist() == b.colors.as_arrow_array().to_pylist()
    assert a.radii.as_arrow_array().to_pylist() == b.radii.as_arrow_array().to_pylist()


def test_graph_edges_take_the_binding_layout():
    edges = np.array([[0, 0, 0, 1, 0, 0, 1.0], [1, 0, 0, 1, 1, 0, 10.0]], dtype=np.float32)
    arch = viz.graph_edges(edges)
    strips = np.asarray(arch.strips.as_arrow_array().to_pylist(), dtype=object)
    assert len(strips) == 2
    assert viz.graph_edges(np.zeros((0, 7), dtype=np.float32)).strips is not None


def test_goal_placeholder_is_not_drawn():
    assert viz.render_goal(PointStamped(x=float("nan"), y=0.0, z=0.0)) is None
    assert viz.render_goal(PointStamped(x=1.0, y=2.0, z=0.0)) is not None


def test_overrides_follow_the_planner_publish_rate():
    off = viz.nav_visual_override(0.0, 0.08, 0.1)
    on = viz.nav_visual_override(2.0, 0.08, 0.1)
    assert off["world/surface_map"] is None and off["world/nodes"] is None
    assert callable(on["world/surface_map"]) and callable(on["world/node_edges"])
    assert callable(on["world/global_map"]) and callable(off["world/path"])
