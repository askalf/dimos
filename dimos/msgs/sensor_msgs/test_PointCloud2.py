#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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


import json
from typing import Any

import numpy as np
import open3d.core as o3c
import pytest

from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.type.lidar import pointcloud2_from_webrtc_lidar
from dimos.utils.testing.replay import SensorReplay


@pytest.mark.self_hosted
def test_lcm_encode_decode() -> None:
    """Test LCM encode/decode preserves pointcloud data."""
    replay = SensorReplay("office_lidar", autocast=pointcloud2_from_webrtc_lidar)
    lidar_msg: PointCloud2 = replay.load_one("lidar_data_021")

    binary_msg = lidar_msg.lcm_encode()
    decoded = PointCloud2.lcm_decode(binary_msg)

    # 1. Check number of points
    original_points, _ = lidar_msg.as_numpy()
    decoded_points, _ = decoded.as_numpy()

    assert len(original_points) == len(decoded_points), (
        f"Point count mismatch: {len(original_points)} vs {len(decoded_points)}"
    )

    # 2. Check point coordinates are preserved (within floating point tolerance)
    if len(original_points) > 0:
        np.testing.assert_allclose(
            original_points,
            decoded_points,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Point coordinates don't match between original and decoded",
        )

    # 3. Check frame_id is preserved
    assert lidar_msg.frame_id == decoded.frame_id, (
        f"Frame ID mismatch: '{lidar_msg.frame_id}' vs '{decoded.frame_id}'"
    )

    # 4. Check timestamp is preserved (within reasonable tolerance for float precision)
    if lidar_msg.ts is not None and decoded.ts is not None:
        assert abs(lidar_msg.ts - decoded.ts) < 1e-6, (
            f"Timestamp mismatch: {lidar_msg.ts} vs {decoded.ts}"
        )

    # 5. Check pointcloud properties
    assert len(lidar_msg.pointcloud.points) == len(decoded.pointcloud.points), (
        "Open3D pointcloud size mismatch"
    )


def test_lcm_intensity_round_trip() -> None:
    """Test that intensity values survive an lcm_encode → lcm_decode round trip."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    intensities = np.array([0.25, 1.1, 0.0], dtype=np.float32)

    original = PointCloud2.from_numpy(
        points, frame_id="map", timestamp=42.0, intensities=intensities
    )

    # Verify getter before encoding
    got = original.intensities_f32()
    assert got is not None, "intensities_f32() returned None on source cloud"
    np.testing.assert_allclose(got, intensities, atol=1e-6)

    # Round-trip through LCM
    binary = original.lcm_encode()
    decoded = PointCloud2.lcm_decode(binary)

    # Positions preserved
    decoded_pts, _ = decoded.as_numpy()
    np.testing.assert_allclose(decoded_pts.astype(np.float32), points, atol=1e-6)

    # Intensities preserved
    decoded_intensities = decoded.intensities_f32()
    assert decoded_intensities is not None, "intensities lost after lcm_decode"
    np.testing.assert_allclose(decoded_intensities, intensities, atol=1e-6)


def test_lcm_no_intensity_round_trip() -> None:
    """Clouds without intensity should round-trip without creating spurious intensities."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    original = PointCloud2.from_numpy(points, frame_id="map", timestamp=1.0)

    assert original.intensities_f32() is None

    binary = original.lcm_encode()
    decoded = PointCloud2.lcm_decode(binary)

    # No intensities should appear (all-zero wire data is ignored)
    assert decoded.intensities_f32() is None, "Spurious intensities created from zero wire data"

    decoded_pts, _ = decoded.as_numpy()
    np.testing.assert_allclose(decoded_pts.astype(np.float32), points, atol=1e-6)


def test_lcm_per_point_timing_round_trip() -> None:
    """offset_time/tag/line survive an lcm_encode → lcm_decode round trip."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    intensities = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    # First point offset 0 is meaningful and must survive (no nonzero filtering).
    offset_times = np.array([0, 41_666, 83_332], dtype=np.uint32)
    tags = np.array([0, 16, 32], dtype=np.uint8)
    lines = np.array([0, 1, 3], dtype=np.uint8)

    original = PointCloud2.from_numpy(
        points,
        frame_id="mid360_link",
        timestamp=100.5,
        intensities=intensities,
        offset_times=offset_times,
        tags=tags,
        lines=lines,
    )

    got_offsets = original.offset_times_u32()
    assert got_offsets is not None
    np.testing.assert_array_equal(got_offsets, offset_times)

    decoded = PointCloud2.lcm_decode(original.lcm_encode())

    decoded_pts, _ = decoded.as_numpy()
    np.testing.assert_allclose(decoded_pts.astype(np.float32), points, atol=1e-6)
    decoded_intensities = decoded.intensities_f32()
    assert decoded_intensities is not None
    np.testing.assert_allclose(decoded_intensities, intensities, atol=1e-6)

    decoded_offsets = decoded.offset_times_u32()
    assert decoded_offsets is not None, "offset_time lost after lcm_decode"
    assert decoded_offsets.dtype == np.uint32
    np.testing.assert_array_equal(decoded_offsets, offset_times)

    decoded_tags = decoded.tags_u8()
    assert decoded_tags is not None, "tag lost after lcm_decode"
    np.testing.assert_array_equal(decoded_tags, tags)

    decoded_lines = decoded.lines_u8()
    assert decoded_lines is not None, "line lost after lcm_decode"
    np.testing.assert_array_equal(decoded_lines, lines)


def test_bounding_box_intersects() -> None:
    """Test bounding_box_intersects method with various scenarios."""
    # Test 1: Overlapping boxes
    pc1 = PointCloud2.from_numpy(np.array([[0, 0, 0], [2, 2, 2]]))
    pc2 = PointCloud2.from_numpy(np.array([[1, 1, 1], [3, 3, 3]]))
    assert pc1.bounding_box_intersects(pc2)
    assert pc2.bounding_box_intersects(pc1)  # Should be symmetric

    # Test 2: Non-overlapping boxes
    pc3 = PointCloud2.from_numpy(np.array([[0, 0, 0], [1, 1, 1]]))
    pc4 = PointCloud2.from_numpy(np.array([[2, 2, 2], [3, 3, 3]]))
    assert not pc3.bounding_box_intersects(pc4)
    assert not pc4.bounding_box_intersects(pc3)

    # Test 3: Touching boxes (edge case - should be True)
    pc5 = PointCloud2.from_numpy(np.array([[0, 0, 0], [1, 1, 1]]))
    pc6 = PointCloud2.from_numpy(np.array([[1, 1, 1], [2, 2, 2]]))
    assert pc5.bounding_box_intersects(pc6)
    assert pc6.bounding_box_intersects(pc5)

    # Test 4: One box completely inside another
    pc7 = PointCloud2.from_numpy(np.array([[0, 0, 0], [3, 3, 3]]))
    pc8 = PointCloud2.from_numpy(np.array([[1, 1, 1], [2, 2, 2]]))
    assert pc7.bounding_box_intersects(pc8)
    assert pc8.bounding_box_intersects(pc7)

    # Test 5: Boxes overlapping only in 2 dimensions (not all 3)
    pc9 = PointCloud2.from_numpy(np.array([[0, 0, 0], [2, 2, 1]]))
    pc10 = PointCloud2.from_numpy(np.array([[1, 1, 2], [3, 3, 3]]))
    assert not pc9.bounding_box_intersects(pc10)
    assert not pc10.bounding_box_intersects(pc9)

    # Test 6: Real-world detection scenario with floating point coordinates
    detection1_points = np.array(
        [[-3.5, -0.3, 0.1], [-3.3, -0.2, 0.1], [-3.5, -0.3, 0.3], [-3.3, -0.2, 0.3]]
    )
    pc_det1 = PointCloud2.from_numpy(detection1_points)

    detection2_points = np.array(
        [[-3.4, -0.25, 0.15], [-3.2, -0.15, 0.15], [-3.4, -0.25, 0.35], [-3.2, -0.15, 0.35]]
    )
    pc_det2 = PointCloud2.from_numpy(detection2_points)

    assert pc_det1.bounding_box_intersects(pc_det2)

    # Test 7: Single point clouds
    pc_single1 = PointCloud2.from_numpy(np.array([[1.0, 1.0, 1.0]]))
    pc_single2 = PointCloud2.from_numpy(np.array([[1.0, 1.0, 1.0]]))
    pc_single3 = PointCloud2.from_numpy(np.array([[2.0, 2.0, 2.0]]))

    # Same point should intersect
    assert pc_single1.bounding_box_intersects(pc_single2)
    # Different points should not intersect
    assert not pc_single1.bounding_box_intersects(pc_single3)

    # Test 8: Empty point clouds
    pc_empty1 = PointCloud2.from_numpy(np.array([]).reshape(0, 3))
    pc_empty2 = PointCloud2.from_numpy(np.array([]).reshape(0, 3))
    PointCloud2.from_numpy(np.array([[1.0, 1.0, 1.0]]))

    # Empty clouds should handle gracefully (Open3D returns inf bounds)
    # This might raise an exception or return False - we should handle gracefully
    try:
        result = pc_empty1.bounding_box_intersects(pc_empty2)
        # If no exception, verify behavior is consistent
        assert isinstance(result, bool)
    except Exception:
        # If it raises an exception, that's also acceptable for empty clouds
        pass


def test_to_rerun_points_mode_is_screen_space() -> None:
    """ "points" must be flat screen-space dots, not the world-space spheres branch."""
    cloud = PointCloud2.from_numpy(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    points = cloud.to_rerun(mode="points", voxel_size=0.05, ui_radius=1.5)
    spheres = cloud.to_rerun(mode="spheres", voxel_size=0.05)

    # Negative radii are UI points in rerun; positive ones are world-space.
    assert points.radii.as_arrow_array().to_pylist() == pytest.approx([-1.5])
    assert spheres.radii.as_arrow_array().to_pylist() == pytest.approx([0.025])


def test_to_rerun_keeps_the_clouds_own_rgb() -> None:
    """An RGBD cloud renders in its own colors; rgb=False falls back to the height ramp."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    cloud = PointCloud2(pointcloud=pcd)

    colored = cloud.to_rerun(mode="points")
    assert colored.colors is not None
    assert colored.class_ids is None

    ramp = cloud.to_rerun(mode="points", rgb=False)
    assert ramp.colors is None
    assert ramp.class_ids is not None


def _grid(half: float, pitch: float = 0.05) -> np.ndarray:
    g = np.stack(np.meshgrid(np.arange(-half, half, pitch), np.arange(-half, half, pitch)), -1)
    return g.reshape(-1, 2)


def _keys(encoded: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for k, v in encoded.items():
        out.add(k)
        if isinstance(v, dict):
            out |= {f"{k}.{kk}" for kk in v}
    return out


LEGEND_KEYS = {
    "frame_id",
    "ts",
    "num_points",
    "nonfinite_points",
    "source_dtype",
    "scalar_rounding_m",
    "window_m",
    "window_m.x",
    "window_m.y",
    "window_m.z",
    "centroid_xy_m",
    "xy_footprint_m2",
    "bounds",
    "bounds.columns",
    "bounds.rows",
    "bounds.omitted_points",
    "bounds.max_extent_m",
}


def _assert_complete_bounds(points: np.ndarray, encoded: dict[str, Any]) -> None:
    """Every stored finite return has an enclosing row within the reported extent."""
    finite = points[np.isfinite(points).all(axis=1)].astype(np.float64)
    table = encoded["bounds"]
    rows = np.asarray(table["rows"])
    covered = np.zeros(len(finite), dtype=bool)

    assert table["columns"] == ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "points"]
    assert table["omitted_points"] == 0
    assert sum(row[6] for row in table["rows"]) == len(finite)
    assert encoded["num_points"] == len(finite) + encoded["nonfinite_points"]
    assert rows.shape[1] == 7
    assert np.isfinite(rows).all()
    assert (rows[:, 6] > 0).all()
    assert (rows[:, 6] == np.floor(rows[:, 6])).all()
    lo, hi = rows[:, :6:2], rows[:, 1:6:2]
    assert (lo <= hi).all()
    extents = np.asarray(table["max_extent_m"])
    assert extents.shape == (3,)
    assert np.isfinite(extents).all()
    assert (extents >= 0).all()
    assert ((hi - lo) <= np.nextafter(extents, np.inf)).all()
    for row_lo, row_hi, count in zip(lo, hi, rows[:, 6], strict=True):
        contained = ((finite >= row_lo) & (finite <= row_hi)).all(axis=1)
        # Outward rounding may enclose neighboring groups as well as this row's returns.
        assert contained.sum() >= count
        covered |= contained
    assert covered.all()


def test_agent_encode_scalars_describe_all_returns() -> None:
    wall = np.stack(
        [np.full(200, 2.0), np.linspace(-1.0, 1.0, 200), np.linspace(0.2, 0.9, 200)],
        axis=1,
    )
    floor = np.stack([np.linspace(-3, 3, 100), np.linspace(-3, 3, 100), np.zeros(100)], axis=1)
    cloud = PointCloud2.from_numpy(np.vstack([wall, floor]), timestamp=12.345)

    encoded = cloud.agent_encode()

    assert encoded["num_points"] == 300
    assert encoded["ts"] == 12.345
    assert encoded["window_m"] == {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": [0.0, 0.9]}
    assert encoded["centroid_xy_m"] == [1.33, 0.0]
    _assert_complete_bounds(cloud.points_f32(), encoded)


@pytest.mark.parametrize(
    "points",
    [np.empty((0, 3)), np.array([[1.0, 2.0, 0.3]]), np.column_stack([_grid(1.0), np.zeros(1600)])],
)
def test_agent_encode_schema_is_present_for_empty_and_nonempty_clouds(points) -> None:
    encoded = PointCloud2.from_numpy(points).agent_encode()

    assert _keys(encoded) == LEGEND_KEYS


def test_agent_encode_empty_cloud() -> None:
    encoded = PointCloud2.from_numpy(np.empty((0, 3)), frame_id="empty_frame").agent_encode()

    assert encoded["frame_id"] == "empty_frame"
    assert encoded["num_points"] == 0
    assert encoded["nonfinite_points"] == 0
    assert encoded["centroid_xy_m"] == []
    assert encoded["window_m"] == {"x": [], "y": [], "z": []}
    assert encoded["bounds"]["rows"] == []
    assert encoded["bounds"]["omitted_points"] == 0
    assert encoded["bounds"]["max_extent_m"] == []
    json.dumps(encoded, allow_nan=False)


@pytest.mark.parametrize(
    ("scale", "offset"),
    [(1.0, 0.0), (1e-9, 0.0), (0.001, -10.0), (10000.0, 1000000.0)],
)
def test_agent_encode_bounds_cover_scaled_translated_stored_returns(scale, offset) -> None:
    xy = np.stack(np.meshgrid(np.arange(15), np.arange(9)), axis=-1).reshape(-1, 2)
    points = np.column_stack([xy, np.linspace(-9, 21, len(xy))]) * scale + offset
    cloud = PointCloud2.from_numpy(points, frame_id="camera_optical", timestamp=20.25)

    encoded = cloud.agent_encode()

    assert encoded["frame_id"] == "camera_optical"
    assert encoded["ts"] == 20.25
    _assert_complete_bounds(cloud.points_f32(), encoded)
    assert len(json.dumps(encoded)) <= PointCloud2.ENCODE_SOFT_CAP


@pytest.mark.parametrize("axis_order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_agent_encode_uses_actual_axes_for_separated_surfaces(axis_order) -> None:
    """A gap remains represented when its normal points along any cloud axis."""
    yz = _grid(0.5, 0.25)
    surfaces = np.vstack([np.column_stack([np.full(len(yz), x), yz]) for x in (-5.0, 5.0)])
    points = surfaces[:, axis_order] + np.array([-100.0, 400.0, -30.0])
    cloud = PointCloud2.from_numpy(points, frame_id="sensor_custom_axes")

    encoded = cloud.agent_encode()

    separation_axis = axis_order.index(0)
    midpoint = [-100.0, 400.0, -30.0][separation_axis]
    rows = encoded["bounds"]["rows"]
    assert encoded["frame_id"] == "sensor_custom_axes"
    assert all(
        row[2 * separation_axis + 1] < midpoint or row[2 * separation_axis] > midpoint
        for row in rows
    )
    for axis, name in enumerate("xyz"):
        assert encoded["window_m"][name] == [points[:, axis].min(), points[:, axis].max()]
    _assert_complete_bounds(cloud.points_f32(), encoded)


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 0.001, 1.0, 10000.0])
def test_agent_encode_preserves_small_spans_and_reports_rounding(scale) -> None:
    cloud = PointCloud2.from_numpy(np.array([[1, 2, -3], [4, 6, 7]]) * scale)
    stored = cloud.points_f32().astype(np.float64)

    encoded = cloud.agent_encode()

    for i, axis in enumerate("xyz"):
        bounds = encoded["window_m"][axis]
        tolerance = encoded["scalar_rounding_m"][i] / 2 + np.spacing(abs(stored[:, i]).max())
        assert bounds == pytest.approx([stored[:, i].min(), stored[:, i].max()], abs=tolerance)
        assert bounds[1] > bounds[0]
    assert encoded["source_dtype"] == "float32"
    _assert_complete_bounds(stored, encoded)


@pytest.mark.parametrize("offset", [1000000.0, 100000000.0])
def test_agent_encode_centroid_stays_within_translated_stored_bounds(offset) -> None:
    points = np.tile(np.array([[offset, -offset, 0], [offset + 8, -offset + 16, 1]]), (10000, 1))
    cloud = PointCloud2.from_numpy(points)
    stored = cloud.points_f32().astype(np.float64)

    encoded = cloud.agent_encode()

    assert encoded["centroid_xy_m"] == pytest.approx(stored[:, :2].mean(axis=0), abs=0.005)
    for i, axis in enumerate("xy"):
        lo, hi = encoded["window_m"][axis]
        assert lo <= encoded["centroid_xy_m"][i] <= hi
    _assert_complete_bounds(stored, encoded)


@pytest.mark.parametrize("cap", [6000, 24000])
@pytest.mark.parametrize("scale,offset", [(1e-12, 0), (1, 1e6), (1e20, 1e30)])
def test_agent_encode_budget_includes_coordinate_precision(monkeypatch, cap, scale, offset) -> None:
    monkeypatch.setattr(PointCloud2, "ENCODE_SOFT_CAP", cap)
    xy = np.stack(np.meshgrid(np.arange(48), np.arange(48)), axis=-1).reshape(-1, 2)
    points = np.column_stack([xy * scale + offset, np.full(len(xy), 0.5)])
    cloud = PointCloud2.from_numpy(points)

    encoded = cloud.agent_encode()

    assert len(json.dumps(encoded, allow_nan=False)) <= cap
    _assert_complete_bounds(cloud.points_f32(), encoded)


@pytest.mark.parametrize("cap", [6000, 24000])
def test_agent_encode_dense_cloud_fits_budget_without_omitting_returns(monkeypatch, cap) -> None:
    monkeypatch.setattr(PointCloud2, "ENCODE_SOFT_CAP", cap)
    rng = np.random.default_rng(7)
    grid = _grid(6.0, 0.08)
    points = np.vstack(
        [
            np.column_stack([grid, np.zeros(len(grid))]),
            rng.uniform([-6, -6, -2], [6, 6, 4], size=(4000, 3)),
        ]
    )
    cloud = PointCloud2.from_numpy(points)

    encoded = cloud.agent_encode()

    assert len(json.dumps(encoded, allow_nan=False)) <= cap
    _assert_complete_bounds(cloud.points_f32(), encoded)


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.ones((1, 3))])
def test_agent_encode_rejects_metadata_larger_than_budget(points) -> None:
    cloud = PointCloud2.from_numpy(points, frame_id="x" * PointCloud2.ENCODE_SOFT_CAP)

    with pytest.raises(ValueError, match="metadata exceeds .* byte budget"):
        cloud.agent_encode()


@pytest.mark.parametrize("cap", [6000, 24000])
def test_agent_encode_is_deterministic_under_return_reordering(monkeypatch, cap) -> None:
    monkeypatch.setattr(PointCloud2, "ENCODE_SOFT_CAP", cap)
    rng = np.random.default_rng(13)
    points = rng.normal(size=(2000, 3)) + np.array([2000.0, -3000.0, 4000.0])
    points = np.vstack([points, points[:100], [[np.nan, 0, 0]]])
    cloud = PointCloud2.from_numpy(points, frame_id="map", timestamp=5.0)
    reordered = PointCloud2.from_numpy(
        points[rng.permutation(len(points))], frame_id="map", timestamp=5.0
    )

    encoded = cloud.agent_encode()

    assert encoded == cloud.agent_encode()
    assert json.dumps(encoded) == json.dumps(reordered.agent_encode())


def test_agent_encode_excludes_and_counts_nonfinite_rows() -> None:
    cloud = PointCloud2.from_numpy(np.array([[1, 2, 3], [np.nan, 0, 0], [0, np.inf, 0]]))

    encoded = cloud.agent_encode()

    assert encoded["num_points"] == 3
    assert encoded["nonfinite_points"] == 2
    assert encoded["window_m"] == {"x": [1, 1], "y": [2, 2], "z": [3, 3]}
    assert encoded["xy_footprint_m2"] == 0.04
    _assert_complete_bounds(cloud.points_f32(), encoded)
    json.dumps(encoded, allow_nan=False)


def test_agent_encode_all_nonfinite_has_empty_geometry() -> None:
    encoded = PointCloud2.from_numpy(np.array([[np.nan, 1, 2]])).agent_encode()

    assert encoded["num_points"] == 1
    assert encoded["nonfinite_points"] == 1
    assert encoded["window_m"] == {"x": [], "y": [], "z": []}
    assert encoded["bounds"]["rows"] == []
    assert encoded["bounds"]["omitted_points"] == 0
    assert encoded["bounds"]["max_extent_m"] == []


def test_agent_encode_nearby_float64_boundaries_conserve_returns() -> None:
    z = np.array([-4, np.nextafter(-2.0, -np.inf), -2, np.nextafter(-2.0, np.inf), 0, 2, 4])
    points = np.column_stack([np.zeros(len(z)), np.zeros(len(z)), z])
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    encoded = cloud.agent_encode()

    _assert_complete_bounds(points, encoded)


def test_agent_encode_close_large_values_preserve_distinct_extrema() -> None:
    points = np.array([[0, 0, 1e16], [0, 0, np.nextafter(1e16, np.inf)]])
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    encoded = cloud.agent_encode()

    assert encoded["window_m"]["z"] == points[:, 2].tolist()
    _assert_complete_bounds(points, encoded)


@pytest.mark.parametrize("scale", [1e-9, 1.0, 10000.0])
def test_agent_encode_separated_point_groups_preserve_gaps(scale) -> None:
    points = np.array([[0, 0, 7], [0.1, 0, 7], [1, 0, 7], [1.1, 0, 7], [4, 0, 7]]) * scale
    cloud = PointCloud2.from_numpy(points)

    encoded = cloud.agent_encode()

    rows = encoded["bounds"]["rows"]
    for gap_center in (0.5 * scale, 2.0 * scale):
        assert all(row[1] < gap_center or row[0] > gap_center for row in rows)
    _assert_complete_bounds(cloud.points_f32(), encoded)


@pytest.mark.parametrize(
    "points", [np.array([[-1e308, 0, 0], [1e308, 0, 0]]), np.array([[0, 0, 0], [0, 0, 5e-324]])]
)
def test_agent_encode_rejects_unrepresentable_numeric_ranges(points) -> None:
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    with pytest.raises(ValueError, match="numeric range"):
        cloud.agent_encode()


def test_agent_encode_preserves_stored_float64_coordinates() -> None:
    cloud = PointCloud2()
    points = np.array([[1e8, 0, 0], [1e8 + 0.001, 0, 0]])
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    encoded = cloud.agent_encode()

    assert encoded["source_dtype"] == "float64"
    assert encoded["window_m"]["x"] == points[:, 0].tolist()
    assert encoded["window_m"]["x"][1] > encoded["window_m"]["x"][0]
    _assert_complete_bounds(points, encoded)


def test_agent_encode_large_finite_centroid_serializes_without_overflow() -> None:
    points = np.tile(np.array([[0.0, 0.0, 0.0], [1e307, 1e307, 0.0]]), (100, 1))
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    encoded = cloud.agent_encode()

    assert encoded["centroid_xy_m"] == pytest.approx([5e306, 5e306], rel=1e-14)
    _assert_complete_bounds(points, encoded)
    assert len(json.dumps(encoded, allow_nan=False)) <= PointCloud2.ENCODE_SOFT_CAP


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("coordinate", [-1e308, 1e308])
def test_agent_encode_rejects_overflowing_footprint_coordinates(axis, coordinate) -> None:
    points = np.zeros((2, 3))
    points[1, axis] = coordinate
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    with pytest.raises(ValueError, match="numeric range"):
        cloud.agent_encode()


def test_agent_encode_centroid_rounding_is_independent_of_return_order() -> None:
    x = np.r_[np.zeros(18999), np.full(1000, 0.01), 90.0]
    points = np.column_stack([x, np.zeros((len(x), 2))])
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)
    descending = PointCloud2()
    descending.pointcloud_tensor.point["positions"] = o3c.Tensor(points[::-1].copy())

    encoded = cloud.agent_encode()

    assert encoded == descending.agent_encode()
    assert encoded["centroid_xy_m"][0] == pytest.approx(0.005, abs=0.005)
