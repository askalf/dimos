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


def test_agent_encode_complete_metric_summary() -> None:
    points = np.array([[-3, -3, 0], [3, 3, 0], [2, -1, 0.5], [2, 1, 1]])
    cloud = PointCloud2.from_numpy(points, frame_id="sensor_native", timestamp=12.345)

    encoded = cloud.agent_encode()

    assert encoded["frame_id"] == "sensor_native"
    assert encoded["ts"] == 12.345
    assert encoded["num_points"] == 4
    assert encoded["window_m"] == {"x": [-3, 3], "y": [-3, 3], "z": [0, 1]}
    assert encoded["centroid_xy_m"] == [1, 0]
    assert sum(row[6] for row in encoded["sections"]["rows"]) == 4
    assert encoded["sections"]["omitted_points"] == 0


def test_agent_encode_empty_and_nonfinite_geometry() -> None:
    empty = PointCloud2.from_numpy(np.zeros((0, 3))).agent_encode()
    invalid = PointCloud2.from_numpy(np.array([[np.nan, 1, 2]])).agent_encode()

    for encoded in [empty, invalid]:
        assert encoded["window_m"] == {"x": [], "y": [], "z": []}
        assert encoded["sections"]["rows"] == []
        assert encoded["axis_gaps_m"] == {"x": [], "y": [], "z": []}
        json.dumps(encoded, allow_nan=False)
    assert empty["num_points"] == 0
    assert invalid["nonfinite_points"] == invalid["num_points"] == 1


def test_agent_encode_counts_nonfinite_returns_without_using_their_geometry() -> None:
    points = np.array([[1, 2, 3], [np.nan, 0, 0], [0, np.inf, 0]])

    encoded = PointCloud2.from_numpy(points).agent_encode()

    assert encoded["num_points"] == 3
    assert encoded["nonfinite_points"] == 2
    assert encoded["window_m"] == {"x": [1, 1], "y": [2, 2], "z": [3, 3]}
    assert encoded["sections"]["rows"] == [[1, 1, 2, 2, 3, 3, 1]]
    json.dumps(encoded, allow_nan=False)


@pytest.mark.parametrize("scale,offset", [(2**-30, 0), (1, 1e6), (2**20, -1e8)])
def test_agent_encode_sections_preserve_scaled_translated_returns(scale, offset) -> None:
    points = np.column_stack([np.arange(9), np.zeros(9), np.arange(-4, 5)]) * scale + offset
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    encoded = cloud.agent_encode()

    represented = np.zeros(len(points), dtype=int)
    for row in encoded["sections"]["rows"]:
        lo, hi = np.array(row[:6:2]), np.array(row[1:6:2])
        contained = ((points >= lo) & (points <= hi)).all(axis=1)
        assert contained.sum() == row[6] == 1
        represented += contained
    assert represented.tolist() == [1] * len(points)
    assert sum(row[6] for row in encoded["sections"]["rows"]) == len(points)
    assert encoded["sections"]["omitted_points"] == 0


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 0.001, 1.0, 10000.0])
def test_agent_encode_preserves_native_axis_extrema_and_declared_precision(scale) -> None:
    cloud = PointCloud2.from_numpy(np.array([[1, 2, -3], [4, 6, 7]]) * scale)
    stored = cloud.points_f32().astype(np.float64)

    encoded = cloud.agent_encode()

    for i, axis in enumerate("xyz"):
        bounds = encoded["window_m"][axis]
        tolerance = encoded["scalar_rounding_m"][i] / 2 + np.spacing(abs(stored[:, i]).max())
        assert bounds == pytest.approx([stored[:, i].min(), stored[:, i].max()], abs=tolerance)
        assert bounds[1] > bounds[0]
    assert encoded["source_dtype"] == "float32"


def test_agent_encode_projected_gaps_use_all_returns_and_stored_precision() -> None:
    points = np.array([[1e8 + x, 0, -2] for x in [0, 0.001, 0.001, 0.002, 0.01]])
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    encoded = cloud.agent_encode()

    assert encoded["axis_gaps_m"]["x"][0] == [points[3, 0], points[4, 0]]
    assert len(encoded["axis_gaps_m"]["x"]) == 3
    assert encoded["axis_gaps_m"]["y"] == encoded["axis_gaps_m"]["z"] == []
    assert encoded["source_dtype"] == "float64"
    assert encoded["window_m"]["x"] == [points[0, 0], points[4, 0]]


def test_agent_encode_sections_keep_disconnected_heights() -> None:
    points = np.array([[x, 0, z] for x in [0, 0.01, 0.02] for z in [-3, 4]])

    encoded = PointCloud2.from_numpy(points).agent_encode()

    rows = encoded["sections"]["rows"]
    assert [row[4:7] for row in rows] == [[-3, -3, 3], [4, 4, 3]]
    assert all(row[1] - row[0] < 0.04 for row in rows)


def test_agent_encode_sections_leave_large_x_gaps_open() -> None:
    points = np.array([[x, 0, 7] for x in [0, 0.01, 1, 1.01, 4]])

    encoded = PointCloud2.from_numpy(points).agent_encode()

    rows = encoded["sections"]["rows"]
    assert [row[6] for row in rows] == [2, 2, 1]
    assert rows[0][1] < rows[1][0] < rows[1][1] < rows[2][0]
    assert encoded["sections"]["section_m"] < 0.99


def test_agent_encode_budget_coarsens_without_discarding_returns(monkeypatch) -> None:
    axis = np.arange(24) * 0.25
    points = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3)
    cloud = PointCloud2.from_numpy(points)

    encoded = cloud.agent_encode()
    monkeypatch.setattr(PointCloud2, "ENCODE_SOFT_CAP", 24000)
    refined = cloud.agent_encode()

    assert len(json.dumps(encoded)) <= 6000
    assert len(json.dumps(refined)) <= 24000
    assert encoded["sections"]["section_m"] > refined["sections"]["section_m"]
    for summary in [encoded, refined]:
        assert sum(row[6] for row in summary["sections"]["rows"]) == len(points)
        assert summary["sections"]["omitted_points"] == 0
        tolerance = 2 * np.array(summary["scalar_rounding_m"])
        for row in summary["sections"]["rows"]:
            lo, hi = np.array(row[:6:2]), np.array(row[1:6:2])
            assert np.all((hi - lo)[1:] <= summary["sections"]["section_m"] + tolerance[1:])


def test_agent_encode_is_independent_of_return_order() -> None:
    rng = np.random.default_rng(17)
    points = rng.uniform([-10, -2, -0.2], [5, 4, 2], (3500, 3))
    cloud = PointCloud2.from_numpy(points)
    permuted = PointCloud2.from_numpy(points[rng.permutation(len(points))])

    assert cloud.agent_encode() == permuted.agent_encode()


@pytest.mark.parametrize("offset", [1000000.0, 100000000.0])
def test_agent_encode_centroid_stays_within_translated_stored_bounds(offset) -> None:
    pts = np.tile(np.array([[offset, -offset, 0], [offset + 8, -offset + 16, 1]]), (10000, 1))
    cloud = PointCloud2.from_numpy(pts)
    stored = cloud.points_f32().astype(np.float64)

    encoded = cloud.agent_encode()

    assert encoded["centroid_xy_m"] == pytest.approx(stored[:, :2].mean(axis=0), abs=0.005)
    for i, axis in enumerate("xy"):
        lo, hi = encoded["window_m"][axis]
        assert lo <= encoded["centroid_xy_m"][i] <= hi


@pytest.mark.parametrize(
    "points", [np.array([[-1e308, 0, 0], [1e308, 0, 0]]), np.array([[0, 0, 0], [0, 0, 5e-324]])]
)
def test_agent_encode_rejects_unrepresentable_numeric_ranges(points) -> None:
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    with pytest.raises(ValueError, match="numeric range"):
        cloud.agent_encode()


@pytest.mark.parametrize("axes", [(2, 0, 1), (1, 2, 0), (0, 2, 1)])
def test_agent_encode_axis_permutations_describe_native_coordinates(axes) -> None:
    original = np.array([[0, -4, 10], [0.125, 3, 20], [1, 8, 30], [4, 9, 40]])
    points = original[:, axes]
    cloud = PointCloud2.from_numpy(points, frame_id="optical_native")
    original_encoding = PointCloud2.from_numpy(original).agent_encode()

    encoded = cloud.agent_encode()

    assert encoded["frame_id"] == "optical_native"
    for axis, source in zip("xyz", axes, strict=True):
        assert encoded["axis_gaps_m"][axis] == original_encoding["axis_gaps_m"]["xyz"[source]]
        assert encoded["window_m"][axis] == original_encoding["window_m"]["xyz"[source]]
    covered = np.zeros(len(points), dtype=int)
    for row in encoded["sections"]["rows"]:
        lo, hi = np.array(row[:6:2]), np.array(row[1:6:2])
        contains = ((points >= lo) & (points <= hi)).all(axis=1)
        assert contains.sum() == row[6]
        covered += contains
    assert covered.tolist() == [1] * len(points)


def test_agent_encode_centroid_rounding_is_independent_of_return_order() -> None:
    x = np.r_[np.zeros(18999), np.full(1000, 0.01), 90.0]
    points = np.column_stack([x, np.zeros((len(x), 2))])
    forward, reversed_cloud = PointCloud2(), PointCloud2()
    forward.pointcloud_tensor.point["positions"] = o3c.Tensor(points)
    reversed_cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points[::-1].copy())

    assert forward.agent_encode() == reversed_cloud.agent_encode()


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.ones((1, 3))])
def test_agent_encode_rejects_metadata_larger_than_byte_budget(points) -> None:
    cloud = PointCloud2.from_numpy(points, frame_id="x" * PointCloud2.ENCODE_SOFT_CAP)

    with pytest.raises(ValueError, match="metadata exceeds .* byte budget"):
        cloud.agent_encode()


def test_agent_encode_centroid_handles_large_finite_coordinate_sums() -> None:
    points = np.tile(np.array([[0.0, 0.0, 0.0], [1e307, 1e307, 0.0]]), (100, 1))
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(points)

    encoded = cloud.agent_encode()

    assert encoded["centroid_xy_m"] == pytest.approx([5e306, 5e306])
    assert sum(row[6] for row in encoded["sections"]["rows"]) == len(points)
    json.dumps(encoded, allow_nan=False)


def test_agent_encode_rejects_unsupported_footprint_coordinate_range() -> None:
    cloud = PointCloud2()
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(np.array([[1e308, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="XY footprint exceeds .* numeric range"):
        cloud.agent_encode()


def test_agent_encode_rejects_section_overflow_under_metadata_pressure(monkeypatch) -> None:
    monkeypatch.setattr(PointCloud2, "ENCODE_SOFT_CAP", 6000)
    cloud = PointCloud2(frame_id="x" * 5470, ts=0.0)
    cloud.pointcloud_tensor.point["positions"] = o3c.Tensor(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7e308]])
    )

    # An infinite section merges both rows and fits this budget, but is invalid JSON.
    with pytest.raises(ValueError, match="sections exceed .* numeric range"):
        cloud.agent_encode()
