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


def _keys(encoded: dict[str, object]) -> set[str]:
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
    "floor_footprint_m2",
    "raster",
    "raster.cell_m",
    "raster.origin_xy_m",
    "raster.z_step_m",
    "raster.z_min_m",
    "raster.z_error_m",
    "raster.rows",
    "boxes",
    "boxes.z_m",
    "boxes.omitted_count",
    "boxes.xmin:xmax@ymin:ymax",
}


def test_agent_encode_scalars_are_exact() -> None:
    """Full-cloud bounds include returns outside the explicitly encoded z interval."""
    wall = np.stack(
        [
            np.full(200, 2.0),
            np.linspace(-1.0, 1.0, 200),
            np.linspace(0.2, 0.9, 200),
        ],
        axis=1,
    )
    floor = np.stack([np.linspace(-3, 3, 100), np.linspace(-3, 3, 100), np.zeros(100)], axis=1)
    encoded = PointCloud2.from_numpy(np.vstack([wall, floor]), timestamp=12.345).agent_encode()

    assert encoded["num_points"] == 300
    assert encoded["ts"] == 12.345
    assert encoded["window_m"] == {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": [0.0, 0.9]}
    boxes = encoded["boxes"]
    assert isinstance(boxes, dict)
    assert boxes["z_m"] == [0.15, 1.0]
    # body-height boxes see the wall only: x pinned at 2.0, floor (z=0) excluded
    assert all(b.startswith("2.0@") for b in str(boxes["xmin:xmax@ymin:ymax"]).split(","))
    assert encoded["centroid_xy_m"] == [1.33, 0.0]  # 200 wall pts at x=2, 100 floor at mean 0


def test_agent_encode_every_key_on_every_frame() -> None:
    """A reader indexes the encoding by key; a frame that drops one is a KeyError
    in the agent's code. The empty cloud carries the same keys, empty."""
    empty = PointCloud2.from_numpy(np.zeros((0, 3))).agent_encode()
    assert empty["num_points"] == 0
    assert _keys(empty) == LEGEND_KEYS
    one = PointCloud2.from_numpy(np.array([[1.0, 2.0, 0.3]])).agent_encode()
    assert _keys(one) == LEGEND_KEYS
    floor = PointCloud2.from_numpy(np.column_stack([_grid(3.0), np.zeros(14400)])).agent_encode()
    assert _keys(floor) == LEGEND_KEYS
    raster = floor["raster"]
    assert isinstance(raster, dict)
    assert raster["rows"]
    widths = {len(row.split(maxsplit=1)[1]) for row in raster["rows"]}
    assert len(widths) == 1


def test_agent_encode_empty_cloud() -> None:
    encoded = PointCloud2.from_numpy(np.zeros((0, 3))).agent_encode()
    assert encoded["num_points"] == 0
    assert encoded["centroid_xy_m"] == []
    assert encoded["window_m"] == {"x": [], "y": [], "z": []}
    raster = encoded["raster"]
    assert isinstance(raster, dict)
    assert raster["rows"] == []


@pytest.mark.parametrize(
    ("scale", "offset"),
    [(1.0, 0.0), (1e-9, 0.0), (0.001, -10.0), (10000.0, 1000000.0)],
)
def test_agent_encode_raster_quantization_round_trips(scale, offset) -> None:
    """Every occupied cell retains its z extrema within the advertised error."""
    xy = np.stack(np.meshgrid(np.arange(15), np.arange(9)), axis=-1).reshape(-1, 2)
    z = np.linspace(-9, 21, len(xy))
    cloud = PointCloud2.from_numpy(
        np.column_stack([xy * scale, z * scale + offset]), frame_id="camera_optical"
    )
    pts = cloud.points_f32().astype(np.float64)

    encoded = cloud.agent_encode()
    raster = encoded["raster"]
    alphabet = PointCloud2._RASTER_ALPHABET
    origin = np.array(raster["origin_xy_m"])
    cell = raster["cell_m"]
    indices = np.floor((pts[:, :2] - origin) / cell).astype(int)
    rows = list(reversed(raster["rows"]))
    tolerance = raster["z_error_m"] + np.finfo(float).eps * max(1, np.abs(pts[:, 2]).max()) * 4

    assert encoded["frame_id"] == "camera_optical"
    assert len(rows) <= PointCloud2._RASTER_MAX_CELLS
    for j, row in enumerate(rows):
        label, cells = row.split(maxsplit=1)
        assert float(label) == pytest.approx(origin[1] + j * cell, abs=cell * 1e-8)
        assert len(cells) // 2 <= PointCloud2._RASTER_MAX_CELLS
        for i in range(len(cells) // 2):
            pair = cells[2 * i : 2 * i + 2]
            selected = pts[(indices[:, 0] == i) & (indices[:, 1] == j), 2]
            if len(selected) == 0:
                assert pair == ".."
                continue
            decoded = [raster["z_min_m"] + alphabet.index(c) * raster["z_step_m"] for c in pair]
            assert decoded == pytest.approx([selected.min(), selected.max()], abs=tolerance)
    assert len(json.dumps(encoded)) <= PointCloud2.ENCODE_SOFT_CAP


def test_agent_encode_raster_cell_rule() -> None:
    """0.25 m for a single sweep, 0.5 m once a fused map would pass 48 cells."""
    small = PointCloud2.from_numpy(np.column_stack([_grid(3.0, 0.1), np.zeros(3600)]))
    large = PointCloud2.from_numpy(np.column_stack([_grid(7.5, 0.1), np.zeros(22500)]))
    small_raster = small.agent_encode()["raster"]
    large_raster = large.agent_encode()["raster"]
    assert isinstance(small_raster, dict) and isinstance(large_raster, dict)
    assert small_raster["cell_m"] == 0.25
    assert small_raster["origin_xy_m"] == [-3.0, -3.0]
    assert len(small_raster["rows"]) == 24
    assert large_raster["cell_m"] == 0.5
    assert len(large_raster["rows"]) == 30


def test_agent_encode_raster_single_return_is_min_equals_max() -> None:
    raster = PointCloud2.from_numpy(np.array([[0.1, 0.1, 0.62]])).agent_encode()["raster"]
    assert isinstance(raster, dict)
    (row,) = raster["rows"]
    pair = row.split(" ", 1)[1]
    assert len(pair) == 2
    assert pair[0] == pair[1]
    decoded = raster["z_min_m"] + PointCloud2._RASTER_ALPHABET.index(pair[0]) * raster["z_step_m"]
    assert decoded == pytest.approx(0.62)
    assert raster["z_error_m"] == 0.0


def test_agent_encode_stays_within_prompt_budget() -> None:
    """The encoding is prompt text, so its size is a hard product constraint.

    A busy room -- floor, four walls, scattered clutter -- is the verbose case.
    The encoder trims its own tail rather than letting a dense frame run over.
    """
    rng = np.random.default_rng(7)
    grid = _grid(6.0, 0.04)
    parts = [np.column_stack([grid, np.zeros(len(grid))])]
    for edge in (-6.0, 6.0):
        span = np.arange(-6, 6, 0.02)
        height = rng.uniform(0.15, 1.0, span.size)
        parts.append(np.column_stack([np.full(span.size, edge), span, height]))
        parts.append(np.column_stack([span, np.full(span.size, edge), height]))
    parts.append(
        np.column_stack(
            [rng.uniform(-6, 6, 4000), rng.uniform(-6, 6, 4000), rng.uniform(0.15, 1.0, 4000)]
        )
    )

    encoded = PointCloud2.from_numpy(np.vstack(parts)).agent_encode()

    assert len(json.dumps(encoded)) <= PointCloud2.ENCODE_SOFT_CAP
    assert _keys(encoded) == LEGEND_KEYS


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
    assert encoded["raster"]["cell_m"] < scale


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


@pytest.mark.parametrize("scale,offset", [(1e-12, 0), (1, 1e6), (1e20, 1e30)])
def test_agent_encode_budget_includes_coordinate_labels(scale, offset) -> None:
    xy = np.stack(np.meshgrid(np.arange(48), np.arange(48)), axis=-1).reshape(-1, 2)
    pts = np.column_stack([xy * scale + offset, np.full(len(xy), 0.5)])

    encoded = PointCloud2.from_numpy(pts).agent_encode()

    assert len(json.dumps(encoded)) <= PointCloud2.ENCODE_SOFT_CAP
    assert len(encoded["raster"]["rows"]) <= PointCloud2._RASTER_MAX_CELLS


def test_agent_encode_excludes_and_counts_nonfinite_rows() -> None:
    cloud = PointCloud2.from_numpy(np.array([[1, 2, 3], [np.nan, 0, 0], [0, np.inf, 0]]))

    encoded = cloud.agent_encode()

    assert encoded["num_points"] == 3
    assert encoded["nonfinite_points"] == 2
    assert encoded["window_m"] == {"x": [1, 1], "y": [2, 2], "z": [3, 3]}
    assert encoded["floor_footprint_m2"] == 0.04
    json.dumps(encoded, allow_nan=False)


def test_agent_encode_all_nonfinite_has_empty_geometry() -> None:
    encoded = PointCloud2.from_numpy(np.array([[np.nan, 1, 2]])).agent_encode()

    assert encoded["nonfinite_points"] == 1
    assert encoded["window_m"] == {"x": [], "y": [], "z": []}
    assert encoded["raster"]["rows"] == []


def test_agent_encode_reports_omitted_complete_boxes(monkeypatch) -> None:
    xy = np.stack(np.meshgrid(np.arange(20), np.arange(20)), axis=-1).reshape(-1, 2)
    cloud = PointCloud2.from_numpy(np.column_stack([xy, np.full(len(xy), 0.5)]))
    monkeypatch.setattr(PointCloud2, "ENCODE_SOFT_CAP", 20000)
    complete = cloud.agent_encode()
    complete_boxes = complete["boxes"]["xmin:xmax@ymin:ymax"].split(",")
    monkeypatch.setattr(PointCloud2, "ENCODE_SOFT_CAP", 1000)

    encoded = cloud.agent_encode()
    packed_boxes = encoded["boxes"]["xmin:xmax@ymin:ymax"].split(",")

    assert encoded["boxes"]["omitted_count"] > 0
    assert len(packed_boxes) + encoded["boxes"]["omitted_count"] == len(complete_boxes)
    assert packed_boxes == complete_boxes[: len(packed_boxes)]
    assert len(json.dumps(encoded)) <= 1000


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


@pytest.mark.self_hosted
def test_agent_encode_fused_map_fits_the_cap() -> None:
    """A whole recording fused into one map is the largest cloud the agent
    reads; it must still come back in one readout, at a coarser cell."""
    from dimos.evals.suites.lib.generate import _dataset
    from dimos.mapping.voxels.module import VoxelMapTransformer
    from dimos.memory.transform import downsample

    with _dataset("go2_china_office") as store:
        fused = (
            store.streams.lidar.range_time(0, 138)
            .transform(downsample(6))
            .transform(VoxelMapTransformer(voxel_size=0.05, device="CPU:0", emit_every=0))
            .last()
            .data
        )
    encoded = fused.agent_encode()
    raster = encoded["raster"]
    assert isinstance(raster, dict)
    assert len(json.dumps(encoded)) <= PointCloud2.ENCODE_SOFT_CAP
    assert raster["cell_m"] == 0.5
    assert len(raster["rows"]) <= 48
    assert _keys(encoded) == LEGEND_KEYS
