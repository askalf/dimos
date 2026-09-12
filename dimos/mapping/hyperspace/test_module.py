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

"""Ingest and query through a real memory store, with a stub model: frames
taken from known poses around a planted object go in as keyframes + patch
vectors, and asking for the object lights up its voxel."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import typer

from dimos.mapping.hyperspace import cli, patches as hs, segmenter as seg
from dimos.mapping.hyperspace.ingest import (
    COMPLETE_STREAM,
    KEYFRAME_STREAM,
    PATCH_STREAM,
    IngestConfig,
    PatchIngestor,
)
from dimos.mapping.hyperspace.query import HyperspaceQuery
from dimos.mapping.hyperspace.segments import SEGMENT_STREAM
from dimos.memory.store.sqlite import SqliteStore
from dimos.models.embedding.base import Embedding
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.std_msgs.String import String
from dimos.msgs.tf2_msgs.TFMessage import TFMessage

SIDE = 8  # 8x8 patch grid: small, but the same code path as 24x24
DIM = 8
WIDTH, HEIGHT = 64, 48
CAMERA = "camera_optical"
WORLD = "odom"
OBJECT = np.array([3.0, 2.0, 0.5])


class StubModel:
    """A 'model' whose patch under the object points along axis 0, everything
    else along axis 1, and whose text for "object" is axis 0."""

    patches_per_side = SIDE
    dim = DIM
    # A real model always names itself; the stream of its vectors is named after it.
    specs = ["stub"]
    tags = ["stub"]

    def __init__(self) -> None:
        self.pixel: tuple[float, float] | None = None

    def embed_patches(self, image: Image) -> list[np.ndarray]:
        grid = np.zeros((SIDE * SIDE, DIM), dtype=np.float32)
        grid[:, 1] = 1.0
        assert self.pixel is not None
        u, v = self.pixel
        col, row = int(u * SIDE / WIDTH), int(v * SIDE / HEIGHT)
        grid[row * SIDE + col] = 0.0
        grid[row * SIDE + col, 0] = 1.0
        return [grid]

    @staticmethod
    def embed_text(text: str) -> np.ndarray:
        vector = np.zeros(DIM, dtype=np.float32)
        vector[0 if text == "object" else 1] = 1.0
        return vector


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = position
    return pose


def quaternion_of(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = matrix
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return x, y, z, w


def camera_info() -> CameraInfo:
    info = CameraInfo(width=WIDTH, height=HEIGHT, distortion_model="plumb_bob", frame_id=CAMERA)
    info.set_K_matrix(np.array([[48.0, 0.0, 32.0], [0.0, 48.0, 24.0], [0.0, 0.0, 1.0]]))
    return info


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    memory = SqliteStore(path=str(tmp_path / "hyperspace.db"))
    memory.start()
    yield memory
    memory.stop()


def fill(store: SqliteStore, poses: list[np.ndarray], *, with_depth: bool = True) -> PatchIngestor:
    model = StubModel()
    config = IngestConfig(
        gate=hs.KeyframeGateConfig(
            lookahead=0,
            # Every distinct pose is a keyframe here: these tests are about what an
            # ingest writes, not about the tuned novelty of a real recording.
            novelty_threshold=0.0,
            patch_novelty_threshold=None,
            max_angular_velocity=None,
            max_dark_fraction=None,
            min_interval=None,
        ),
        min_frame_interval_s=0.0,
    )
    ingestor = PatchIngestor(store, model, config)  # type: ignore[arg-type]
    ingestor.add_camera_info(camera_info())
    for index, pose in enumerate(poses):
        ts = 10.0 + index
        x, y, z, w = quaternion_of(pose[:3, :3])
        transform = Transform(
            translation=Vector3(*pose[:3, 3]),
            rotation=Quaternion(x, y, z, w),
            frame_id=WORLD,
            child_frame_id=CAMERA,
            ts=ts,
        )
        ingestor.add_tf(TFMessage(transform), ts=ts)
        local = np.linalg.inv(pose) @ np.append(OBJECT, 1.0)
        depth = float(local[2])
        model.pixel = (local[0] / local[2] * 48.0 + 32.0, local[1] / local[2] * 48.0 + 24.0)
        if with_depth:
            ingestor.add_depth(
                Image.from_numpy(
                    np.full((HEIGHT, WIDTH), int(depth * 1000), dtype=np.uint16),
                    frame_id=CAMERA,
                    ts=ts,
                )
            )
        ingestor.add_image(
            Image.from_numpy(
                np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8), frame_id=CAMERA, ts=ts
            )
        )
    ingestor.flush()
    return ingestor


def ring(count: int, radius: float) -> list[np.ndarray]:
    return [
        look_at(OBJECT + np.array([radius * math.cos(a), radius * math.sin(a), 0.3]), OBJECT)
        for a in (i / count * math.tau for i in range(count))
    ]


def voxel_of(point: np.ndarray, size: float) -> tuple[int, ...]:
    return tuple(int(v) for v in np.floor(point / size))


def test_ingest_writes_keyframes_and_one_vector_per_patch(store: SqliteStore) -> None:
    ingestor = fill(store, ring(3, 2.5))
    assert ingestor.stats["kept"] == 3
    assert store.stream(KEYFRAME_STREAM, dict).count() == 3
    assert store.stream(cli.patch_stream_for("", "stub"), dict).count() == 3 * SIDE * SIDE
    first = store.stream(KEYFRAME_STREAM, dict).order_by("ts").first()
    assert first.data["camera_frame"] == CAMERA
    assert first.data["grid"].shape == (SIDE * SIDE, DIM)
    assert np.isfinite(first.data["patch_depth"]).all()


def test_query_lights_up_the_object_voxel(store: SqliteStore) -> None:
    fill(store, ring(3, 2.5))
    engine = HyperspaceQuery(
        store,
        StubModel.embed_text,
        hs.QueryConfig(hot_threshold=0.3, background_prompts=["background"]),
        world_frame=WORLD,
        voxel_size=0.1,
    )
    answer = engine.answer("object", 1)
    assert answer["voxels"] > 0, answer["stats"]
    assert answer["stats"]["keyframes_placed"] == 3
    best = np.asarray(answer["best"][0]["xyz"])
    # An 8 px patch at 2.5 m is ~0.4 m; the peak sits within that of the point.
    assert np.abs(best - OBJECT).max() <= 0.5, best
    # Only the object's patch is hot in each frame: one hot patch per keyframe.
    assert answer["stats"]["hot_patches"] == 3


class CountingStream:
    """Forwards to a real stream, tallying the observations pulled from it."""

    def __init__(self, inner: object, tally: list[int]) -> None:
        self.inner, self.tally = inner, tally

    def __getattr__(self, name: str) -> object:
        attribute = getattr(self.inner, name)
        if not callable(attribute):
            return attribute

        def call(*args: object, **kwargs: object) -> object:
            result = attribute(*args, **kwargs)
            return CountingStream(result, self.tally) if hasattr(result, "__iter__") else result

        return call

    def __iter__(self):  # type: ignore[no-untyped-def]
        for observation in self.inner:
            self.tally[0] += 1
            yield observation


def test_tf_is_read_once_not_on_every_query(store: SqliteStore) -> None:
    ingestor = fill(store, ring(3, 2.5))
    engine = HyperspaceQuery(store, StubModel.embed_text, hs.QueryConfig(), WORLD, 0.1)
    engine.read_tf()
    tally = [0]
    original = store.stream
    store.stream = lambda *a, **k: CountingStream(original(*a, **k), tally)  # type: ignore[method-assign]
    try:
        engine.read_tf()
        assert tally[0] <= 1, "an unchanged tf stream should not be re-scanned"
        ingestor.add_tf(
            TFMessage(
                Transform(
                    translation=Vector3(9.0, 0.0, 0.0),
                    rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                    frame_id=WORLD,
                    child_frame_id="extra",
                    ts=99.0,
                )
            ),
            ts=99.0,
        )
        tally[0] = 0
        engine.read_tf()
        # The one new observation is read (plus the row the scan stops on).
        assert tally[0] <= 2, tally[0]
        assert engine.tf.get(WORLD, "extra", 99.0, warn=False) is not None
    finally:
        store.stream = original  # type: ignore[method-assign]


def test_live_transforms_land_in_the_buffer_the_answers_read(store: SqliteStore) -> None:
    fill(store, ring(3, 2.5))
    engine = HyperspaceQuery(store, StubModel.embed_text, hs.QueryConfig(), WORLD, 0.1)
    engine.placer(WORLD)  # reads what the store holds
    # What Hyperspace.handle_tf does with a transform published while running.
    engine.tf.receive_tfmessage(
        TFMessage(
            Transform(
                translation=Vector3(9.0, 0.0, 0.0),
                rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                frame_id=WORLD,
                child_frame_id="extra",
                ts=99.0,
            )
        )
    )
    assert engine.tf.get(WORLD, "extra", 99.0, warn=False) is not None


def test_scene_voxels_come_from_depth_thumbnails(store: SqliteStore) -> None:
    fill(store, ring(3, 2.5))
    engine = HyperspaceQuery(store, StubModel.embed_text, hs.QueryConfig(), WORLD, 0.1)
    scene = engine.scene_voxels(min_samples=1)
    assert scene
    centres = (np.asarray([i for i, _ in scene], dtype=float) + 0.5) * 0.1
    # A flat depth wall at the object's range: its voxels sit around the object.
    assert np.linalg.norm(centres.mean(axis=0) - OBJECT) < 1.5


def test_without_depth_nothing_is_placed(store: SqliteStore) -> None:
    fill(store, ring(3, 2.5), with_depth=False)
    engine = HyperspaceQuery(
        store,
        StubModel.embed_text,
        hs.QueryConfig(hot_threshold=0.3, background_prompts=["background"]),
        WORLD,
        0.1,
    )
    answer = engine.answer("object", 1)
    assert answer["voxels"] == 0
    assert answer["stats"]["hot_patches_without_depth"] == answer["stats"]["hot_patches"] == 3


def add_segment(store: SqliteStore, pose: np.ndarray, ts: float, name: str, cell: int) -> None:
    """One segment record covering ``cell`` of the grid at the object's depth."""
    depth = float((np.linalg.inv(pose) @ np.append(OBJECT, 1.0))[2])
    segment = seg.Segment(
        label=1,
        name=name,
        confidence=0.8,
        area=48,
        bbox=(0, 0, 8, 6),
        depth_fraction=1.0,
        flat_fraction=0.0,
        rle=[],
    )
    camera = hs.Intrinsics(width=WIDTH, height=HEIGHT, fx=48.0, fy=48.0, cx=32.0, cy=24.0)
    store.stream(SEGMENT_STREAM, dict).append(
        seg.segment_record(
            segment,
            camera_frame=CAMERA,
            ts=ts,
            width=WIDTH,
            height=HEIGHT,
            cells=[[cell, 1.0, depth]],
            grid=(SIDE, SIDE),
            intrinsics=camera.__dict__,
        ),
        ts=ts,
        tags={"camera_frame": CAMERA, "name": name},
        embedding=Embedding(vector=StubModel.embed_text(name)),
    )


def test_segments_add_a_red_channel_on_top_of_the_patches(store: SqliteStore) -> None:
    poses = ring(3, 2.5)
    ingestor = fill(store, poses)
    for index, pose in enumerate(poses):
        local = np.linalg.inv(pose) @ np.append(OBJECT, 1.0)
        u, v = local[0] / local[2] * 48.0 + 32.0, local[1] / local[2] * 48.0 + 24.0
        cell = int(v * SIDE / HEIGHT) * SIDE + int(u * SIDE / WIDTH)
        add_segment(store, pose, 10.0 + index, "object", cell)  # the object's own cell
        add_segment(store, pose, 10.0 + index, "other", (cell + 3) % (SIDE * SIDE))
    # One stray "object" segment on a cell no patch lit: a segment-only region.
    add_segment(store, poses[0], 10.0, "object", (cell + 1) % (SIDE * SIDE))
    # Two labels only, so z-scores are +-1: the floor has to sit below that.
    config = hs.QueryConfig(hot_threshold=0.3, background_prompts=["background"], segment_min_z=0.5)
    engine = HyperspaceQuery(store, StubModel.embed_text, config, world_frame=WORLD)
    answer = engine.answer("object", 1)
    stats = answer["stats"]
    # "other" embeds along axis 1: cosine 0 with the query, so only the four
    # "object" segments (one cell each) are read and become hot.
    assert stats["segment_labels"] == ["object"]
    assert stats["segment_hot_patches"] == 4
    assert stats["segment_read"] == 4
    assert stats["voxels_in_both"] > 0
    result: hs.Heatmap = answer["heatmap"]
    best_index, best_score = result.voxels[0]
    assert best_score == 1.0
    assert np.abs(np.asarray(answer["best"][0]["xyz"]) - OBJECT).max() <= 0.5
    # The winner is lit by both channels; the stray segment's voxels rank below it.
    patch_score, segment_score = result.channels[best_index]
    assert patch_score > 0 and segment_score > 0
    single = [s for e, s in result.channels.values() if e == 0]
    assert single and max(single) < best_score
    # Turning the channel off gives the patch-only answer back.
    off = HyperspaceQuery(
        store,
        StubModel.embed_text,
        hs.QueryConfig(hot_threshold=0.3, background_prompts=["background"], segment_weight=0),
        world_frame=WORLD,
    ).answer("object", 2)
    assert "segment_hot_patches" not in off["stats"]
    assert not off["heatmap"].channels
    del ingestor


class StubEnsemble:
    """Two stub members on different grids. Member A (8x8) sees the object;
    member B (4x4) sees it too but also hallucinates a hot cell in the top-left
    corner that A does not have, so a per-cell minimum should drop it."""

    specs = ["stub-a", "stub-b@16"]
    tags = ["stub-a", "stub-b-16"]

    def __init__(self) -> None:
        self.pixel: tuple[float, float] | None = None

    def _grid(self, side: int, hallucinate: bool) -> np.ndarray:
        grid = np.zeros((side * side, DIM), dtype=np.float32)
        grid[:, 1] = 1.0
        assert self.pixel is not None
        u, v = self.pixel
        col, row = int(u * side / WIDTH), int(v * side / HEIGHT)
        grid[row * side + col] = 0.0
        grid[row * side + col, 0] = 1.0
        if hallucinate:
            grid[0] = 0.0
            grid[0, 0] = 1.0
        return grid

    def embed_grids(self, image: Image) -> list[tuple[np.ndarray, tuple[int, int]]]:
        return [(self._grid(8, False), (8, 8)), (self._grid(4, True), (4, 4))]

    @staticmethod
    def embed_text(text: str) -> list[np.ndarray]:
        return [StubModel.embed_text(text), StubModel.embed_text(text)]


def fill_ensemble(store: SqliteStore, poses: list[np.ndarray]) -> PatchIngestor:
    model = StubEnsemble()
    config = IngestConfig(
        gate=hs.KeyframeGateConfig(
            lookahead=0,
            # Every distinct pose is a keyframe here: these tests are about what an
            # ingest writes, not about the tuned novelty of a real recording.
            novelty_threshold=0.0,
            patch_novelty_threshold=None,
            max_angular_velocity=None,
            max_dark_fraction=None,
            min_interval=None,
        ),
        min_frame_interval_s=0.0,
    )
    ingestor = PatchIngestor(store, model, config)  # type: ignore[arg-type]
    ingestor.add_camera_info(camera_info())
    for index, pose in enumerate(poses):
        ts = 10.0 + index
        x, y, z, w = quaternion_of(pose[:3, :3])
        transform = Transform(
            translation=Vector3(*pose[:3, 3]),
            rotation=Quaternion(x, y, z, w),
            frame_id=WORLD,
            child_frame_id=CAMERA,
            ts=ts,
        )
        ingestor.add_tf(TFMessage(transform), ts=ts)
        local = np.linalg.inv(pose) @ np.append(OBJECT, 1.0)
        model.pixel = (local[0] / local[2] * 48.0 + 32.0, local[1] / local[2] * 48.0 + 24.0)
        ingestor.add_depth(
            Image.from_numpy(
                np.full((HEIGHT, WIDTH), int(local[2] * 1000), dtype=np.uint16),
                frame_id=CAMERA,
                ts=ts,
            )
        )
        ingestor.add_image(
            Image.from_numpy(
                np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8), frame_id=CAMERA, ts=ts
            )
        )
    ingestor.flush()
    return ingestor


def test_cell_matrix_spreads_and_averages_exactly() -> None:
    identity = hs.cell_matrix((8, 8), (8, 8))
    assert np.allclose(identity, np.eye(64))
    up = hs.cell_matrix((2, 2), (4, 4))  # each source cell covers a 2x2 block of targets
    assert up.shape == (16, 4)
    assert np.allclose(up.sum(axis=1), 1.0)
    assert up[0, 0] == 1.0 and up[15, 3] == 1.0
    down = hs.cell_matrix((4, 4), (2, 2))  # each target averages a 2x2 block of sources
    assert np.allclose(down[0, [0, 1, 4, 5]], 0.25)
    odd = hs.cell_matrix((14, 14), (24, 24))  # grids that do not divide each other
    assert odd.shape == (576, 196)
    assert np.allclose(odd.sum(axis=1), 1.0)


def test_pool_cells() -> None:
    a = np.array([0.5, 0.1, 0.0], dtype=np.float32)
    b = np.array([0.4, 0.3, 0.2], dtype=np.float32)
    c = np.array([0.6, 0.0, 0.1], dtype=np.float32)
    assert np.allclose(hs.pool_cells([a, b, c], "min"), [0.4, 0.0, 0.0])
    assert np.allclose(hs.pool_cells([a, b, c], "2nd"), [0.5, 0.1, 0.1])
    assert np.allclose(hs.pool_cells([a, b, c], "mean"), [0.5, 0.4 / 3, 0.1])
    assert np.allclose(hs.pool_cells([a], "2nd"), a)  # one member: nothing to pool
    with pytest.raises(ValueError):
        hs.pool_cells([a, b], "median")


def test_ensemble_keyframes_carry_every_member_and_pool_with_a_minimum(store: SqliteStore) -> None:
    ingestor = fill_ensemble(store, ring(3, 2.5))
    assert ingestor.stats["kept"] == 3
    first = next(iter(store.stream(KEYFRAME_STREAM, dict).order_by("ts"))).data
    assert first["members"] == ["stub-a", "stub-b-16"]
    assert first["member_specs"] == ["stub-a", "stub-b@16"]
    assert first["grid_shapes"] == [[8, 8], [4, 4]]
    assert (first["rows"], first["cols"]) == (24, 24)
    assert len(first["patch_depth"]) == 24 * 24
    # One searchable index PER MODEL, each holding that model's own cells.
    assert cli.patch_stream_for("", "stub-a") == f"{PATCH_STREAM}__m_stub_a"
    assert store.stream(f"{PATCH_STREAM}__m_stub_a", dict).count() == 3 * 8 * 8
    assert store.stream(f"{PATCH_STREAM}__m_stub_b_16", dict).count() == 3 * 4 * 4

    config = hs.QueryConfig(
        structural_gate=False, segment_weight=0.0, pool="min", pooled_hot_threshold=0.005
    )
    engine = HyperspaceQuery(store, StubEnsemble.embed_text, config, world_frame=WORLD)
    assert engine.members() == ["stub-a", "stub-b-16"]
    result = engine.heatmap("object")
    assert result.voxels, "the object every member sees must light up"
    best = (np.asarray(result.voxels[0][0]) + 0.5) * result.voxel_size
    assert np.linalg.norm(best - OBJECT) < 0.35
    # member B's corner hallucination is not in member A, so the minimum has no
    # hot cell there: nothing is placed along the top-left rays
    hot, _ = engine.hot_patches(StubEnsemble.embed_text("object"))
    assert hot and all(h.patch != 0 for h in hot)

    # "2nd" of two members is the maximum: the hallucination comes back
    engine.config.pool = "2nd"
    hot, _ = engine.hot_patches(StubEnsemble.embed_text("object"))
    assert any(h.patch == 0 for h in hot)

    # a query side with the wrong number of text towers is refused
    with pytest.raises(ValueError):
        engine.hot_patches([StubModel.embed_text("object")])


def test_member_specs_and_tags() -> None:
    from dimos.mapping.hyperspace.embedder import member_tag, parse_member

    assert parse_member("google/siglip2-base-patch16-naflex@576") == (
        "google/siglip2-base-patch16-naflex",
        576,
        None,
    )
    assert parse_member("/models/siglip2-so400m-patch16-384") == (
        "/models/siglip2-so400m-patch16-384",
        None,
        None,
    )
    assert parse_member("google/siglip2-base-patch16-224#2x3") == (
        "google/siglip2-base-patch16-224",
        None,
        (2, 3),
    )
    assert member_tag("google/siglip2-base-patch16-naflex@576") == "base-patch16-naflex-576"
    assert member_tag("/models/siglip2-so400m-patch16-384") == "so400m-patch16-384"
    assert member_tag("google/siglip2-base-patch16-224#2x3") == "base-patch16-224-2x3"


def test_tiles_cover_the_frame_exactly_and_stitch_back() -> None:
    """A tiled member must rebuild the frame's geometry with no patch dropped,
    duplicated or averaged -- the stitched grid is the tiles laid side by side."""
    import numpy as np
    from PIL import Image as PILImage

    from dimos.mapping.hyperspace.embedder import stitch_tiles, tile_image

    frame = PILImage.fromarray(np.arange(480 * 848 * 3, dtype=np.uint8).reshape(480, 848, 3))
    crops = tile_image(frame, 2, 3)
    assert len(crops) == 6
    assert [c.size for c in crops] == [(282, 240), (283, 240), (283, 240)] * 2
    assert sum(w * h for w, h in (c.size for c in crops)) == 848 * 480  # exact cover

    side, dim = 14, 4
    # Each tile carries its own index, so a mis-stitch shows up as a wrong block.
    grids = [np.full((side * side, dim), i, np.float32) for i in range(6)]
    stitched = stitch_tiles(grids, 2, 3, side)
    assert stitched.shape == (2 * side * 3 * side, dim)
    block = stitched.reshape(2 * side, 3 * side, dim)
    for index in range(6):
        r, c = divmod(index, 3)
        tile = block[r * side : (r + 1) * side, c * side : (c + 1) * side]
        assert (tile == index).all(), f"tile {index} landed in the wrong place"


def test_ensemble_cell_grid_follows_the_finest_member() -> None:
    """A tiled member's resolution must survive the common cell grid: pooling
    two members onto a fixed 24x24 would throw the tiling away."""
    import numpy as np

    from dimos.mapping.hyperspace.ingest import IngestConfig, PatchIngestor

    ingestor = PatchIngestor.__new__(PatchIngestor)
    ingestor.config = IngestConfig(gate=hs.KeyframeGateConfig())

    def grids(*shapes: tuple[int, int]) -> list:
        return [(np.zeros((r * c, 4), np.float32), (r, c)) for r, c in shapes]

    assert ingestor.cell_grid(grids((14, 14))) == (14, 14)  # single member keeps its own
    assert ingestor.cell_grid(grids((14, 14), (16, 16))) == (24, 24)  # untiled pair unchanged
    assert ingestor.cell_grid(grids((14, 14), (28, 42))) == (28, 42)  # tiling survives
    ingestor.config = IngestConfig(gate=hs.KeyframeGateConfig(), cell_grid=(24, 24))
    assert ingestor.cell_grid(grids((14, 14), (28, 42))) == (24, 24)  # explicit still wins


# --- one recording, one file ---------------------------------------------------------
# The keyframes and patches belong in the recording they describe. Only an .mcap, which
# cannot be written to, gets a companion db beside it.


def test_a_db_indexes_itself_and_only_an_mcap_gets_a_companion() -> None:
    assert cli.memory_db_for(Path("/data/grocery.db")) == Path("/data/grocery.db")
    assert cli.memory_db_for(Path("/data/grocery.mcap")) == Path("/data/grocery.hyperspace.db")


def test_keyframes_alone_are_reusable_and_a_legacy_marker_is_dropped_with_them(
    store: SqliteStore,
) -> None:
    """No completeness marker (Jeff, 2026-09-12): keyframes being there is the whole test.

    The cost, stated so a later reader does not think it an oversight: a run killed
    mid-ingest leaves keyframes that read as a finished index, and only --no-reuse
    replaces them.
    """
    assert not cli.index_is_finished(store)
    fill(store, ring(3, 2.5))
    assert store.stream(KEYFRAME_STREAM, dict).count() == 3
    assert cli.index_is_finished(store)

    # Dbs indexed before the marker went away still carry one; dropping the index must
    # take it too, or it would vouch for keyframes that are no longer there.
    store.stream(COMPLETE_STREAM, String).append(String("3 keyframes"), ts=1.0)
    cli.drop_index(store)
    assert not cli.index_is_finished(store)
    for name in (KEYFRAME_STREAM, PATCH_STREAM, COMPLETE_STREAM):
        assert name not in store.list_streams(), name


def test_an_empty_source_stream_is_refused_before_the_old_index_is_touched(
    store: SqliteStore,
) -> None:
    """The pre-flight is the whole point: refusing after the drop costs the index."""
    fill(store, ring(3, 2.5))
    store.stream("empty_camera_info", CameraInfo)  # named, never written to

    with pytest.raises(typer.BadParameter, match="nothing was changed"):
        cli.refuse_unless_readable(store, (KEYFRAME_STREAM, "empty_camera_info"))

    # The refusal cost nothing: the index that was there is still there.
    assert cli.index_is_finished(store)
    assert store.stream(KEYFRAME_STREAM, dict).count() == 3


def test_reingesting_in_place_replaces_the_keyframes_rather_than_appending(
    store: SqliteStore,
) -> None:
    """Writing into the recording means a rerun would otherwise double every keyframe."""
    fill(store, ring(3, 2.5))
    patches_after_one_run = store.stream(PATCH_STREAM, dict).count()

    cli.drop_index(store)  # what the ingest does before it re-embeds
    fill(store, ring(3, 2.5))

    assert store.stream(KEYFRAME_STREAM, dict).count() == 3
    assert store.stream(PATCH_STREAM, dict).count() == patches_after_one_run


# --- several models in one recording ---------------------------------------------------
# A recording can carry more than one index so two checkpoints can be compared on the
# same question. Each keyframe and patch says which model wrote it.


def test_every_keyframe_and_patch_records_the_model_that_wrote_it(store: SqliteStore) -> None:
    fill(store, ring(2, 2.5))
    first = store.stream(KEYFRAME_STREAM, dict).order_by("ts").first()
    assert first.data["model"], "a keyframe with no model is a keyframe nobody can place"
    assert "member_specs" in first.data and "members" in first.data
    assert first.tags["model"] == first.data["model"]
    patch = store.stream(cli.patch_stream_for("", "stub"), dict).order_by("ts").first()
    assert patch.data["model"] == first.data["model"]


def test_a_second_model_gets_its_own_index_instead_of_overwriting_the_first(
    store: SqliteStore,
) -> None:
    """The point of keeping both is comparison; clobbering one would defeat it."""
    one = ["google/siglip2-base-patch16-224"]
    two = ["google/siglip2-base-patch16-384"]

    assert cli.pick_index(store, one) == "", "the first model in takes the canonical streams"
    fill(store, ring(2, 2.5))  # writes the canonical pair

    # Same checkpoints again: the same index, so a re-ingest replaces rather than forks.
    specs = store.stream(KEYFRAME_STREAM, dict).order_by("ts").first().data["member_specs"]
    assert cli.pick_index(store, specs) == ""

    # A different checkpoint: its own streams, named after it.
    slug = cli.pick_index(store, two)
    assert slug == cli.index_slug(two) != ""
    keyframes, patches = cli.stream_names(slug)
    assert keyframes == f"{KEYFRAME_STREAM}__{slug}"
    assert patches == f"{PATCH_STREAM}__{slug}"


def test_dropping_one_index_leaves_the_others_alone(store: SqliteStore) -> None:
    fill(store, ring(2, 2.5))
    other = cli.index_slug(["google/siglip2-base-patch16-384"])
    other_keyframes, other_patches = cli.stream_names(other)
    store.stream(other_keyframes, dict).append({"model": other}, ts=1.0)
    store.stream(other_patches, dict).append({"model": other}, ts=1.0)

    cli.drop_index(store, other)
    assert other_keyframes not in store.list_streams()
    assert cli.index_is_finished(store), "the canonical index must survive its neighbour"

    cli.drop_index(store)
    assert not cli.index_is_finished(store)


def test_every_model_gets_its_own_searchable_index(store: SqliteStore, tmp_path: Path) -> None:
    """A query asks each model its own nearest-neighbour question.

    It cannot do that against one table holding only the primary model, and reading a
    patch out of a keyframe's `grids` blob means unpickling the whole keyframe -- which
    is why a query used to load every one of them into memory.
    """
    # One model keeps the bare name, so stores written before ensembles still read.
    fill(store, ring(2, 2.5))
    assert store.stream(cli.patch_stream_for("", "stub"), dict).count() == 2 * SIDE * SIDE

    other = SqliteStore(path=str(tmp_path / "ensemble.db"))
    other.start()
    try:
        fill_ensemble(other, ring(2, 2.5))
        names = set(other.list_streams())
        assert f"{PATCH_STREAM}__m_stub_a" in names, sorted(names)
        assert f"{PATCH_STREAM}__m_stub_b_16" in names, sorted(names)
        # Each index holds ITS OWN model's cells, not a copy of the primary's.
        assert other.stream(f"{PATCH_STREAM}__m_stub_a", dict).count() == 2 * 8 * 8
        assert other.stream(f"{PATCH_STREAM}__m_stub_b_16", dict).count() == 2 * 4 * 4
        # A re-ingest must replace them, not append a second copy of every vector.
        cli.drop_index(other)
        assert f"{PATCH_STREAM}__m_stub_a" not in set(other.list_streams())
    finally:
        other.stop()
