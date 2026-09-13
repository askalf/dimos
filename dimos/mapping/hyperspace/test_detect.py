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

"""The frames-first path: episodes, the 2D box plus depth becoming a 3D box, and what
happens when the detector refuses.

The geometry tests use numbers small enough to check by hand -- a 64x48 camera with
fx = fy = 48 and its centre at (32, 24), and a square of depth planted at a known
distance -- so a wrong answer says which step is wrong rather than only that one is.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dimos.mapping.hyperspace import patches as hs
from dimos.mapping.hyperspace.detect import (
    DetectConfig,
    Detection,
    RecordingFrames,
    box_from_points,
    detect_episode,
    find,
    object_points,
)
from dimos.mapping.hyperspace.frames import Episode, Frame, Hit, episodes, ranked_episodes
from dimos.memory.codecs.lcm import LcmCodec
from dimos.memory.codecs.lz4 import Lz4Codec
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.tf2_msgs.TFMessage import TFMessage

WIDTH, HEIGHT = 64, 48
FOCAL, CX, CY = 48.0, 32.0, 24.0
CAMERA = "camera_optical"
WORLD = "odom"

# The planted square, in pixels, and how far away it is.
SQUARE = (28, 36, 20, 28)  # left, right, top, bottom
SQUARE_DEPTH = 2.0
BACKGROUND_DEPTH = 8.0

# Eight columns 1/48 of a radian apart, at two metres: the outermost centres sit
# +-3.5/48 * 2 m from the axis, and the 2nd/98th percentiles of eight repeated values
# land on those same outermost ones.
HALF_EXTENT = 3.5 / FOCAL * SQUARE_DEPTH


def intrinsics() -> hs.Intrinsics:
    return hs.Intrinsics(width=WIDTH, height=HEIGHT, fx=FOCAL, fy=FOCAL, cx=CX, cy=CY)


def planted_depth(right: int = SQUARE[1]) -> np.ndarray:
    """Background everywhere, with a square of near depth in the middle."""
    depth = np.full((HEIGHT, WIDTH), BACKGROUND_DEPTH, dtype=np.float32)
    depth[SQUARE[2] : SQUARE[3], SQUARE[0] : right] = SQUARE_DEPTH
    return depth


def frame_at(ts: float, score: float = 1.0, member: str = "stub") -> Frame:
    hit = Hit(
        member=member,
        frame=CAMERA,
        ts=ts,
        cell=0,
        grid=(2, 2),
        ray=(0.0, 0.0),
        depth=SQUARE_DEPTH,
        score=score,
    )
    return Frame(frame=CAMERA, ts=ts, hits=[hit])


# --- episodes -----------------------------------------------------------------------


def test_episodes_split_on_a_gap_in_time() -> None:
    frames = [frame_at(ts) for ts in (0.0, 0.25, 0.5, 5.0, 5.25)]
    found = episodes(frames, gap_s=1.0)
    assert [len(episode.frames) for episode in found] == [3, 2]
    assert found[0].span == pytest.approx(0.5)
    assert found[1].start == pytest.approx(5.0)


def test_a_wider_gap_joins_what_a_narrow_one_split() -> None:
    frames = [frame_at(ts) for ts in (0.0, 0.25, 5.0)]
    assert len(episodes(frames, gap_s=1.0)) == 2
    assert len(episodes(frames, gap_s=10.0)) == 1


def test_the_peak_frame_is_the_one_with_the_most_match_in_it() -> None:
    frames = [frame_at(0.0, score=0.1), frame_at(0.25, score=0.9), frame_at(0.5, score=0.4)]
    episode = Episode(frames=frames)
    assert episode.peak.ts == pytest.approx(0.25)
    assert [frame.ts for frame in episode.by_weight()] == [0.25, 0.5, 0.0]


def test_ranked_episodes_drop_strays_and_rank_by_score() -> None:
    frames = [
        frame_at(0.0, score=0.2),
        frame_at(0.25, score=0.2),
        frame_at(9.0, score=5.0),  # one lone frame, however hot
        frame_at(20.0, score=0.9),
        frame_at(20.25, score=0.9),
    ]
    found = ranked_episodes(frames, gap_s=1.0, min_frames=2)
    assert [episode.start for episode in found] == [20.0, 0.0]
    assert all(len(episode.frames) >= 2 for episode in found)


# --- a 2D box and a depth image becoming a 3D box -----------------------------------


def test_object_points_measure_the_planted_square() -> None:
    found = object_points((28.0, 20.0, 36.0, 28.0), (WIDTH, HEIGHT), planted_depth(), intrinsics())
    assert found is not None
    points, median = found
    assert median == pytest.approx(SQUARE_DEPTH)
    assert len(points) == 8 * 8
    assert points[:, 2].min() == pytest.approx(SQUARE_DEPTH)
    assert points[:, 0].min() == pytest.approx(-HALF_EXTENT)
    assert points[:, 0].max() == pytest.approx(HALF_EXTENT)


def test_the_depth_band_drops_the_background_showing_through_the_box() -> None:
    """A box that is two thirds object and one third the aisle behind it."""
    depth = planted_depth()
    box = (28.0, 20.0, 40.0, 28.0)  # four columns wider than the square
    found = object_points(box, (WIDTH, HEIGHT), depth, intrinsics())
    assert found is not None
    points, median = found
    assert median == pytest.approx(SQUARE_DEPTH)
    # Only the square's own 64 pixels survive; the 32 background ones are 6 m away.
    assert len(points) == 8 * 8
    assert points[:, 2].max() == pytest.approx(SQUARE_DEPTH)


def test_no_usable_depth_inside_the_box_is_reported_rather_than_guessed() -> None:
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    assert object_points((28.0, 20.0, 36.0, 28.0), (WIDTH, HEIGHT), empty, intrinsics()) is None


def test_box_from_points_under_the_identity_pose() -> None:
    found = object_points((28.0, 20.0, 36.0, 28.0), (WIDTH, HEIGHT), planted_depth(), intrinsics())
    assert found is not None
    points, median = found
    box = box_from_points(points, np.eye(4), WORLD, median)
    assert box.frame == WORLD
    assert box.centre == pytest.approx((0.0, 0.0, SQUARE_DEPTH), abs=1e-6)
    assert box.extent == pytest.approx((2 * HALF_EXTENT, 2 * HALF_EXTENT, 0.0), abs=1e-6)
    assert box.pixels == 64
    assert box.depth_m == pytest.approx(SQUARE_DEPTH)


def test_box_from_points_carries_the_camera_pose() -> None:
    """Camera at (10, -4, 1) yawed 90 degrees: optical +z points along world +y.

    Optical axes are x right, y down, z forward. A yaw of 90 degrees about world z
    sends camera x -> world -x... so the columns below spell out where each optical
    axis goes, and the object two metres ahead lands two metres along world +y.
    """
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack(
        [
            [-1.0, 0.0, 0.0],  # optical x (right) -> world -x
            [0.0, 0.0, -1.0],  # optical y (down)  -> world -z
            [0.0, 1.0, 0.0],  # optical z (ahead) -> world +y
        ]
    )
    pose[:3, 3] = [10.0, -4.0, 1.0]
    found = object_points((28.0, 20.0, 36.0, 28.0), (WIDTH, HEIGHT), planted_depth(), intrinsics())
    assert found is not None
    points, median = found
    box = box_from_points(points, pose, WORLD, median)
    assert box.centre == pytest.approx((10.0, -4.0 + SQUARE_DEPTH, 1.0), abs=1e-6)
    # The square's width is now along world x and its height along world z.
    assert box.extent == pytest.approx((2 * HALF_EXTENT, 0.0, 2 * HALF_EXTENT), abs=1e-6)


def test_a_box_on_a_different_grid_than_the_intrinsics_still_lands() -> None:
    """Depth at half the colour camera's resolution: the box must land in the same place.

    The square is then 4x4 pixels instead of 8x8, so its extent is measured between the
    centres of the outermost of four columns rather than of eight: +-1.5 px at a
    half-grid focal length of 24, which is +-0.125 m at two metres.
    """
    half = planted_depth()[::2, ::2]
    found = object_points(
        (28.0, 20.0, 36.0, 28.0), (WIDTH, HEIGHT), half, intrinsics(), min_pixels=8
    )
    assert found is not None
    points, median = found
    assert len(points) == 4 * 4
    box = box_from_points(points, np.eye(4), WORLD, median)
    assert box.centre == pytest.approx((0.0, 0.0, SQUARE_DEPTH), abs=1e-6)
    assert box.extent[0] == pytest.approx(2 * (1.5 / (FOCAL / 2)) * SQUARE_DEPTH, abs=1e-6)


# --- the whole episode, against a real store ----------------------------------------


class StubBoxes:
    """A detector that answers with a fixed box, or refuses, and counts its calls."""

    def __init__(self, box: tuple[float, float, float, float] | None, score: float = 0.5) -> None:
        self.box = box
        self.score = score
        self.calls = 0

    def best(
        self, image: Image, text: str
    ) -> tuple[tuple[float, float, float, float], float] | None:
        self.calls += 1
        del image, text
        return None if self.box is None else (self.box, self.score)


class RefusesThenAnswers(StubBoxes):
    """Refuses the first frame it is shown and answers the second."""

    def best(
        self, image: Image, text: str
    ) -> tuple[tuple[float, float, float, float], float] | None:
        self.calls += 1
        del image, text
        if self.calls == 1:
            return None
        return (self.box, self.score)  # type: ignore[return-value]


def camera_info(frame_id: str = CAMERA) -> CameraInfo:
    info = CameraInfo(width=WIDTH, height=HEIGHT, distortion_model="plumb_bob", frame_id=frame_id)
    info.set_K_matrix(np.array([[FOCAL, 0.0, CX], [0.0, FOCAL, CY], [0.0, 0.0, 1.0]]))
    return info


@pytest.fixture
def recording(tmp_path: Path) -> SqliteStore:
    """A minimal recording: four colour frames, their depth, intrinsics and one tf.

    The frames come in two pairs a long way apart in time, so the same fixture serves a
    single-episode test and one that needs two episodes.
    """
    store = SqliteStore(path=str(tmp_path / "recording.db"))
    store.start()
    store.stream("camera_info", CameraInfo).append(camera_info(), ts=10.0)
    store.stream("depth_camera_info", CameraInfo).append(camera_info(), ts=10.0)
    colors = store.stream("color_image", Image)
    # A real recording stores depth lz4+lcm; the default Image codec is JPEG, which
    # would quietly hand back an 8-bit RGB picture of the depth instead of metres.
    depths = store.stream("depth_image", Image, codec=Lz4Codec(LcmCodec(Image)))
    for ts in (10.0, 10.25, 30.0, 30.25):
        colors.append(
            Image.from_numpy(
                np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8), frame_id=CAMERA, ts=ts
            ),
            ts=ts,
        )
        depths.append(
            Image.from_numpy(
                (planted_depth() * 1000).astype(np.uint16),
                format=ImageFormat.DEPTH16,
                frame_id=CAMERA,
                ts=ts,
            ),
            ts=ts,
        )
    store.stream("tf", TFMessage).append(
        TFMessage(
            Transform(
                translation=Vector3(1.0, 2.0, 3.0),
                rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                frame_id=WORLD,
                child_frame_id=CAMERA,
                ts=10.0,
            )
        ),
        ts=10.0,
    )
    yield store
    store.stop()


def test_detect_episode_places_the_box_in_the_world(recording: SqliteStore) -> None:
    episode = Episode(frames=[frame_at(10.0, 0.9), frame_at(10.25, 0.4)])
    boxes = StubBoxes((28.0, 20.0, 36.0, 28.0))
    config = DetectConfig(world_frame=WORLD)
    found = detect_episode(
        episode, "a square", RecordingFrames(recording, config=config), boxes, rank=1, config=config
    )
    assert found.found
    assert boxes.calls == 1, "the peak frame answered, so nothing else should be tried"
    assert found.ts == pytest.approx(10.0)
    assert found.score == pytest.approx(0.5)
    assert found.box3d is not None
    # The camera sits at (1, 2, 3) with no rotation, so the square is two metres along
    # its optical z, which is world z.
    assert found.box3d.centre == pytest.approx((1.0, 2.0, 3.0 + SQUARE_DEPTH), abs=1e-6)
    assert found.box3d.frame == WORLD
    assert found.box3d.depth_m == pytest.approx(SQUARE_DEPTH)


def test_an_episode_the_detector_refuses_is_reported_not_dropped(recording: SqliteStore) -> None:
    episode = Episode(frames=[frame_at(10.0, 0.9), frame_at(10.25, 0.4)])
    boxes = StubBoxes(None)
    config = DetectConfig(world_frame=WORLD, attempts=2)
    found = detect_episode(
        episode, "a square", RecordingFrames(recording, config=config), boxes, rank=3, config=config
    )
    assert isinstance(found, Detection)
    assert not found.found
    assert found.box2d is None and found.box3d is None
    assert found.rank == 3 and found.episode_frames == 2
    assert boxes.calls == 2, "both frames of the episode should have been tried"
    assert "refused" in found.note


def test_the_second_frame_is_tried_when_the_peak_frame_fails(recording: SqliteStore) -> None:
    episode = Episode(frames=[frame_at(10.0, 0.9), frame_at(10.25, 0.4)])
    boxes = RefusesThenAnswers((28.0, 20.0, 36.0, 28.0))
    config = DetectConfig(world_frame=WORLD, attempts=3)
    found = detect_episode(
        episode, "a square", RecordingFrames(recording, config=config), boxes, rank=1, config=config
    )
    assert found.found
    assert found.attempts == 2
    assert found.ts == pytest.approx(10.25), "the answer came from the second-best frame"


def test_a_detection_without_depth_still_reports_the_2d_box(recording: SqliteStore) -> None:
    """Depth that is all holes: the image answer survives, the world answer says why."""
    blank = recording.stream("depth_image", Image, codec=Lz4Codec(LcmCodec(Image)))
    blank.append(
        Image.from_numpy(
            np.zeros((HEIGHT, WIDTH), dtype=np.uint16),
            format=ImageFormat.DEPTH16,
            frame_id=CAMERA,
            ts=50.0,
        ),
        ts=50.0,
    )
    recording.stream("color_image", Image).append(
        Image.from_numpy(
            np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8), frame_id=CAMERA, ts=50.0
        ),
        ts=50.0,
    )
    episode = Episode(frames=[frame_at(50.0, 0.9)])
    config = DetectConfig(world_frame=WORLD)
    found = detect_episode(
        episode,
        "a square",
        RecordingFrames(recording, config=config),
        StubBoxes((28.0, 20.0, 36.0, 28.0)),
        rank=1,
        config=config,
    )
    assert found.found and found.box2d is not None
    assert found.box3d is None
    assert found.note == "no usable depth inside the box"


def test_find_yields_episodes_one_at_a_time(recording: SqliteStore, monkeypatch) -> None:
    """The detector must not have run for every episode before the first result."""
    frames = [frame_at(ts, 0.9) for ts in (10.0, 10.25)] + [
        frame_at(ts, 0.4) for ts in (30.0, 30.25)
    ]
    monkeypatch.setattr("dimos.mapping.hyperspace.frames.hot_frames", lambda *a, **k: frames)
    boxes = StubBoxes((28.0, 20.0, 36.0, 28.0))
    config = DetectConfig(world_frame=WORLD)
    answers = find(
        recording,
        recording,
        "a square",
        config=config,
        frames=RecordingFrames(recording, config=config),
        boxes=boxes,
    )
    first = next(answers)
    assert first.rank == 1
    assert boxes.calls == 1, "the second episode should not have been detected yet"
    rest = list(answers)
    assert len(rest) == 1 and rest[0].rank == 2
    assert boxes.calls == 2


def test_geometry_matches_a_hand_computation() -> None:
    """The whole chain, written out longhand, against the code.

    A pixel at column *c* of a camera with focal length f and centre cx sees a ray of
    slope ``(c + 0.5 - cx) / f``; at distance d that is ``slope * d`` metres off the
    axis. Nothing else is involved.
    """
    left, right, top, bottom = SQUARE
    xs, ys = [], []
    for col in range(left, right):
        for row in range(top, bottom):
            xs.append((col + 0.5 - CX) / FOCAL * SQUARE_DEPTH)
            ys.append((row + 0.5 - CY) / FOCAL * SQUARE_DEPTH)
    by_hand_x = np.percentile(xs, 98) - np.percentile(xs, 2)
    by_hand_y = np.percentile(ys, 98) - np.percentile(ys, 2)

    found = object_points((28.0, 20.0, 36.0, 28.0), (WIDTH, HEIGHT), planted_depth(), intrinsics())
    assert found is not None
    box = box_from_points(found[0], np.eye(4), WORLD, found[1])
    assert box.extent[0] == pytest.approx(by_hand_x)
    assert box.extent[1] == pytest.approx(by_hand_y)
    assert math.isclose(box.extent[0], 2 * HALF_EXTENT, rel_tol=1e-9)


def test_an_unplaceable_frame_falls_through_to_the_next_one(recording: SqliteStore) -> None:
    """Detected but with no depth behind the box: try the next look at the same thing.

    Stereo gives nothing back off glass, a dark shelf or a shiny floor, and that is a
    property of one frame, not of the object. The next-best frame of the episode is
    another chance at the same thing, so an empty depth image must not end the episode.
    """
    colors = recording.stream("color_image", Image)
    depths = recording.stream("depth_image", Image, codec=Lz4Codec(LcmCodec(Image)))
    for ts, depth in ((60.0, np.zeros((HEIGHT, WIDTH), np.uint16)), (60.25, None)):
        colors.append(
            Image.from_numpy(
                np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8), frame_id=CAMERA, ts=ts
            ),
            ts=ts,
        )
        pixels = (planted_depth() * 1000).astype(np.uint16) if depth is None else depth
        depths.append(
            Image.from_numpy(pixels, format=ImageFormat.DEPTH16, frame_id=CAMERA, ts=ts), ts=ts
        )

    episode = Episode(frames=[frame_at(60.0, 0.9), frame_at(60.25, 0.4)])
    boxes = StubBoxes((28.0, 20.0, 36.0, 28.0))
    config = DetectConfig(world_frame=WORLD, attempts=3)
    found = detect_episode(
        episode, "a square", RecordingFrames(recording, config=config), boxes, rank=1, config=config
    )
    assert found.found
    assert found.box3d is not None, "the second frame's depth should have placed it"
    assert found.ts == pytest.approx(60.25)
    assert boxes.calls == 2


def test_a_detection_that_can_never_be_placed_is_still_returned(recording: SqliteStore) -> None:
    """Every frame of the episode is unplaceable: report the 2D answer and say why."""
    colors = recording.stream("color_image", Image)
    depths = recording.stream("depth_image", Image, codec=Lz4Codec(LcmCodec(Image)))
    for ts in (70.0, 70.25):
        colors.append(
            Image.from_numpy(
                np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8), frame_id=CAMERA, ts=ts
            ),
            ts=ts,
        )
        depths.append(
            Image.from_numpy(
                np.zeros((HEIGHT, WIDTH), np.uint16),
                format=ImageFormat.DEPTH16,
                frame_id=CAMERA,
                ts=ts,
            ),
            ts=ts,
        )
    episode = Episode(frames=[frame_at(70.0, 0.9), frame_at(70.25, 0.4)])
    config = DetectConfig(world_frame=WORLD, attempts=3)
    found = detect_episode(
        episode,
        "a square",
        RecordingFrames(recording, config=config),
        StubBoxes((28.0, 20.0, 36.0, 28.0)),
        rank=1,
        config=config,
    )
    assert found.found and found.box3d is None
    assert found.ts == pytest.approx(70.0), "the first, strongest attempt is the one kept"
    assert found.note == "no usable depth inside the box"
