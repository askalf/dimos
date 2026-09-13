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

"""Frames first, geometry second: one good image per episode, handed to a detector.

The patches rank *frames*, which is the thing they are actually good at, and an
open-vocabulary detector does the finding. Each episode's best frame goes to OWLv2 with
the query words; its 2D box plus that frame's full depth image become a 3D box.

What this removes, compared with placing every hot patch in the world and hoping the
pyramids overlap: no overlap margin to tune, no group score to invent, and no
requirement that per-patch depth be accurate -- the box comes from the detector and the
depth image, not from a pyramid. Cross-model agreement survives as *which models voted
for this episode*, which needs nothing to coincide in space.

Results are yielded per episode, as they are found. A detector frame costs the better
part of a second, so a caller that waits for the whole list waits for all of them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.mapping.hyperspace import patches as hs
from dimos.mapping.hyperspace.flextf import FlexTf
from dimos.mapping.hyperspace.frames import Episode, Frame
from dimos.mapping.hyperspace.ingest import TF_STREAM, decoded, intrinsics_of
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = setup_logger()

OWLV2_CHECKPOINT = "google/owlv2-base-patch16-ensemble"


@dataclass
class DetectConfig:
    """Everything the detector path can be turned by, in one place."""

    # OWLv2's own per-box acceptance score. Its scores are calibrated, so this is a real
    # refusal threshold rather than a ranking cut.
    threshold: float = 0.15
    checkpoint: str = OWLV2_CHECKPOINT
    device: str = ""
    # Frames of one episode to try before calling it undetected. The peak frame is
    # usually right, but an object can be half out of the frame at the moment it scores
    # highest, and the next look is free of that.
    attempts: int = 3
    # Episodes to run the detector over at all, strongest first. Detection is ~0.7 s a
    # frame, so this is the knob that decides what a query costs.
    max_episodes: int = 12
    # A run of fewer frames than this is a stray: at ~4 Hz an object the camera really
    # passed is seen several times running.
    min_episode_frames: int = 2
    episode_gap_s: float = 1.0
    # Depth beyond this is a hole, not a reading: RealSense 65535 mm sentinels, and
    # stereo that has given up.
    max_depth_m: float = 10.0
    # Depth pixels inside the 2D box this far from the box's median depth are not the
    # object -- they are the aisle behind it showing through, or the shelf in front.
    depth_band_m: float = 0.5
    # Below this many usable depth pixels the 3D box would be noise; the detection is
    # still reported, without one.
    min_depth_pixels: int = 20
    # The box is the 2nd-98th percentile of the object's points per axis, so one stray
    # pixel on a background surface cannot stretch it across the aisle.
    trim_percentile: float = 2.0
    # How far a depth frame may be from the colour frame to be its pair.
    depth_max_dt: float = 0.05
    world_frame: str = "odom"


@dataclass
class Box3D:
    """An axis-aligned box in the world frame, with what it was measured from."""

    frame: str
    centre: tuple[float, float, float]
    extent: tuple[float, float, float]
    pixels: int
    depth_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "centre": list(self.centre),
            "extent": list(self.extent),
            "pixels": self.pixels,
            "depth_m": self.depth_m,
        }


@dataclass
class Detection:
    """One episode's answer: where in the recording, and where in the world."""

    query: str
    rank: int
    ts: float
    camera_frame: str
    episode_frames: int
    episode_span: float
    episode_score: float
    models: list[str]
    attempts: int
    score: float = 0.0
    box2d: tuple[float, float, float, float] | None = None
    box3d: Box3D | None = None
    note: str = ""
    # Set by `merge_duplicates`: the rank of the detection this one is another look at.
    duplicate_of: int | None = None
    image: Image | None = field(default=None, repr=False)

    @property
    def found(self) -> bool:
        return self.box2d is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rank": self.rank,
            "ts": self.ts,
            "camera_frame": self.camera_frame,
            "episode_frames": self.episode_frames,
            "episode_span": self.episode_span,
            "episode_score": self.episode_score,
            "models": self.models,
            "attempts": self.attempts,
            "score": self.score,
            "box2d": None if self.box2d is None else list(self.box2d),
            "box3d": None if self.box3d is None else self.box3d.as_dict(),
            "note": self.note,
            "duplicate_of": self.duplicate_of,
        }


def object_points(
    box: Sequence[float],
    image_size: tuple[int, int],
    depth: NDArray[np.floating],
    intrinsics: hs.Intrinsics,
    *,
    max_depth_m: float = 10.0,
    band_m: float = 0.5,
    min_pixels: int = 20,
) -> tuple[NDArray[np.float64], float] | None:
    """The object's points, in the camera's own frame, from a 2D box and a depth image.

    *box* is ``(x1, y1, x2, y2)`` in the pixels of an image of *image_size*; *depth* is
    metres on the colour camera's grid, zero where there is no reading.

    A detector box is tight around the object but never only the object: its corners
    see whatever is behind. So the box's median depth is taken to be the object -- for a
    box that is mostly its subject, it is -- and pixels further than *band_m* from that
    are dropped as background showing through.
    """
    height, width = depth.shape
    image_width, image_height = image_size
    if not image_width or not image_height or not width or not height:
        return None
    # The box arrives in the decoded image's pixels; the depth may be on a different
    # grid. Going through fractions of the frame keeps the two independent of each other.
    left = int(np.clip(np.floor(box[0] / image_width * width), 0, width - 1))
    top = int(np.clip(np.floor(box[1] / image_height * height), 0, height - 1))
    right = int(np.clip(np.ceil(box[2] / image_width * width), left + 1, width))
    bottom = int(np.clip(np.ceil(box[3] / image_height * height), top + 1, height))

    window = np.asarray(depth[top:bottom, left:right], dtype=np.float64)
    usable = np.isfinite(window) & (window > 0) & (window <= max_depth_m)
    if int(usable.sum()) < min_pixels:
        return None
    median = float(np.median(window[usable]))
    keep = usable & (np.abs(window - median) <= band_m)
    if int(keep.sum()) < min_pixels:
        return None

    rows, cols = np.nonzero(keep)
    z = window[rows, cols]
    # Intrinsics scaled onto whatever grid the depth is actually on.
    scale_x, scale_y = width / intrinsics.width, height / intrinsics.height
    us = (cols + left + 0.5 - intrinsics.cx * scale_x) / (intrinsics.fx * scale_x)
    vs = (rows + top + 0.5 - intrinsics.cy * scale_y) / (intrinsics.fy * scale_y)
    return np.stack([us * z, vs * z, z], axis=1), median


def box_from_points(
    points: NDArray[np.floating],
    world_from_camera: NDArray[np.floating],
    frame: str,
    depth_m: float,
    *,
    trim_percentile: float = 2.0,
) -> Box3D:
    """Camera-frame points through a pose into an axis-aligned world box."""
    moved = (world_from_camera[:3, :3] @ np.asarray(points, dtype=np.float64).T).T
    moved += world_from_camera[:3, 3]
    low = np.percentile(moved, trim_percentile, axis=0)
    high = np.percentile(moved, 100.0 - trim_percentile, axis=0)
    centre = (low + high) / 2.0
    extent = high - low
    return Box3D(
        frame=frame,
        centre=(float(centre[0]), float(centre[1]), float(centre[2])),
        extent=(float(extent[0]), float(extent[1]), float(extent[2])),
        pixels=len(moved),
        depth_m=float(depth_m),
    )


class RecordingFrames:
    """The recording's own images, depth and transforms, read one moment at a time.

    The index says *which* moments are interesting; the pictures themselves still live
    in the recording. Depth is re-rendered onto the colour camera's grid on the way out,
    the same way the ingest pairs them, so a box drawn on the colour image indexes
    straight into it.
    """

    def __init__(
        self,
        recording: Any,
        *,
        color_stream: str = "color_image",
        depth_stream: str = "depth_image",
        color_info_stream: str = "camera_info",
        depth_info_stream: str = "depth_camera_info",
        tf_stream: str = TF_STREAM,
        config: DetectConfig | None = None,
    ) -> None:
        self.recording = recording
        self.color_stream = color_stream
        self.depth_stream = depth_stream
        self.config = config or DetectConfig()
        self.intrinsics: dict[str, hs.Intrinsics] = {}
        for name in (color_info_stream, depth_info_stream):
            if name in recording.list_streams():
                for observation in recording.streams[name].order_by("ts").limit(1):
                    info = observation.data
                    self.intrinsics[info.frame_id] = intrinsics_of(info)
        self.tf = FlexTf()
        self._tf_stream = tf_stream
        self._tf_loaded = False

    def load_tf(self) -> None:
        if self._tf_loaded or self._tf_stream not in self.recording.list_streams():
            self._tf_loaded = True
            return
        for observation in self.recording.stream(self._tf_stream, TFMessage).order_by("ts"):
            self.tf.receive_tfmessage(observation.data)
        self._tf_loaded = True

    def pose(self, camera_frame: str, ts: float, world_frame: str) -> NDArray[np.float64] | None:
        self.load_tf()
        poses, valid = self.tf.batch_get(world_frame, camera_frame, [ts])
        return poses[0] if valid[0] else None

    def color(self, ts: float) -> Image | None:
        found = self.recording.streams[self.color_stream].at(ts, tolerance=0.05).to_list()
        if not found:
            return None
        nearest = min(found, key=lambda observation: abs(float(observation.ts) - ts))
        return decoded(nearest.data)

    def depth(self, camera_frame: str, ts: float) -> NDArray[np.float32] | None:
        """Metres on the colour camera's grid, zero where there is no reading."""
        tolerance = self.config.depth_max_dt
        found = self.recording.streams[self.depth_stream].at(ts, tolerance=tolerance).to_list()
        if not found:
            return None
        nearest = min(found, key=lambda observation: abs(float(observation.ts) - ts))
        if abs(float(nearest.ts) - ts) > tolerance:
            return None
        image = decoded(nearest.data)
        raw = np.asarray(image.as_numpy())
        metres = raw.astype(np.float32) * (0.001 if raw.dtype == np.uint16 else 1.0)
        metres[(metres > self.config.max_depth_m) | ~np.isfinite(metres)] = 0.0

        color = self.intrinsics.get(camera_frame)
        depth_frame = image.frame_id
        if color is None:
            return metres
        if depth_frame == camera_frame:
            return metres
        depth_intrinsics = self.intrinsics.get(depth_frame)
        if depth_intrinsics is None:
            return metres
        self.load_tf()
        poses, valid = self.tf.batch_get(camera_frame, depth_frame, [ts])
        if not valid[0]:
            return metres
        return hs.reproject_depth(metres, depth_intrinsics, color, poses[0])


class Owlv2Boxes:
    """Core's OWLv2 detector, loaded once and asked one image at a time."""

    def __init__(self, config: DetectConfig | None = None) -> None:
        self.config = config or DetectConfig()
        self._detector: Any = None

    @property
    def detector(self) -> Any:
        if self._detector is None:
            from dimos.perception.detection.detectors.owlv2 import Owlv2Detector

            # A Configurable builds its own config from keyword arguments; handing it a
            # ready-made one is rejected as an extra input.
            settings: dict[str, Any] = {"model_name": self.config.checkpoint}
            if self.config.device:
                settings["device"] = self.config.device
            self._detector = Owlv2Detector(**settings)
        return self._detector

    def best(
        self, image: Image, text: str
    ) -> tuple[tuple[float, float, float, float], float] | None:
        """The strongest box for *text*, or nothing if the detector refuses the image."""
        found = self.detector.query_detections(image, [text], threshold=self.config.threshold)
        if not found.detections:
            return None
        best = max(found.detections, key=lambda detection: detection.confidence)
        x1, y1, x2, y2 = (float(v) for v in best.bbox)
        return (x1, y1, x2, y2), float(best.confidence)


def detect_episode(
    episode: Episode,
    query: str,
    frames: RecordingFrames,
    boxes: Owlv2Boxes,
    *,
    rank: int,
    config: DetectConfig,
    keep_image: bool = False,
) -> Detection:
    """Run the detector over one episode's best frames until one of them answers.

    A frame can fail twice over: the detector may refuse it, or it may be detected and
    then not placeable because the stereo gave nothing back inside the box (glass,
    a dark shelf, a shiny floor). Either way the next-best frame of the same episode is
    another look at the same thing, so both failures fall through rather than ending it.
    A detection that was found but never placed is kept as the answer of last resort.
    """
    peak = episode.peak
    detection = Detection(
        query=query,
        rank=rank,
        ts=peak.ts,
        camera_frame=peak.frame,
        episode_frames=len(episode.frames),
        episode_span=episode.span,
        episode_score=episode.score,
        models=sorted(episode.members),
        attempts=0,
    )
    flat: Detection | None = None
    refusals = 0
    for candidate in episode.by_weight()[: max(1, config.attempts)]:
        detection.attempts += 1
        image = frames.color(candidate.ts)
        if image is None:
            detection.note = "no colour frame at that stamp"
            continue
        found = boxes.best(image, query)
        if found is None:
            refusals += 1
            detection.note = f"detector refused {refusals} frame(s)"
            continue
        attempt = replace(detection, ts=candidate.ts, camera_frame=candidate.frame, note="")
        attempt.box2d, attempt.score = found
        if keep_image:
            attempt.image = image
        _place(attempt, candidate, image, frames, config)
        if attempt.box3d is not None:
            return attempt
        flat = flat or attempt
    return flat or detection


def _place(
    detection: Detection,
    frame: Frame,
    image: Image,
    frames: RecordingFrames,
    config: DetectConfig,
) -> None:
    """Turn a 2D box into a 3D one, or say why it could not be."""
    assert detection.box2d is not None
    depth = frames.depth(frame.frame, frame.ts)
    if depth is None:
        detection.note = "no depth frame paired with that image"
        return
    intrinsics = frames.intrinsics.get(frame.frame)
    if intrinsics is None:
        detection.note = f"no camera_info for {frame.frame!r}"
        return
    rgb = np.asarray(image.to_rgb().data)
    measured = object_points(
        detection.box2d,
        (int(rgb.shape[1]), int(rgb.shape[0])),
        depth,
        intrinsics,
        max_depth_m=config.max_depth_m,
        band_m=config.depth_band_m,
        min_pixels=config.min_depth_pixels,
    )
    if measured is None:
        detection.note = "no usable depth inside the box"
        return
    points, median = measured
    pose = frames.pose(frame.frame, frame.ts, config.world_frame)
    if pose is None:
        detection.note = f"no transform {config.world_frame} <- {frame.frame}"
        return
    detection.box3d = box_from_points(
        points, pose, config.world_frame, median, trim_percentile=config.trim_percentile
    )


def merge_duplicates(detections: Sequence[Detection], merge_m: float = 0.75) -> int:
    """Mark detections that are another look at the same thing. Returns how many places.

    Episodes are split on time, deliberately: the trolley passes the cheese counter
    four times and that is four chances at it rather than one. But the four answers are
    one place, so the last step is to say so -- in 3D, where "the same place" means
    something, rather than in the episode split, where it would cost the extra chances.

    The strongest detection of a group keeps its rank and the rest point at it; nothing
    is dropped, because a second look is evidence and a caller may want to show it.
    """
    placed = [d for d in detections if d.box3d is not None]
    for detection in detections:
        detection.duplicate_of = None
    for detection in sorted(placed, key=lambda d: -d.score):
        if detection.duplicate_of is not None:
            continue
        assert detection.box3d is not None
        here = np.asarray(detection.box3d.centre)
        for other in placed:
            if other is detection or other.duplicate_of is not None:
                continue
            assert other.box3d is not None
            if float(np.linalg.norm(np.asarray(other.box3d.centre) - here)) <= merge_m:
                other.duplicate_of = detection.rank
    return sum(1 for d in placed if d.duplicate_of is None)


def find(
    store: Any,
    recording: Any,
    query: str,
    *,
    config: DetectConfig | None = None,
    models: Sequence[str] | None = None,
    towers: Any = None,
    frames: RecordingFrames | None = None,
    boxes: Owlv2Boxes | None = None,
    keep_images: bool = False,
) -> Iterator[Detection]:
    """The whole chain, yielding one detection per episode as it is found.

    *store* holds the patch index, *recording* the pictures. They are usually the same
    file -- a recording indexes itself -- but an .mcap keeps its index alongside.
    """
    from dimos.mapping.hyperspace.frames import hot_frames, ranked_episodes

    config = config or DetectConfig()
    frames = frames or RecordingFrames(recording, config=config)
    boxes = boxes or Owlv2Boxes(config)
    matched = hot_frames(store, query, towers=towers, models=models)
    if not matched:
        return
    found = ranked_episodes(
        matched,
        gap_s=config.episode_gap_s,
        min_frames=config.min_episode_frames,
        limit=config.max_episodes,
    )
    for rank, episode in enumerate(found, 1):
        yield detect_episode(
            episode,
            query,
            frames,
            boxes,
            rank=rank,
            config=config,
            keep_image=keep_images,
        )
