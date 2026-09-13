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
import time
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
    # Frames handed to the detector in one forward pass. One, because batching was
    # measured on this Mac and LOST: OWLv2 pads every frame to 960x960, so the cost is
    # per pixel and there is little per-call overhead to amortize, while the bigger
    # activation tensor pushes MPS around -- 275 ms a frame at 1, 300 at 4, 514 at 12.
    # The knob stays because a CUDA box with headroom is the case where it should win;
    # raise it there and measure before believing it.
    batch: int = 1
    # Episodes considered before the strongest `max_episodes` of them are detected.
    # Only bounds the work of placing them; a query with more candidates than this is
    # already answering about a very common thing.
    episode_pool: int = 500
    # Group candidate episodes by roughly where they are and give every group a look
    # before any group gets a second one. Off means strongest-first, which spends the
    # detector on four looks at the nearest chair before it has seen the far one.
    spread_places: bool = True
    # Cells whose depth is within this of the strongest hit's are the same surface it
    # is on. What keeps a cone's cells and drops the floor under it.
    place_band_m: float = 0.3
    # How close two episodes have to be to count as the same place for that ordering.
    # Deliberately looser than the radius answers are merged at: this estimate comes
    # from patch rays before any detector has looked, and one cone at a metre placed
    # itself over a 1.35 m spread, so 0.75 split it three ways and spent three looks on
    # it. Measured on that cone against the two that exist: at 0.75 the first twelve
    # held two looks at the near cone and reached the far one seventh; at 1.5 one look
    # and fifth; at 2.0, fourth; at 3.0, third. Wider costs a genuinely separate object
    # 2 m away its turn in the first round, so this stops at the first value that fixed
    # the repeat.
    place_radius_m: float = 1.5
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
    # Which place in the world this answer belongs to. Stable across the query, so a
    # caller can tell a new thing from a better look at a thing it already has.
    place_id: int | None = None
    # That place's box once this answer is folded in: a second look sharpens the box
    # rather than adding another one beside it.
    refined: Box3D | None = None
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
            "arrived": self.arrived,
            "duplicate_of": self.duplicate_of,
            "place_id": self.place_id,
            "refined": None if self.refined is None else self.refined.as_dict(),
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

    def warm(self) -> float:
        """Do the first lookup's work now, while nobody is waiting on an answer.

        A recording's transforms are read in one pass the first time anything is placed
        -- a quarter of a million rows on sf_office -- and the image streams build their
        by-stamp lookup on first use. Together that was three seconds charged to whoever
        asked the first question. Returns the seconds spent.
        """
        started = time.monotonic()
        self.load_tf()
        for name in (self.color_stream, self.depth_stream):
            if name not in self.recording.list_streams():
                continue
            first = self.recording.streams[name].order_by("ts").limit(1).to_list()
            if first:
                self.recording.streams[name].at(
                    float(first[0].ts), tolerance=self.config.depth_max_dt
                ).to_list()
        return time.monotonic() - started

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
    """Core's OWLv2 detector, loaded once and asked about whole rounds of frames."""

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

    def warm(self) -> float:
        """Load the weights and run one frame through, before anyone is waiting.

        Building the detector is lazy twice over -- the object defers the model, and the
        model defers the weights until something is detected -- so the first real query
        of a process paid about five seconds that had nothing to do with it. A blank
        frame costs one forward pass and moves that cost to startup, where a wait is
        free. Returns the seconds spent, for a caller that wants to say so.
        """
        started = time.monotonic()
        blank = Image.from_numpy(np.zeros((32, 32, 3), dtype=np.uint8), frame_id="warmup", ts=0.0)
        self.best_many([blank], "a thing")
        return time.monotonic() - started

    def best(
        self, image: Image, text: str
    ) -> tuple[tuple[float, float, float, float], float] | None:
        """The strongest box for *text*, or nothing if the detector refuses the image."""
        return self.best_many([image], text)[0]

    def best_many(
        self, images: Sequence[Image], text: str
    ) -> list[tuple[tuple[float, float, float, float], float] | None]:
        """``best`` over many images, in as few forward passes as the cap allows.

        One answer per image, in input order, so a caller can keep its own bookkeeping
        beside the list. The images of one round are unrelated to each other -- this is
        purely about paying the per-call overhead once instead of once per frame.
        """
        answers: list[tuple[tuple[float, float, float, float], float] | None] = []
        size = max(1, self.config.batch)
        for start in range(0, len(images), size):
            chunk = list(images[start : start + size])
            for found in self.detector.query_detections_batch(
                chunk, [text], threshold=self.config.threshold
            ):
                if not found.detections:
                    answers.append(None)
                    continue
                best = max(found.detections, key=lambda detection: detection.confidence)
                x1, y1, x2, y2 = (float(v) for v in best.bbox)
                answers.append(((x1, y1, x2, y2), float(best.confidence)))
        return answers


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
    Detection(
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
    return detect_episodes(
        [episode], query, frames, boxes, config=config, keep_images=keep_image, first_rank=rank
    )[0]


@dataclass
class _Try:
    """One episode part-way through its attempts, so a round can be shared."""

    detection: Detection
    candidates: list[Frame]
    answer: Detection | None = None
    flat: Detection | None = None
    refusals: int = 0

    @property
    def settled(self) -> bool:
        return self.answer is not None

    def finish(self) -> Detection:
        return self.answer or self.flat or self.detection


def place_of(
    episode: Episode, frames: RecordingFrames, world_frame: str, *, band_m: float = 0.3
) -> NDArray[np.float64] | None:
    """Roughly where an episode's match is, before any detector has looked at it.

    A hot patch already carries the ray through its cell and the depth the sensor read
    there, which is all the dense path ever had; one transform puts it in the world.

    The hot cells are not the object. Measured on one cone: a 24x42 grid over an
    848x480 frame makes a cell about 11 cm of scene at 2.3 m, so a 40 cm cone is four
    cells -- and a hundred cells came back hot, a tenth of the frame. Those extra cells
    are mostly the floor under and in front of it, at their own perfectly correct
    depths, so averaging over all of them lands between two surfaces: the strongest hit
    alone was 0.17 m from the cone where the median of the top five was 1.54 m.

    So the strongest hit picks the surface and *band_m* keeps the cells that agree with
    it, which the object's do and the floor's do not. The same move `object_points`
    makes inside the detector's box, one step earlier.
    """
    peak = episode.peak
    usable = sorted(
        (hit for hit in peak.hits if np.isfinite(hit.depth) and hit.depth > 0),
        key=lambda hit: -hit.score,
    )
    if not usable:
        return None
    surface = usable[0].depth
    kept = [hit for hit in usable if abs(hit.depth - surface) <= band_m]
    weights = np.array([max(hit.score, 1e-6) for hit in kept])
    points = np.array([[hit.ray[0] * hit.depth, hit.ray[1] * hit.depth, hit.depth] for hit in kept])
    here = (points * weights[:, None]).sum(axis=0) / weights.sum()
    pose = frames.pose(peak.frame, peak.ts, world_frame)
    if pose is None:
        return None
    return np.asarray(pose @ np.append(here, 1.0))[:3]


def spread_by_place(
    episodes: Sequence[Episode], frames: RecordingFrames, *, config: DetectConfig
) -> list[Episode]:
    """Order episodes so every place gets a look before any place gets a second one.

    Strongest-first spends the detector on whatever the camera saw most of: on
    sf_office, seven of twelve cone episodes were the same cone, while a second cone
    across the room never got a look. Grouping by where the patches say they are and
    taking one from each group in turn buys distinct answers with the same budget.

    Only the order changes. An episode the grouping gets wrong is detected sooner or
    later than it would have been, which is a different thing from being dropped, and
    the grouping leans on patch depth -- reliable up close, not at ten metres.
    """
    places = [
        place_of(episode, frames, config.world_frame, band_m=config.place_band_m)
        for episode in episodes
    ]
    groups: list[list[int]] = []
    centres: list[NDArray[np.float64] | None] = []
    for index, here in enumerate(places):
        joined = False
        if here is not None:
            for group, centre in zip(groups, centres, strict=True):
                if centre is None:
                    continue
                if float(np.linalg.norm(here - centre)) <= config.place_radius_m:
                    group.append(index)
                    joined = True
                    break
        if not joined:
            # An episode we could not place is its own group rather than dropped: not
            # knowing where it is says nothing about whether it is worth detecting.
            groups.append([index])
            centres.append(here)

    # The episodes arrive strongest-first, so the groups are already in that order and
    # so is each group's own list.
    ordered: list[Episode] = []
    for round_ in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if round_ < len(group):
                ordered.append(episodes[group[round_]])
    return ordered


def detect_episodes(
    episodes: Sequence[Episode],
    query: str,
    frames: RecordingFrames,
    boxes: Owlv2Boxes,
    *,
    config: DetectConfig,
    keep_images: bool = False,
    first_rank: int = 1,
) -> list[Detection]:
    """Every episode's answer, in rank order. See `stream_episodes` for the order of work."""
    return list(
        stream_episodes(
            episodes,
            query,
            frames,
            boxes,
            config=config,
            keep_images=keep_images,
            first_rank=first_rank,
        )
    )


def stream_episodes(
    episodes: Sequence[Episode],
    query: str,
    frames: RecordingFrames,
    boxes: Owlv2Boxes,
    *,
    config: DetectConfig,
    keep_images: bool = False,
    first_rank: int = 1,
) -> Iterator[Detection]:
    """Answers as they settle, in rank order.

    A detector call costs the same whether it is shown one frame or several, so when a
    batch is worth having the episodes go through it together: every unanswered one's
    next-best frame in a single pass, then the round after that. Nothing can be said
    about any of them until the round returns, so the answers arrive in a burst.

    At a batch of one -- the default, because batching measured slower on this
    hardware -- there is nothing to gather, and waiting would buy only a longer silence.
    So each episode is finished and handed back before the next one starts, and the
    first answer arrives after one detector call rather than after twelve.
    """
    tries = [
        _Try(
            detection=Detection(
                query=query,
                rank=rank,
                ts=episode.peak.ts,
                camera_frame=episode.peak.frame,
                episode_frames=len(episode.frames),
                episode_span=episode.span,
                episode_score=episode.score,
                models=sorted(episode.members),
                attempts=0,
            ),
            candidates=list(episode.by_weight()[: max(1, config.attempts)]),
        )
        for rank, episode in enumerate(episodes, first_rank)
    ]

    if config.batch <= 1:
        for one in tries:
            _attempt_rounds([one], query, frames, boxes, config=config, keep_images=keep_images)
            yield one.finish()
        return

    _attempt_rounds(tries, query, frames, boxes, config=config, keep_images=keep_images)
    for one in tries:
        yield one.finish()


def _attempt_rounds(
    tries: Sequence[_Try],
    query: str,
    frames: RecordingFrames,
    boxes: Owlv2Boxes,
    *,
    config: DetectConfig,
    keep_images: bool,
) -> None:
    """Take every unsettled episode through its next frame, until they run out."""
    for round_ in range(max(1, config.attempts)):
        pending: list[tuple[_Try, Frame, Image]] = []
        for attempt_of in tries:
            if attempt_of.settled or round_ >= len(attempt_of.candidates):
                continue
            candidate = attempt_of.candidates[round_]
            attempt_of.detection.attempts += 1
            image = frames.color(candidate.ts)
            if image is None:
                attempt_of.detection.note = "no colour frame at that stamp"
                continue
            pending.append((attempt_of, candidate, image))
        if not pending:
            break
        found_in_round = boxes.best_many([image for _, _, image in pending], query)
        for (attempt_of, candidate, image), found in zip(pending, found_in_round, strict=True):
            if found is None:
                attempt_of.refusals += 1
                attempt_of.detection.note = f"detector refused {attempt_of.refusals} frame(s)"
                continue
            attempt = replace(
                attempt_of.detection, ts=candidate.ts, camera_frame=candidate.frame, note=""
            )
            attempt.box2d, attempt.score = found
            if keep_images:
                attempt.image = image
            _place(attempt, candidate, image, frames, config)
            if attempt.box3d is not None:
                attempt_of.answer = attempt
            else:
                attempt_of.flat = attempt_of.flat or attempt


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
    """Group answers that are the same place, and sharpen each place as looks arrive.

    Episodes are split on time deliberately: the trolley passes the cheese counter four
    times and that is four chances at it. But the four answers are one place, so the
    last step is to say so -- in 3D, where "the same place" means something.

    A second look does not add a box beside the first. It joins the place, and the
    place's box becomes the average of its looks weighted by what the detector thought
    of each, so `refined` on any answer is that place's box as of that moment. Nothing
    is dropped: every answer keeps its own box too, because a second look is evidence.

    Answers are folded in the order they arrived rather than strongest-first, and the
    place belongs to the look that found it. A caller replaying a query then sees what
    a caller watching it saw.
    """
    placed = [detection for detection in detections if detection.box3d is not None]
    for detection in detections:
        detection.duplicate_of = None
        detection.place_id = None
        detection.refined = None

    places: list[dict[str, Any]] = []
    for detection in sorted(placed, key=lambda detection: detection.rank):
        assert detection.box3d is not None
        here = np.asarray(detection.box3d.centre, dtype=float)
        size = np.asarray(detection.box3d.extent, dtype=float)
        # A weak answer should not drag a place around, but a zero score still counts.
        weight = max(float(detection.score), 1e-6)
        joined = None
        for place in places:
            if float(np.linalg.norm(place["centre"] - here)) <= merge_m:
                joined = place
                break
        if joined is None:
            joined = {
                "id": len(places) + 1,
                "centre": here,
                "extent": size,
                "weight": weight,
                "first": detection.rank,
            }
            places.append(joined)
        else:
            total = joined["weight"] + weight
            joined["centre"] = (joined["centre"] * joined["weight"] + here * weight) / total
            joined["extent"] = (joined["extent"] * joined["weight"] + size * weight) / total
            joined["weight"] = total
            detection.duplicate_of = int(joined["first"])
        detection.place_id = int(joined["id"])
        detection.refined = Box3D(
            frame=detection.box3d.frame,
            centre=tuple(float(v) for v in joined["centre"]),
            extent=tuple(float(v) for v in joined["extent"]),
            pixels=detection.box3d.pixels,
            depth_m=detection.box3d.depth_m,
        )
    return len(places)


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
    resident: Any = None,
    timings: dict[str, float] | None = None,
) -> Iterator[Detection]:
    """The whole chain: text in, one detection per episode out, in rank order.

    *store* holds the patch index, *recording* the pictures. They are usually the same
    file -- a recording indexes itself -- but an .mcap keeps its index alongside.

    The episodes are detected as a group rather than one at a time, so results arrive
    together at the end rather than trickling out. That is the price of batching, and
    it is worth paying: the detector is a fixed cost per call, so twelve episodes in
    one pass is most of a query's detector time saved, while the trickle only ever
    bought a progress bar.
    """
    from dimos.mapping.hyperspace.frames import hot_frames, ranked_episodes

    config = config or DetectConfig()
    frames = frames or RecordingFrames(recording, config=config)
    boxes = boxes or Owlv2Boxes(config)
    asked = at = time.monotonic()
    matched = hot_frames(store, query, towers=towers, models=models, resident=resident)
    if timings is not None:
        timings["search"] = time.monotonic() - at
        timings["frames_matched"] = float(len(matched))
    if not matched:
        return
    at = time.monotonic()
    found = ranked_episodes(
        matched,
        gap_s=config.episode_gap_s,
        min_frames=config.min_episode_frames,
        limit=config.episode_pool if config.spread_places else config.max_episodes,
    )
    if config.spread_places:
        # Ordered before it is cut, or the cut is what decides which places are seen.
        found = spread_by_place(found, frames, config=config)
    found = found[: config.max_episodes]
    if timings is not None:
        timings["episodes"] = time.monotonic() - at
    at = time.monotonic()
    first: float | None = None
    for answer in stream_episodes(
        found, query, frames, boxes, config=config, keep_images=keep_images
    ):
        # From the question, not from the detector: this is when someone watching the
        # module would have seen the answer appear.
        answer.arrived = time.monotonic() - asked
        if first is None:
            first = time.monotonic() - at
        yield answer
    if timings is not None:
        timings["detect"] = time.monotonic() - at
        timings["first_result"] = first or 0.0
