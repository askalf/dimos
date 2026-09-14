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

"""Hyperspace as two memory modules.

``HyperspacePatches`` watches colour + depth + tf, keeps the frames worth
keeping, and writes each kept frame's 576 text-aligned patch embeddings into
the recording's memory store (one vector per patch, so the store's own vector
search finds them). ``Hyperspace`` answers questions against that store: text
in, scored voxels out, placing every keyframe through the recorded tf at query
time: a keyframe stores no pose, so corrected transforms would move old answers
once the transform buffer lets a rewrite win (see test_rewriting_tf_moves_the_answer).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.agents.annotation import skill
from dimos.agents.skill_result import SkillResult
from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.mapping.hyperspace import patches as hs
from dimos.mapping.hyperspace.embedder import (
    DEFAULT_MEMBERS,
    SIGLIP2_MODEL_NAME,
    PatchEnsemble,
)
from dimos.mapping.hyperspace.ingest import IngestConfig, PatchIngestor, transform_to_matrix
from dimos.mapping.hyperspace.live import LiveConfig, LiveQuery
from dimos.mapping.hyperspace.msgs import FoundObjects
from dimos.mapping.hyperspace.query import HyperspaceQuery
from dimos.mapping.hyperspace.refine import refine_config_of
from dimos.memory.module import MemoryModule, MemoryModuleConfig
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.String import String
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from numpy.typing import NDArray

logger = setup_logger()


def pick_device(device: str, *, allow_mps: bool = True) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def depth2depth_model_of(name: str) -> str:
    """ "" = off, "default" = the package's checkpoint, anything else verbatim."""
    if name != "default":
        return name
    from dimos.perception.depth2depth.fusion import DEPTH_MODEL_NAME

    return DEPTH_MODEL_NAME


class HyperspacePatchesConfig(MemoryModuleConfig):
    # The checkpoints that embed every keyframe, Hugging Face ids or local
    # directories, NaFlex ones optionally with "@<patch budget>". More than
    # one makes an ensemble whose scores are pooled per cell at query time
    # (embedder.PatchEnsemble); the default pair is what the size sweep
    # picked. ["google/siglip2-so400m-patch16-384"] is the original single model.
    models: list[str] = DEFAULT_MEMBERS
    # Kept for callers that predate `models`; used only when `models` is empty.
    model_name: str = SIGLIP2_MODEL_NAME
    # "auto" = cuda if available, else cpu (never mps inside a worker, see start()).
    device: str = "auto"
    # Frame the quality gate measures camera motion against.
    motion_reference_frame: str = "odom"
    # Never embed frames closer together than this (s), which is also the ceiling on
    # the keyframe rate: novelty spikes the keeps up to here and no further.
    min_frame_interval_s: float = 1.0 / hs.MAX_KEYFRAME_HZ

    # Keyframe gate. Negative turns a gate off.
    # Frames held after a candidate to swap a blurry one for a sharp one; the rate
    # comes from novelty, not from this.
    lookahead: int = 2
    quality_margin: float = 0.25
    novelty_threshold: float = 0.30
    # Negative = off, which is the default: see KeyframeGateConfig.
    patch_novelty_threshold: float = -1.0
    max_angular_velocity: float = 1.5
    max_linear_velocity: float = -1.0
    max_dark_fraction: float = 0.6
    max_bright_fraction: float = -1.0
    min_keyframe_interval: float = 1.0 / hs.MAX_KEYFRAME_HZ

    # Depth readings beyond this (m) are holes: RealSense frames carry 65535 mm
    # "no reading" sentinels and occasional 20-40 m glitches.
    max_depth_m: float = 10.0
    # Fill those holes from the colour frame before measuring patch depth: a
    # depth-anything checkpoint id (see dimos.perception.depth2depth), or ""
    # for raw sensor depth. "default" takes the package's own checkpoint.
    depth2depth_model: str = ""
    # A colour frame pairs with the depth frame within this many seconds of it.
    depth_max_dt: float = 0.05
    depth_history: int = 64
    # Stride of the depth thumbnail kept per keyframe, for scene rendering.
    depth_thumbnail_stride: int = 4
    # Keep the colour frame behind each embedding frame, so the detector can be shown
    # it later. Required for a LIVE run and pointless for an ingest of a recording:
    # see IngestConfig.keep_frames.
    keep_frames: bool = False
    # The layout the detector path reads. Live it has to be on -- a keyframe blob is
    # not searchable one patch at a time -- so it defaults on here rather than in
    # IngestConfig, which still has every old recording to think about.
    flat: bool = True


def _optional(value: float) -> float | None:
    return None if value < 0 else value


def store_members(store: Any, wait_s: float = 0.0) -> list[str]:
    """The checkpoint specs an existing store's keyframes were embedded with,
    or [] when the store is empty or predates the member record. Waits up to
    ``wait_s`` for the first keyframe when the writer starts alongside."""
    from dimos.mapping.hyperspace.ingest import KEYFRAME_STREAM

    deadline = time.monotonic() + wait_s
    while True:
        if KEYFRAME_STREAM in store.list_streams():
            first = next(iter(store.stream(KEYFRAME_STREAM, dict).order_by("ts")), None)
            if first is not None:
                return list(first.data.get("member_specs", []))
        if time.monotonic() >= deadline:
            return []
        time.sleep(0.5)


def open_store_with_retry(module: MemoryModule, attempts: int = 20, wait_s: float = 0.5) -> None:
    """Touch ``module.store`` until it opens. The writer and the reader start in
    parallel on the same fresh file, and sqlite refuses the second
    ``PRAGMA journal_mode=WAL`` while the first is still creating it."""
    import sqlite3

    for attempt in range(attempts):
        try:
            _ = module.store
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) or attempt == attempts - 1:
                raise
            module._store = None
            time.sleep(wait_s)


class HyperspacePatches(MemoryModule):
    """Keeps the frames worth keeping and writes their patch embeddings to memory."""

    config: HyperspacePatchesConfig

    color_image: In[Image]
    depth_image: In[Image]
    camera_info: In[CameraInfo]
    depth_camera_info: In[CameraInfo]
    tf: In[TFMessage]

    ingestor: PatchIngestor | None = None

    @rpc
    def start(self) -> None:
        # Everything the handlers need exists before Module.start binds them:
        # frames arrive the moment the ports connect.
        # "auto" never picks MPS: Metal asserts inside dimos's forkserver
        # workers on macOS (MPSKernelDAG.mm failed assertion) and the worker
        # dies without a traceback. Pass device="mps" to try anyway.
        device = pick_device(self.config.device, allow_mps=False)
        specs = self.config.models or [self.config.model_name]
        logger.info(f"hyperspace patches: loading {specs} on {device}")
        self.model = self.register_disposable(PatchEnsemble(specs, device=device, towers="vision"))
        self.model.start()
        logger.info(f"hyperspace patches: model ready, opening {self.config.db_path}")
        open_store_with_retry(self)
        gate = hs.KeyframeGateConfig(
            lookahead=self.config.lookahead,
            quality_margin=self.config.quality_margin,
            novelty_threshold=self.config.novelty_threshold,
            patch_novelty_threshold=_optional(self.config.patch_novelty_threshold),
            max_angular_velocity=_optional(self.config.max_angular_velocity),
            max_linear_velocity=_optional(self.config.max_linear_velocity),
            max_dark_fraction=_optional(self.config.max_dark_fraction),
            max_bright_fraction=_optional(self.config.max_bright_fraction),
            min_interval=_optional(self.config.min_keyframe_interval),
        )
        self.ingestor = PatchIngestor(
            self.store,
            self.model,
            IngestConfig(
                gate=gate,
                motion_reference_frame=self.config.motion_reference_frame,
                min_frame_interval_s=self.config.min_frame_interval_s,
                max_depth_m=self.config.max_depth_m,
                depth2depth_model=depth2depth_model_of(self.config.depth2depth_model),
                depth_max_dt=self.config.depth_max_dt,
                depth_history=self.config.depth_history,
                depth_thumbnail_stride=self.config.depth_thumbnail_stride,
                keep_frames=self.config.keep_frames,
                flat=self.config.flat,
            ),
            lookup=self._lookup,
        )
        logger.info(f"hyperspace patches: {self.model.tags} on {device}, db {self.config.db_path}")
        super().start()

    def _lookup(self, target: str, source: str, ts: float) -> NDArray[np.float64] | None:
        try:
            transform = self.tfbuffer.get(target, source, ts, warn=False)
        except (TypeError, RuntimeError):
            return None
        return None if transform is None else transform_to_matrix(transform)

    async def handle_camera_info(self, info: CameraInfo) -> None:
        if self.ingestor is not None:
            self.ingestor.add_camera_info(info)

    async def handle_depth_camera_info(self, info: CameraInfo) -> None:
        if self.ingestor is not None:
            self.ingestor.add_camera_info(info)

    async def handle_tf(self, msg: TFMessage) -> None:
        if self.ingestor is not None:
            self.ingestor.add_tf(msg)

    async def handle_depth_image(self, image: Image) -> None:
        if self.ingestor is not None:
            self.ingestor.add_depth(image)

    async def handle_color_image(self, image: Image) -> None:
        if self.ingestor is None:
            return
        if self.ingestor.stats["images"] == 0:
            logger.info(
                f"hyperspace patches: first colour frame {image.frame_id} {image.width}x{image.height} {image.format}"
            )
        # Embedding blocks for ~50 ms on a GPU and ~1 s on a CPU; the handler's
        # latest-only dispatch drops the frames that arrive meanwhile.
        await asyncio.get_running_loop().run_in_executor(None, self.ingestor.add_image, image)

    @rpc
    def flush(self) -> int:
        """End of stream: judge what is still buffered. Returns keyframes added."""
        return self.ingestor.flush() if self.ingestor is not None else 0

    @rpc
    def ingest_stats(self) -> dict[str, int]:
        return dict(self.ingestor.stats) if self.ingestor is not None else {}


class HyperspaceConfig(MemoryModuleConfig):
    # Text towers to load: normally left empty, and taken from the store's
    # keyframes (written by HyperspacePatches, whose `models` decide). Set it
    # only to query a store that predates the member record.
    models: list[str] = []
    model_name: str = SIGLIP2_MODEL_NAME
    # "auto" = cuda if available, else cpu (never mps, see start()).
    device: str = "auto"
    # Ensemble stores: how the members' cell scores combine ("min", "2nd",
    # "mean") and the threshold on the pooled score. See QueryConfig.
    pool: str = "min"
    pooled_hot_threshold: float = 0.005
    # Frame answers are given in unless a request names another.
    world_frame: str = "odom"
    voxel_size: float = 0.10
    hot_threshold: float = 0.02
    max_hot_patches: int = 6000
    cap_near: float = 0.99
    cap_far: float = 1.01
    # Comma separated; empty uses the indoor defaults.
    background_prompts: str = ""
    # The segment channel (HyperspaceSegments records) added on top; 0 = off.
    segment_weight: float = 1.0
    segment_min_z: float = 2.0
    # Refinement chain (see refine.py); "default" = QueryConfig.refine, "none" = raw map.
    refine: str = "default"
    # Keyframes a voxel must be seen from (the chain's "support" step). 2 was
    # the single-model setting; an ensemble's hot cells are already vetted by
    # several checkpoints and sit tighter on the object, so their thin
    # pyramids overlap less between views: 1 keeps the recall (plan.md 7).
    refine_min_frames: int = 1
    # Depth samples a voxel needs to appear in scene_map.
    scene_min_samples: int = 3
    # --- the detector path (`find_objects`) -------------------------------------
    # OWLv2's per-box acceptance score. See DetectConfig.threshold for what half
    # costs: it is a real refusal, and the scores are not comparable across words.
    owl_threshold: float = 0.5
    # An OWLv2 checkpoint other than the base ensemble. "" = DetectConfig's own.
    owl_checkpoint: str = ""
    # Where the detector runs. Separate from `device`, which is the text towers':
    # the towers are small enough for a cpu and OWLv2 is not.
    owl_device: str = "auto"
    # Models the frames-first search ranks with. [] = every model in the index,
    # which is what the three-way agreement wants.
    detect_models: list[str] = []
    # Episodes to spend a detector call on, frames of each to try, and frames per
    # forward pass. The detector is ~90% of a query, so these are the cost.
    max_episodes: int = 12
    detect_attempts: int = 3
    detect_batch: int = 1
    min_episode_frames: int = 2
    episode_gap_s: float = 1.0
    # Depth spread that is still the object (m), and how close two answers have to
    # be to be one place.
    depth_band_m: float = 0.5
    merge_m: float = 0.75
    # Fill stereo's holes as boxes are placed, for a store nobody has run `fill_depth`
    # over: "auto" only when there is no filled-depth stream, "" off, or a checkpoint
    # by name. Live there usually is one -- the depth2depth module writes it as the
    # robot drives -- and then this costs nothing.
    detect_depth2depth: str = "auto"

    # Demo: after this many seconds, run demo_queries and publish the answers,
    # then repeat every demo_every_s. 0 disables.
    demo_after_s: float = 0.0
    demo_every_s: float = 30.0
    demo_queries: list[str] = ["a chair"]


class Hyperspace(MemoryModule):
    """Ask the map where something is; get scored voxels back.

    Queries arrive on ``query`` as JSON (``{"id": 7, "text": "a chair"}``) or
    bare text, or through the ``find`` skill. Answers go out on
    ``query_result`` (voxel centers with an ``intensity`` score, for viewers)
    and ``query_answer`` (JSON: the request id, the text, the voxel count and
    the best voxels). A caller pairs answers to requests by the id in
    ``query_answer``; there is no RPC and no blocking call.
    """

    config: HyperspaceConfig

    query: In[String]
    tf: In[TFMessage]
    query_result: Out[PointCloud2]
    query_answer: Out[String]
    scene_map: Out[PointCloud2]
    # Every answer `find_objects` gives, also published, so a viewer or a recorder
    # sees them without having made the call.
    found: Out[FoundObjects]

    @rpc
    def start(self) -> None:
        # Module.start runs main() up to its first yield, so the engine must
        # exist first.
        # No MPS here: two workers bringing up torch on Metal at the same time
        # lose one of them silently on macOS, and HyperspacePatches needs the
        # GPU more. The text tower is fast enough on the CPU.
        device = pick_device(self.config.device, allow_mps=False)
        logger.info(f"hyperspace query: opening {self.config.db_path}")
        open_store_with_retry(self)
        specs = self.config.models or store_members(self.store) or [self.config.model_name]
        logger.info(f"hyperspace query: loading text towers {specs} on {device}")
        self.model = self.register_disposable(PatchEnsemble(specs, device=device, towers="text"))
        self.model.start()
        query_config = hs.QueryConfig(
            hot_threshold=self.config.hot_threshold,
            pool=self.config.pool,
            pooled_hot_threshold=self.config.pooled_hot_threshold,
            max_hot_patches=self.config.max_hot_patches,
            cap_near=self.config.cap_near,
            cap_far=self.config.cap_far,
            segment_weight=self.config.segment_weight,
            segment_min_z=self.config.segment_min_z,
        )
        prompts = [p.strip() for p in self.config.background_prompts.split(",") if p.strip()]
        if prompts:
            query_config.background_prompts = prompts
        self.engine = HyperspaceQuery(
            self.store,
            self.model.embed_text,
            query_config,
            world_frame=self.config.world_frame,
            voxel_size=self.config.voxel_size,
            refine_config=refine_config_of(
                self.config.refine, query_config.refine, min_frames=self.config.refine_min_frames
            ),
        )
        self._lock = threading.Lock()
        self._ids = iter(range(1, 1 << 30))
        self._load_the_detector(specs)
        logger.info(
            f"hyperspace query: {self.model.tags} text towers on {device}, db {self.config.db_path}"
        )
        super().start()

    def _load_the_detector(self, specs: list[str]) -> None:
        """Everything `find_objects` needs, loaded now rather than on the first question.

        OWLv2 is two and a half seconds to bring up and the text towers thirteen, and
        the transforms are a pass over the store. Left lazy, all of that lands on
        whoever asks first -- which on a robot is someone waiting on an answer.
        """
        from dimos.mapping.hyperspace.detect import DetectConfig

        device = pick_device(self.config.owl_device, allow_mps=False)
        self.live = LiveQuery(
            self.store,
            LiveConfig(
                detect=DetectConfig(
                    threshold=self.config.owl_threshold,
                    checkpoint=self.config.owl_checkpoint or DetectConfig.checkpoint,
                    device=device,
                    attempts=self.config.detect_attempts,
                    batch=self.config.detect_batch,
                    max_episodes=self.config.max_episodes,
                    min_episode_frames=self.config.min_episode_frames,
                    episode_gap_s=self.config.episode_gap_s,
                    depth_band_m=self.config.depth_band_m,
                    depth2depth=self.config.detect_depth2depth,
                    max_depth_m=self.config.max_depth_m,
                    world_frame=self.config.world_frame,
                ),
                models=list(self.config.detect_models),
                merge_m=self.config.merge_m,
            ),
        )
        self.register_disposable(self.live)
        loaded = self.live.warm(self.config.detect_models or specs)
        logger.info(
            f"hyperspace detect: OWLv2 on {device} in {loaded['detector']:.1f}s, text towers "
            f"in {loaded['towers']:.1f}s, {int(loaded['index'])} patches already indexed, "
            f"threshold {self.config.owl_threshold}"
        )

    @rpc
    def find_objects(self, text: str, top: int = 0) -> FoundObjects:
        """Where is `text`? Boxes in the world, with the frames that found them.

        The whole chain, on the map as it stands this instant: the patch index ranks
        frames, the models' agreement picks the places worth looking at, OWLv2 draws a
        box on the best frame of each, and that box plus the frame's depth becomes a box
        in the world. Answers come back strongest-first, one per place.

        The same object `dimos map live` drives, so what a page shows offline is what
        the robot does.
        """
        self.live.config.top = top
        result = self.live.ask(text)
        self.found.publish(result)
        return result

    def answer(
        self, text: str, request_id: int | None = None, frame: str | None = None
    ) -> dict[str, Any]:
        """Run one query and publish it on both outputs. Returns the JSON answer."""
        with self._lock:
            request_id = next(self._ids) if request_id is None else request_id
            started = time.monotonic()
            answer = self.engine.answer(text, request_id, frame)
            result: hs.Heatmap = answer.pop("heatmap")
            cloud = PointCloud2.from_numpy(
                result.centres().astype(np.float32),
                frame_id=result.frame,
                timestamp=time.time(),
                intensities=result.scores().astype(np.float32),
            )
            self.query_result.publish(cloud)
            answer["ms"] = round((time.monotonic() - started) * 1000)
            self.query_answer.publish(String(json.dumps(answer)))
            logger.info(
                f"hyperspace {text!r}: {answer['voxels']} voxels in {answer['ms']} ms {result.stats}"
            )
            return answer

    async def handle_tf(self, msg: TFMessage) -> None:
        # The subscription TF(port, buffer_size=inf) would make, without its
        # transport timing: transforms land in the buffer the answers read,
        # which the store still tops up with whatever predates this module.
        self.engine.tf.receive_tfmessage(msg)

    async def handle_query(self, msg: String) -> None:
        payload = msg.data.strip()
        request_id, text, frame = None, payload, None
        if payload.startswith("{"):
            try:
                parsed = json.loads(payload)
                request_id = int(parsed.get("id", 0))
                text = str(parsed.get("text", "")).strip()
                frame = parsed.get("frame") or None
            except (ValueError, TypeError, AttributeError) as error:
                logger.warning(f"hyperspace ignored a query: {error}: {payload!r}")
                return
        if not text:
            return
        await asyncio.get_running_loop().run_in_executor(None, self.answer, text, request_id, frame)

    @skill
    def find(self, text: str, top: int = 10) -> SkillResult:
        """Where in the map is `text`? E.g. "a traffic cone", "the red chair".

        Returns the best-scoring voxel centers (meters, in the world frame) and
        how many voxels answered. Also publishes the full answer on
        ``query_result`` / ``query_answer`` for anything listening.
        """
        text = text.strip()
        if not text:
            return SkillResult.fail("INVALID_INPUT", "text must not be empty")
        answer = self.answer(text)
        best = answer["best"][:top]
        if not best:
            return SkillResult.ok(
                f"nothing in the map looks like {text!r}",
                query=text,
                voxels=0,
                stats=answer["stats"],
            )
        return SkillResult.ok(
            f"{text!r}: {answer['voxels']} voxels, best at {best[0]['xyz']} (score {best[0]['score']})",
            query=text,
            frame=answer["frame"],
            voxels=answer["voxels"],
            best=best,
            stats=answer["stats"],
        )

    def publish_scene(self, frame: str | None = None) -> int:
        """Occupied voxels from the keyframes' depth thumbnails, for viewers."""
        voxels = self.engine.scene_voxels(frame, self.config.scene_min_samples)
        if not voxels:
            return 0
        size = self.config.voxel_size
        centres = (np.asarray([i for i, _ in voxels], dtype=np.float32) + 0.5) * size
        most = max(n for _, n in voxels)
        self.scene_map.publish(
            PointCloud2.from_numpy(
                centres,
                frame_id=frame or self.config.world_frame,
                timestamp=time.time(),
                intensities=np.asarray([n / most for _, n in voxels], dtype=np.float32),
            )
        )
        return len(voxels)

    async def main(self) -> AsyncIterator[None]:
        # Code before the first yield is startup; the demo loop must not block it.
        demo = None
        if self.config.demo_after_s > 0 and self.config.demo_queries:
            demo = asyncio.create_task(self._demo_loop())
        try:
            yield
        finally:
            if demo is not None:
                demo.cancel()

    async def _demo_loop(self) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(self.config.demo_after_s)
        while True:
            scene = await loop.run_in_executor(None, self.publish_scene)
            logger.info(f"hyperspace demo: scene_map {scene} voxels")
            for text in self.config.demo_queries:
                answer = await loop.run_in_executor(None, self.answer, text)
                logger.info(
                    f"hyperspace demo {text!r}: {answer['voxels']} voxels, best {answer['best'][:1]}"
                )
            await asyncio.sleep(self.config.demo_every_s)
