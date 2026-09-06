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

"""Explicit simulator-truth recording and physical checks for authored evals."""

from itertools import pairwise, product
import time

import numpy as np
from scipy.spatial.transform import Rotation

from dimos.core.core import rpc
from dimos.core.stream import In
from dimos.memory.module import Recorder
from dimos.memory.store.base import Store
from dimos.sim2.scene_types import EntityState, RegionState, SceneState


class SceneRecorder(Recorder):
    sim_truth: In[SceneState]
    _ready: bool = False

    @rpc
    def start(self) -> None:
        super().start()
        self._ready = True

    @rpc
    def recording_ready(self) -> bool:
        return self._ready

    @rpc
    def stop(self) -> None:
        self._ready = False
        super().stop()


sim2_eval_recording = SceneRecorder.blueprint(
    record_tf=False,
    poseless_streams=["sim_truth"],
    stream_codecs={"sim_truth": "pickle"},
)


def fresh_states(store: Store, initial: SceneState, *, dwell_s: float = 0.5) -> list[SceneState]:
    """Return a recent observation window from exactly this world/generation.

    Missing data raises LookupError so InteractiveEval waits. Stale data, a
    changed episode or a stopped physics clock is an error, never a score.
    """
    stream = store.streams.sim_truth
    last: SceneState = stream.last().data
    if time.time() - last.ts > 2:
        raise RuntimeError("simulator truth is stale; check SimulationModule and SceneRecorder")
    if (last.world_id, last.generation) != (initial.world_id, initial.generation):
        raise RuntimeError("recording belongs to a different world or reset generation")
    if last.tick <= initial.tick:
        raise LookupError("waiting for physics after setup")
    states = [
        observation.data
        for observation in stream.from_timestamp(last.ts - dwell_s - 0.2)
        if observation.data.world_id == initial.world_id
        and observation.data.generation == initial.generation
        and observation.ts >= initial.ts
    ]
    if not states or states[-1].sim_time - states[0].sim_time < dwell_s:
        raise LookupError("waiting for the physical dwell window")
    if any(b.ts - a.ts > 0.3 for a, b in pairwise(states)):
        raise LookupError("waiting for an uninterrupted 10 Hz truth window")
    return states


def inside_region(point: tuple[float, float, float], region: RegionState) -> bool:
    local = (
        Rotation.from_quat(region.pose.orientation.to_tuple())
        .inv()
        .apply(np.asarray(point) - region.pose.position.to_tuple())
    )
    return bool(np.all(np.abs(local) <= np.asarray(region.size) / 2))


def contained(entity: EntityState, region: RegionState) -> bool:
    """Conservative containment of the entire world-space object bounds."""
    corners = product(*zip(entity.bounds_min, entity.bounds_max, strict=True))
    return all(inside_region((corner[0], corner[1], corner[2]), region) for corner in corners)


def touching(state: SceneState, entity: str, body_prefix: str) -> bool:
    return any(
        (a == entity and b.startswith(body_prefix)) or (b == entity and a.startswith(body_prefix))
        for a, b in state.contacts
    )
