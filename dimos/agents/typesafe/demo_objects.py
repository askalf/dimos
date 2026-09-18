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
"""Publish fixed world-frame objects as Detection3DArray: a stand-in for a 3D detector."""

from __future__ import annotations

import threading
import time
from typing import Any

from dimos_lcm.vision_msgs import (
    BoundingBox3D,
    Detection3D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from pydantic import Field

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray


class DemoObjectsConfig(ModuleConfig):
    # (label, x, y, z) world frame. DimSim apartment spawn is (3, 2) facing east; y=2 is clear for x in 1..5.
    objects: list[tuple[str, float, float, float]] = Field(
        default_factory=lambda: [("chair", 1.2, 2.0, 0.4)]
    )
    rate_hz: float = 2.0


class DemoObjects(Module):
    config: DemoObjectsConfig
    detections_3d: Out[Detection3DArray]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publish_loop, name="DemoObjects", daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            self._thread = None
        super().stop()

    def message(self) -> Detection3DArray:
        now = time.time()
        dets = []
        for label, x, y, z in self.config.objects:
            d = Detection3D()
            d.header = Header(now, "world")
            d.results = [
                ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id=label, score=0.95))
            ]
            d.results_length = 1
            d.bbox = BoundingBox3D(center=Pose(position=(x, y, z)), size=Vector3(0.5, 0.5, 0.9))
            dets.append(d)
        return Detection3DArray(
            detections_length=len(dets), header=Header(now, "world"), detections=dets
        )

    def _publish_loop(self) -> None:
        while not self._stop_event.is_set():
            self.detections_3d.publish(self.message())
            self._stop_event.wait(1.0 / self.config.rate_hz)
