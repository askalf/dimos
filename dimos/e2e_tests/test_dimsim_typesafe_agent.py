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
"""TypeSafe reactive agent drives the Go2 to a named object in DimSim.

Perception is not under test: the chair's world position is published by the
test as a `Detection3DArray`, the way a 3D detector or world belief would.
"""

from collections.abc import Callable, Iterator
import threading
import time
from typing import Any

from dimos_lcm.vision_msgs import (
    BoundingBox3D,
    Detection3D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
import pytest

from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.std_msgs.Header import Header
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray

CHAIR = (4.0, 2.0)


def _chair_detections() -> Detection3DArray:
    d = Detection3D()
    d.header = Header(time.time(), "world")
    d.results = [
        ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id="chair", score=0.95))
    ]
    d.results_length = 1
    d.bbox = BoundingBox3D(
        center=Pose(position=(CHAIR[0], CHAIR[1], 0.4)), size=Vector3(0.5, 0.5, 0.9)
    )
    return Detection3DArray(
        detections_length=1, header=Header(time.time(), "world"), detections=[d]
    )


@pytest.fixture
def chair_publisher() -> Iterator[None]:
    transport = make_transport("/detections_3d", Detection3DArray)
    transport.start()
    stop = threading.Event()

    def pump() -> None:
        while not stop.is_set():
            transport.publish(_chair_detections())
            stop.wait(0.5)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join(timeout=2)
    transport.stop()


@pytest.mark.self_hosted_large
def test_go_to_the_chair(
    lcm_spy: Any,
    start_blueprint: Callable[..., Any],
    wait_for_system_ready: Callable[..., None],
    human_input: Callable[[str], None],
    dim_sim: Any,
    chair_publisher: None,
) -> None:
    start_blueprint("run", "unitree-go2-typesafe", simulator="dimsim")
    wait_for_system_ready(timeout=1200.0)
    # Server physics (odom/lidar) starts only after the browser ships its snapshot.
    lcm_spy.wait_for_message_result(
        "/odom#geometry_msgs.PoseStamped", PoseStamped, lambda _m: True, "no odom", timeout=600
    )

    dim_sim.set_agent_position(1.0, 2.0)
    time.sleep(3.0)

    human_input("go to the chair")

    lcm_spy.wait_until_odom_position(CHAIR[0], CHAIR[1], threshold=1.2, timeout=120)
