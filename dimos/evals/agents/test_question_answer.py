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

import json

import numpy as np

from dimos.evals.agents.question_answer import QuestionAnswer
from dimos.evals.types import RunningEnvironment
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def test_image_ablation_retains_exact_calibration_and_all_other_text(tmp_path):
    cloud = PointCloud2.from_numpy(np.array([[1.125, -2.25, 3.5], [2.75, 1.0, 4.0]]), timestamp=1.0)
    with SqliteStore(path=str(tmp_path / "recording.db")) as store:
        stream = store.stream("cloud", PointCloud2)
        stream.append(cloud, ts=1.0)
        env = RunningEnvironment(mcp_url="", streams=(stream,), artifacts={})

        visual = QuestionAnswer()._observation_blocks(env)
        numeric = QuestionAnswer(include_images=False)._observation_blocks(env)

    assert len(visual) == len(numeric) + 1
    assert all(block["type"] == "text" for block in numeric)
    assert numeric == [block for block in visual if block["type"] == "text"]
    calibration = json.loads(numeric[-1]["text"])
    assert calibration["coordinate_min_m"] == [1.125, -2.25, 3.5]
    assert calibration["coordinate_max_m"] == [2.75, 1.0, 4.0]
    assert "png_base64" not in calibration
