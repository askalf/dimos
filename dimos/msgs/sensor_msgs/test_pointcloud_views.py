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

"""Geometry and transport checks for native point-cloud projections."""

import base64
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from dimos.evals.agents.question_answer import _observation_blocks
from dimos.memory.type.observation import Observation
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def test_projection_preserves_native_axes_and_maximum_at_overlapping_pixels():
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [2.0, 1.0, 0.0]])
    cloud = PointCloud2.from_numpy(points, frame_id="optical")

    encoded = cloud.agent_encode()
    (view,) = encoded["views"]
    png = base64.b64decode(view["png_base64"], validate=True)
    with Image.open(BytesIO(png)) as image:
        rgb = image.convert("RGB")
        # The lower-left XY pixel contains both z=1 and z=3; maximum wins.
        assert rgb.getpixel((48, 30 + 255)) == (220, 55, 40)
        # The remote XY point at x=2, y=1 has z=0 and remains blue.
        assert rgb.getpixel((48 + 255, 30 + 255 - 128)) == (35, 55, 180)
        assert rgb.getpixel((48 + 100, 30 + 100)) == (255, 255, 255)
    assert encoded["frame_id"] == "optical"
    assert view["planes"] == ["xy", "xz", "yz"]
    assert view["pixel_m"] == pytest.approx([2 / 255, 3 / 255, 3 / 255])
    assert view["coordinate_min_m"] == [0.0, 0.0, 0.0]
    assert view["coordinate_max_m"] == [2.0, 1.0, 3.0]
    assert PointCloud2.from_numpy(points[::-1], frame_id="optical").agent_encode() == encoded


def test_static_consumer_delivers_png_without_base64_in_text():
    cloud = PointCloud2.from_numpy(np.array([[1.0, -2.0, 3.0], [2.0, 1.0, 4.0]]))
    encoded = cloud.agent_encode()

    blocks = _observation_blocks(Observation(_data=cloud), "[t=0.0s]")

    metadata = json.loads(blocks[0]["text"].split(" ", 1)[1])
    assert metadata == {key: value for key, value in encoded.items() if key != "views"}
    assert all("png_base64" not in block.get("text", "") for block in blocks)
    image_blocks = [block for block in blocks if block["type"] == "image_url"]
    assert len(image_blocks) == 1
    url = image_blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        assert image.format == "PNG"
        assert image.width > image.height


def test_static_consumer_keeps_legacy_dictionary_input_unchanged():
    class NumericMessage:
        def agent_encode(self):
            return {"frame_id": "native", "num_points": 2, "window_m": {"x": [1.0, 2.0]}}

    message = NumericMessage()

    blocks = _observation_blocks(Observation(_data=message), "[t=1.2s]")

    assert blocks == [{"type": "text", "text": f"[t=1.2s] {json.dumps(message.agent_encode())}"}]
