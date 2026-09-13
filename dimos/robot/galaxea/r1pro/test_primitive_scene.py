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

"""Placement targets rest tilted objects above the support rather than through it."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout, PackingObject
from dimos.robot.galaxea.r1pro.placement_regions import PlacementRegion
from dimos.robot.galaxea.r1pro.primitive_scene import placement_options


def test_tilted_box_uses_its_vertical_extent_for_supported_placement():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="base_link"/>
      <body name="task_bin" pos="10 10 .7"><geom type="box" size=".1 .1 .02"/></body>
      <body name="item" pos="0 0 1" euler="10 0 0"><geom type="box" size=".04 .02 .08"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    obj = PackingObject("item", "box", (0.04, 0.02, 0.08), 0.1, (1, 0, 0, 1), (0, 0, 1), 0)
    context = SimpleNamespace(
        model=model,
        data=data,
        layout=ObjectLayout(0, (obj,)),
        selected=0,
        bottle_id=model.body("item").id,
        arm="right",
    )
    points, _ = placement_options(
        context, PlacementRegion("shelf", (0.5, 0, 0.7), (0.3, 0.3), ("top",)), None
    )
    vertical_extent = 0.08 * np.cos(np.deg2rad(10)) + 0.02 * np.sin(np.deg2rad(10))
    assert points
    assert points[0][2] - vertical_extent == pytest.approx(0.7)
