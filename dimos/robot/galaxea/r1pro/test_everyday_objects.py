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

"""Everyday scenes preserve clearance and cups have actual hollow contact geometry."""

from dataclasses import replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.everyday_objects import sample_everyday_layout
from dimos.robot.galaxea.r1pro.navigation_cloud import environment_cloud
from dimos.robot.galaxea.r1pro.object_appearance import add_object_appearance
from dimos.robot.galaxea.r1pro.object_packing_scene import sample_layout


@pytest.mark.parametrize("seed", [380000, 380001, 380002])
def test_household_variants_retain_sampled_clearance_and_reproduce(seed):
    original = sample_layout(seed, count=5)
    layout = sample_everyday_layout(seed)
    assert layout == sample_everyday_layout(seed)
    assert {o.kind for o in layout.objects} == {
        "cup",
        "bottle",
        "glue_stick",
        "drink_carton",
        "toy_block",
    }
    for before, after in zip(original.objects, layout.objects, strict=True):
        assert after.position == before.position
        assert after.radius == pytest.approx(before.radius)
        assert after.half_size[2] == before.half_size[2]


def test_cup_has_open_cavity_and_retains_its_physical_mass():
    obj = next(o for o in sample_everyday_layout(380000).objects if o.kind == "cup")
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name=obj.name)
    ET.SubElement(body, "freejoint")
    assert add_object_appearance(body, obj, {"rgba": "0 0.5 1 1", "friction": "1 .02 .001"})
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([0, 0, 0.2]), np.array([0.0, 0.0, -1.0]), None, True, -1, gid
    )
    assert model.geom(int(gid[0])).name == obj.name + "_floor"
    assert 0.2 - distance == pytest.approx(-obj.half_size[2] + 0.004)
    assert model.body(obj.name).mass[0] == pytest.approx(obj.mass)


def test_legacy_objects_keep_their_original_geometry_path():
    body = ET.Element("body")
    obj = replace(sample_everyday_layout(380000).objects[0], kind=None)
    assert add_object_appearance(body, obj, {}) is False
    assert list(body) == []


def test_static_map_excludes_free_objects_without_requiring_legacy_bottles():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="box" size=".2 .2 .02"/>
      <body name="base_link" pos="2 0 1"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      <body name="cup" pos="3 0 1"><freejoint/><geom type="cylinder" size=".03 .06"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cloud = environment_cloud(model, data)
    assert np.max(np.abs(cloud[:, :2])) <= 0.201
    assert np.min(cloud[:, 2]) == pytest.approx(-0.02)
    assert np.max(cloud[:, 2]) == pytest.approx(0.02)
