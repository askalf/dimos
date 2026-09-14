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

"""Open-scene supports, randomization and finite navigation geometry."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.everyday_objects import sample_everyday_layout
from dimos.robot.galaxea.r1pro.navigation_cloud import environment_cloud
from dimos.robot.galaxea.r1pro.open_space_scene import OPEN_PLATFORMS, prepare_open_space_scene
from dimos.robot.galaxea.r1pro.primitive_scene import bilateral_layout


@pytest.fixture
def make_scene(tmp_path, mocker):
    # Replace only the expensive external CAD conversion. Platform generation,
    # collision geometry, placement and map sampling use their real code.
    def bench(output, layout, arm, *, scene_package):
        output.parent.mkdir(parents=True)
        root = ET.fromstring(
            "<mujoco><asset/><worldbody>"
            '<geom name="floor" type="plane" size="5 5 .1"/>'
            '<body name="base_link"/>'
            '<body name="task_table" pos=".49 0 .675">'
            '<geom type="box" size=".31 .65 .025"/></body>'
            '<body name="task_bin" pos=".34 -.04 .701">'
            '<geom type="box" size=".155 .155 .0075"/></body>'
            "</worldbody></mujoco>"
        )
        world = root.find("worldbody")
        for obj in layout.objects:
            body = ET.SubElement(world, "body", name=obj.name, pos=" ".join(map(str, obj.position)))
            ET.SubElement(body, "freejoint")
            ET.SubElement(
                body, "geom", type="box", mass=".1", size=" ".join(map(str, obj.half_size))
            )
        ET.ElementTree(root).write(output, encoding="unicode")
        return output, layout

    mocker.patch(
        "dimos.robot.galaxea.r1pro.open_space_scene.prepare_primitive_scene", side_effect=bench
    )

    def generate(seed, name):
        return prepare_open_space_scene(
            tmp_path / name / "scene.xml", bilateral_layout(sample_everyday_layout(seed))
        )

    return generate


def test_named_platforms_have_distinct_physical_heights_and_supported_objects(make_scene):
    path, layout, regions = make_scene(5000, "scene")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert set(regions) == {platform.name for platform in OPEN_PLATFORMS}
    assert sorted(region.center[2] for region in regions.values()) == pytest.approx(
        [0.60, 0.70, 0.80, 0.85, 0.90]
    )
    assigned = []
    for obj in layout.objects:
        supports = [
            region
            for region in regions.values()
            if region.contains(obj.position, obj.radius, obj.half_size[2])
        ]
        assert len(supports) == 1
        assigned.append(supports[0].name)
        geom = model.geom(supports[0].support_geoms[0])
        assert supports[0].center[2] == pytest.approx(data.geom_xpos[geom.id, 2] + geom.size[2])
    assert len(set(assigned)) == 5
    for i, first in enumerate(OPEN_PLATFORMS):
        for second in OPEN_PLATFORMS[i + 1 :]:
            assert np.linalg.norm(np.subtract(first.xy, second.xy)) > 3


def test_spawn_is_reproducible_and_changes_with_seed(make_scene):
    _, first, _ = make_scene(5000, "first")
    _, repeated, _ = make_scene(5000, "repeat")
    _, changed, _ = make_scene(5001, "changed")

    assert first == repeated
    assert [obj.position for obj in first.objects] != [obj.position for obj in changed.objects]


def test_open_floor_builds_a_bounded_navigation_cloud(make_scene):
    path, _, _ = make_scene(5000, "scene")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cloud = environment_cloud(model, data, spacing=0.1)

    assert np.isfinite(cloud).all()
    assert np.min(cloud[:, 0]) == pytest.approx(-6)
    assert np.max(cloud[:, 0]) == pytest.approx(6)
    assert np.min(cloud[:, 1]) == pytest.approx(-6)
    assert np.max(cloud[:, 1]) == pytest.approx(6)
    assert all(
        Path(texture.get("file")).is_file()
        for texture in ET.parse(path).findall(".//texture[@file]")
    )
