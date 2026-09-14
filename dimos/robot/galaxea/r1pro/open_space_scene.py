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

"""Five widely separated physical platforms and randomized household props."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout
from dimos.robot.galaxea.r1pro.placement_regions import (
    PlacementObstacle,
    PlacementRegion,
    placement_candidates,
)
from dimos.robot.galaxea.r1pro.primitive_scene import prepare_primitive_scene


@dataclass(frozen=True)
class OpenPlatform:
    name: str
    xy: tuple[float, float]
    height: float
    color: tuple[float, float, float]


OPEN_PLATFORMS = (
    OpenPlatform("worktable", (0.49, 0.0), 0.70, (0.20, 0.50, 0.70)),
    OpenPlatform("low_bench", (-3.0, 2.5), 0.60, (0.30, 0.65, 0.45)),
    OpenPlatform("display_table", (0.0, 3.5), 0.80, (0.65, 0.45, 0.75)),
    OpenPlatform("tall_table", (3.5, 2.5), 0.90, (0.85, 0.55, 0.25)),
    OpenPlatform("high_counter", (3.5, -2.5), 0.85, (0.75, 0.35, 0.40)),
)


def _label(asset: ET.Element, body: ET.Element, platform: OpenPlatform, output: Path) -> None:
    """Bake readable labels into non-colliding tabletop decals."""
    path = output.parent / f"{platform.name}-label.png"
    with Image.new("RGB", (768, 256), (235, 240, 245)) as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=58)
        draw.text(
            (384, 90),
            platform.name.replace("_", " ").upper(),
            font=font,
            anchor="mm",
            fill=(30, 40, 55),
        )
        draw.text(
            (384, 178), f"{platform.height * 100:.0f} cm", font=font, anchor="mm", fill=(45, 60, 80)
        )
        image.save(path)
    texture = f"{platform.name}_label"
    ET.SubElement(asset, "texture", name=texture, type="2d", file=str(path.resolve()))
    ET.SubElement(asset, "material", name=texture, texture=texture, texuniform="false")
    ET.SubElement(
        body,
        "geom",
        name=texture,
        type="box",
        pos="0.0 0.35 0.0253",
        size="0.28 0.09 0.0002",
        material=texture,
        contype="0",
        conaffinity="0",
        group="2",
        mass="0",
    )


def prepare_open_space_scene(
    output: Path, layout: ObjectLayout
) -> tuple[Path, ObjectLayout, dict[str, PlacementRegion]]:
    """Generate initial poses only; subsequent motion uses the normal physical skills."""
    scene, layout = prepare_primitive_scene(output, layout, "right", scene_package=None)
    tree = ET.parse(scene)
    world, asset = tree.find("worldbody"), tree.find("asset")
    assert world is not None and asset is not None
    floor = world.find('geom[@name="floor"]')
    assert floor is not None
    floor.set("type", "box")
    floor.set("size", "6 6 0.04")
    floor.set("pos", "0 0 -0.04")
    floor.set("rgba", "1 1 1 1")
    ET.SubElement(
        asset,
        "texture",
        name="open_floor",
        type="2d",
        builtin="checker",
        rgb1="0.36 0.40 0.44",
        rgb2="0.42 0.46 0.50",
        width="512",
        height="512",
    )
    ET.SubElement(asset, "material", name="open_floor", texture="open_floor", texrepeat="12 12")
    floor.set("material", "open_floor")
    for light_x in (-3, 3):
        ET.SubElement(world, "light", pos=f"{light_x} 1 5", diffuse="0.5 0.5 0.5")
    for platform in OPEN_PLATFORMS:
        if platform.name == "worktable":
            body = world.find('body[@name="task_table"]')
            assert body is not None
            top = body.find("geom")
            assert top is not None
            top.set("name", "worktable_top")
        else:
            body = ET.SubElement(
                world,
                "body",
                name=f"platform_{platform.name}",
                pos=f"{platform.xy[0]} {platform.xy[1]} {platform.height - 0.025}",
            )
            top = ET.SubElement(
                body,
                "geom",
                name=f"{platform.name}_top",
                type="box",
                size="0.35 0.50 0.025",
                conaffinity="3",
            )
            for x in (-0.27, 0.27):
                for y in (-0.42, 0.42):
                    half = (platform.height - 0.05) / 2
                    ET.SubElement(
                        body,
                        "geom",
                        type="box",
                        name=f"{platform.name}_leg_{x}_{y}",
                        pos=f"{x} {y} {-half - 0.025}",
                        size=f"0.035 0.035 {half}",
                        rgba="0.18 0.21 0.25 1",
                        conaffinity="3",
                    )
        top.set("rgba", " ".join(map(str, (*platform.color, 1))))
        _label(asset, body, platform, output)
    tree.write(scene, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    regions = {}
    for platform in OPEN_PLATFORMS:
        geom = model.geom(f"{platform.name}_top")
        center = data.geom_xpos[geom.id]
        height = float(center[2] + geom.size[2])
        # Reachable strip near the west edge, with space for open fingers.
        regions[platform.name] = PlacementRegion(
            platform.name,
            (0.46, 0.0, height)
            if platform.name == "worktable"
            else (float(center[0] - 0.20), float(center[1]), height),
            (0.18, 0.57) if platform.name == "worktable" else (0.10, 0.36),
            (geom.name,),
        )
    rng = np.random.default_rng(layout.seed + 991)
    stations = ["worktable", *rng.permutation([p.name for p in OPEN_PLATFORMS[1:]])]
    stations = list(rng.permutation(stations[: len(layout.objects)]))
    tray = data.body("task_bin").xpos
    placed = []
    for obj, station in zip(layout.objects, stations, strict=True):
        obstacles = (
            (
                PlacementObstacle(
                    tuple(tray[:2]), (0.16, 0.235), float(tray[2]), float(tray[2] + 0.1)
                ),
            )
            if station == "worktable"
            else ()
        )
        points = placement_candidates(
            regions[station], radius=obj.radius, half_height=obj.half_size[2], obstacles=obstacles
        )
        if not points:
            raise RuntimeError(f"No clear initial object footprint on {station}")
        point = (
            min(points, key=lambda p: float(np.linalg.norm(np.asarray(p[:2]) - obj.position[:2])))
            if station == "worktable"
            else points[int(rng.integers(len(points)))]
        )
        obj = replace(obj, position=(point[0], point[1], point[2] + 0.001), in_tray=False)
        placed.append(obj)
        body = tree.find(f'.//body[@name="{obj.name}"]')
        assert body is not None
        body.set("pos", " ".join(map(str, obj.position)))
    layout = ObjectLayout(layout.seed, tuple(placed))
    ET.indent(tree)
    tree.write(scene, encoding="unicode")
    scene.with_suffix(".objects.json").write_text(json.dumps(layout.to_dict()) + "\n")
    return scene, layout, regions
