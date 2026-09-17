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

"""Recognizable household props with explicit contact geometry and bounded dimensions."""

import math
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    from dimos.robot.galaxea.r1pro.object_packing_scene import PackingObject


def add_object_appearance(body: ET.Element, obj: "PackingObject", contact: dict[str, str]) -> bool:
    """Add household details; return true when this function supplies the physical body.

    The cup has a real hollow polygonal wall and floor. Other objects retain the
    existing box/bottle/cylinder collision bodies; colored bands have no mass or
    contacts and stay within the object's declared envelope.
    """
    if obj.kind is None:
        return False
    x, y, h = obj.half_size
    if obj.kind == "cup":
        if obj.shape != "cylinder":
            raise ValueError("A cup requires a cylindrical envelope")
        segments = 24
        thickness = 0.0015
        floor_half_height = 0.002
        # Seat the walls on the floor, rather than letting all 24 wall bottoms
        # also contact the support. Keep the open cavity and the same outer height.
        radial = x * math.cos(math.pi / segments) - thickness
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            ET.SubElement(
                body,
                "geom",
                name=f"{obj.name}_wall_{i}",
                type="box",
                pos=f"{radial * math.cos(angle)} {radial * math.sin(angle)} {floor_half_height}",
                euler=f"0 0 {angle}",
                size=f"{thickness} {x * math.sin(math.pi / segments)} {h - floor_half_height}",
                mass=str(obj.mass * 0.8 / segments),
                attrib=contact,
            )
        ET.SubElement(
            body,
            "geom",
            name=f"{obj.name}_floor",
            type="cylinder",
            pos=f"0 0 {-h + floor_half_height}",
            size=f"{x} {floor_half_height}",
            mass=str(obj.mass * 0.2),
            attrib=contact,
        )
        return True
    if obj.kind not in ("drink_carton", "toy_block", "glue_stick", "bottle"):
        raise ValueError(f"Unknown household object kind: {obj.kind}")
    visual = dict(contype="0", conaffinity="0", mass="0", group="2")
    if obj.shape == "box":
        # Inset raised print on each face, avoiding a collision-size mismatch.
        for axis, half, span in ((0, x, y), (1, y, x)):
            for sign in (-1, 1):
                position = [0.0, 0.0, -h * 0.1]
                position[axis] = sign * (half + 0.0001)
                size = [span * 0.8, span * 0.8, h * 0.45]
                size[axis] = 0.0001
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{obj.name}_label_{axis}_{sign}",
                    type="box",
                    pos=" ".join(map(str, position)),
                    size=" ".join(map(str, size)),
                    rgba="0.92 0.86 0.65 1" if obj.kind == "toy_block" else "0.95 0.95 0.90 1",
                    attrib=visual,
                )
        return False
    for name, z, radius, half_height, color in (
        ("label", -h * 0.2, x + 0.0002, h * 0.35, "0.95 0.95 0.90 1"),
        (
            "cap",
            h * 0.8,
            x * (0.52 if obj.shape == "bottle" else 1.0) + 0.0002,
            h * 0.19,
            "0.12 0.16 0.24 1",
        ),
    ):
        ET.SubElement(
            body,
            "geom",
            name=f"{obj.name}_{name}_visual",
            type="cylinder",
            pos=f"0 0 {z}",
            size=f"{radius} {half_height}",
            rgba=color,
            attrib=visual,
        )
    return False
