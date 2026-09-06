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

"""Offline preparation of eight retained scenes; never imported by the runtime.

Run with --source /path/to/pimsim-starter --output data/sim2/scenes.
This reads the old directory format once, preserves source geometry and
textures, and writes native MJCF referring to shared local asset files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.sim2.scene_types import (
    SceneDescription,
    SceneEntity,
    SceneJoint,
    SceneRegion,
    SceneUpdate,
)

SCENES = {
    "kitchen": "model-kitchen-1",
    "libero-kitchen-1": "libero-kitchen-scene-1",
    "libero-kitchen-9": "libero-kitchen-scene-9",
    "robocasa-kitchen-1": "robocasa_scene1_style1",
    "robocasa-kitchen-7": "robocasa_scene7_style12",
    "ithor-kitchen": "molmospaces-ithor-15-layout-a90f4e06b03e",
    "procthor-house": "molmospaces-procthor-10k-85-layout-94c16f4e6f84",
    "hssd-home": "hssd-rigid-base-102344022",
}


def _pose(raw: dict[str, Any]) -> Pose:
    quat = np.asarray(raw["quaternion_xyzw"], dtype=float)
    return Pose(raw["position"], quat / np.linalg.norm(quat))


def _assets(spec: mujoco.MjSpec, source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for entries, directory in ((spec.meshes, spec.meshdir), (spec.textures, spec.texturedir)):
        for asset in entries:
            if not asset.file:
                continue
            path = (source / directory / asset.file).resolve()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            target = destination / (digest + path.suffix)
            if not target.exists():
                shutil.copyfile(path, target)
            asset.file = str(target.resolve())
    spec.meshdir = ""
    spec.texturedir = ""


def _metadata(raw: dict[str, Any], name: str, model: mujoco.MjModel) -> SceneDescription:
    entities = {}
    joints = {}
    regions = {}
    initial_joints = {}
    for entity in raw["entities"]:
        key, body = entity["id"], entity["body"]
        bid = model.body(body).id
        jid = int(model.body_jntadr[bid])
        movable = bool(
            model.body_mocapid[bid] >= 0
            or (jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE)
        )
        entities[key] = SceneEntity(
            body=body, label=key.replace("-", " "), kind=entity["class"], movable=movable
        )
        for joint in entity.get("joints", []):
            joint_key = f"{key}/{joint['id']}"
            joints[joint_key] = SceneJoint(
                joint=entity["joint_bodies"][joint["id"]],
                entity=key,
                closed=joint["closed"],
                opened=joint["open"],
            )
            initial_joints[joint_key] = joint["default"]
        for region in entity.get("regions", []):
            if region["kind"] not in ("support", "containment"):
                continue
            frame = region["frame"]
            region_body = body if frame == "asset" else entity["link_bodies"][frame]
            regions[f"{key}/{region['id']}"] = SceneRegion(
                body=region_body,
                kind=region["kind"],
                size=region["size"],
                pose=_pose(
                    {"position": region["center"], "quaternion_xyzw": region["quaternion_xyzw"]}
                ),
            )
    return SceneDescription(
        id=name,
        entities=entities,
        joints=joints,
        regions=regions,
        initial=SceneUpdate(joints=initial_joints),
        spawns={key: _pose(pose) for key, pose in raw.get("spawns", {}).items()},
        hidden_geom_groups=(),
        provenance={
            "source_scene": raw["id"],
            "source": raw.get("source"),
            "source_provenance": raw.get("provenance", {}),
        },
    )


def _add_object(
    spec: mujoco.MjSpec,
    root: Path,
    assets: Path,
    source_name: str,
    name: str,
    xyz: tuple[float, float, float],
) -> SceneEntity:
    folder = root / "objects" / source_name
    item = mujoco.MjSpec.from_file(str(folder / "object.xml"))
    item.option.integrator = spec.option.integrator
    _assets(item, folder, assets)
    for geom in item.geoms:
        if geom.group == 4:
            geom.group = 0
            geom.rgba[3] = 0
        else:
            geom.group = 1
    spec.attach(item, prefix=name + "/", frame=spec.worldbody.add_frame(pos=xyz))
    return SceneEntity(body=f"{name}/{source_name}", label=name, kind=name, movable=True)


def _populate_kitchen(spec: mujoco.MjSpec, root: Path, assets: Path) -> list[SceneEntity]:
    objects = (
        ("Apple_1", "apple", (-2.7, -1.0, 0.95)),
        ("Mug_1", "mug", (-1.8, 2.2, 0.965)),
        ("Plate_1", "plate", (1.9, -1.3, 0.77)),
        ("Salt_Shaker_1", "salt-shaker", (0.32, -0.37, 0.96)),
    )
    result = []
    for source_name, name, xyz in objects:
        result.append(_add_object(spec, root, assets, source_name, name, xyz))
    # A simple calibrated grasp object and open tray make the three examples
    # independent of imported grasp annotations or ambiguous container bounds.
    block = spec.worldbody.add_body(name="block", pos=(0.30, -0.16, 0.926))
    block.add_freejoint(name="block/free")
    block.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.022, 0.022, 0.025),
        rgba=(0.08, 0.6, 0.3, 1),
        mass=0.08,
        friction=(1, 0.01, 0.001),
    )
    tray = spec.worldbody.add_body(name="tray", pos=(0.61, -0.37, 0.91))
    for pos, size in (
        ((0, 0, 0), (0.11, 0.10, 0.01)),
        ((-0.105, 0, 0.055), (0.005, 0.10, 0.055)),
        ((0.105, 0, 0.055), (0.005, 0.10, 0.055)),
        ((0, -0.095, 0.055), (0.10, 0.005, 0.055)),
        ((0, 0.095, 0.055), (0.10, 0.005, 0.055)),
    ):
        tray.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=(0.25, 0.38, 0.55, 1)
        )
    result.extend(
        (
            SceneEntity(body="block", label="Green block", kind="block", movable=True),
            SceneEntity(body="tray", label="Blue tray", kind="tray"),
        )
    )
    return result


def _populate_robocasa(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    metadata: SceneDescription,
    root: Path,
    assets: Path,
    name: str,
) -> list[SceneEntity]:
    counter = "counter_right_main_group/top" if name.endswith("-1") else "counter_1_left_group/top"
    region = metadata.regions[counter]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body(region.body).id
    center = data.xpos[body] + data.xmat[body].reshape(3, 3) @ region.pose.position.to_tuple()
    return [
        _add_object(
            spec,
            root,
            assets,
            source_name,
            key,
            tuple(center + data.xmat[body].reshape(3, 3) @ np.array((offset, 0, height))),
        )
        for source_name, key, offset, height in (
            ("Apple_1", "apple", -0.18, 0.055),
            ("Salt_Shaker_1", "salt-shaker", 0, 0.065),
            ("Mug_1", "mug", 0.18, 0.07),
        )
    ]


def prepare(source: Path, output: Path, names: list[str]) -> None:
    report = []
    for name in names:
        started = time.monotonic()
        folder = source / SCENES[name]
        raw = json.loads((folder / "scene.json").read_text())
        spec = mujoco.MjSpec.from_file(str(folder / "scene.xml"))
        _assets(spec, folder, output / "_assets")
        config = raw.get("mujoco", {})
        remapping = {int(k): v for k, v in config.get("geom_group_remapping", {}).items()}
        hidden_groups = set(config.get("hidden_geom_groups", []))
        if name == "hssd-home":
            # This retained export has separate textured group-1 visuals but
            # omitted the mask for its opaque group-3 convex collision meshes.
            hidden_groups.add(3)
        for geom in spec.geoms:
            # Render masks must not change the source compiler's inertia groups.
            if remapping.get(geom.group, geom.group) in hidden_groups:
                geom.rgba[3] = 0
        added = _populate_kitchen(spec, source, output / "_assets") if name == "kitchen" else []
        for key in list(spec.keys):
            spec.delete(key)
        model = spec.compile()
        metadata = _metadata(raw, name, model)
        if name.startswith("robocasa-"):
            added = _populate_robocasa(spec, model, metadata, source, output / "_assets", name)
            model = spec.compile()
        metadata.entities.update({entity.kind: entity for entity in added})
        if name == "kitchen":
            metadata.regions["tray/interior"] = SceneRegion(
                body="tray", kind="containment", pose=Pose(0, 0, 0.065), size=(0.2, 0.18, 0.10)
            )
            metadata.regions["walkway-goal"] = SceneRegion(
                kind="navigation", pose=Pose(-0.8, -1.8, 0.8), size=(0.6, 0.6, 1.6)
            )
            metadata.initial.joints["cabinet-1/door-hinge"] = 0.5
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        xml = ET.fromstring(spec.to_xml())
        for element in xml.iter():
            if element.tag in ("mesh", "texture") and element.get("file"):
                element.set("file", os.path.relpath(element.attrib["file"], destination))
            if element.tag == "texture" and element.get("gridsize"):
                # MuJoCo 3.10 writes all 12 gridlayout slots even for a smaller
                # source atlas. Only rows*columns slots belong to that atlas.
                rows, columns = map(int, element.attrib["gridsize"].split())
                layout = element.attrib["gridlayout"]
                if any(value != "." for value in layout[rows * columns :]):
                    raise ValueError("texture atlas has populated cells outside its declared size")
                element.set("gridlayout", layout[: rows * columns])
        ET.indent(xml)
        ET.ElementTree(xml).write(destination / "scene.xml", encoding="unicode")
        (destination / "scene.json").write_text(metadata.model_dump_json(indent=2) + "\n")
        row = {
            "scene": name,
            "entities": len(metadata.entities),
            "movable": sum(e.movable for e in metadata.entities.values()),
            "joints": len(metadata.joints),
            "regions": len(metadata.regions),
            "spawns": sorted(metadata.spawns),
            "ngeoms": model.ngeom,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        report.append(row)
        print(json.dumps(row), flush=True)
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    args = parser.parse_args()
    prepare(args.source, args.output, args.scenes)
