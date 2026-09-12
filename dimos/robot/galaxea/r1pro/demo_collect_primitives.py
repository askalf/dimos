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

"""Resumeable collection of separate ACT pick and place demonstrations for both arms."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_FPS
from dimos.robot.galaxea.r1pro.object_packing_scene import (
    ObjectLayout,
    prepare_object_scene,
    sample_layout,
)
from dimos.robot.galaxea.r1pro.object_packing_state import object_extent
from dimos.robot.galaxea.r1pro.object_primitive_task import ObjectPrimitiveTask
from dimos.robot.galaxea.r1pro.object_primitives import (
    ARMS,
    PRIMITIVES,
    Arm,
    Primitive,
    primitive_profile,
)
from dimos.robot.galaxea.r1pro.placement_regions import (
    PlacementObstacle,
    PlacementRegion,
    placement_candidates,
)


def prepare_primitive_scene(
    output: Path, layout: ObjectLayout, arm: Arm
) -> tuple[Path, ObjectLayout]:
    """A symmetric physical bench overlay; only initial scene generation mirrors objects."""
    if arm == "left":
        layout = ObjectLayout(
            layout.seed,
            tuple(
                replace(o, position=(o.position[0], -o.position[1], o.position[2]), yaw=-o.yaw)
                for o in layout.objects
            ),
        )
    scene = prepare_object_scene(output, layout)
    tree = ET.parse(scene)
    root = tree.getroot()
    table = root.find('.//body[@name="task_table"]')
    tray = root.find('.//body[@name="task_bin"]')
    if table is None or tray is None or (top := table.find("geom")) is None:
        raise ValueError("Primitive bench requires a physical source table and tray")
    table.set("pos", "0.49 0 0.675")
    top.set("size", "0.31 0.65 0.025")
    if arm == "left":
        tray.set("pos", "0.34 0.04 0.701")
    tree.write(scene, encoding="unicode")
    scene.with_suffix(".objects.json").write_text(json.dumps(layout.to_dict()) + "\n")
    return scene, layout


def choose_placement(
    task: ObjectPrimitiveTask, destination: str, seed: int
) -> tuple[NDArray[np.float64], PlacementRegion]:
    """Geometric placement, then prepositioning; no extra ACT command is inferred."""
    if destination == "tray":
        tray = task.data.body("task_bin")
        region = PlacementRegion(
            "tray", tuple(tray.xpos + np.array([0, 0, 0.015])), (0.145, 0.105), ("bin_floor",)
        )
    elif destination == "table":
        sign = -1 if task.arm == "right" else 1
        top = next(
            task.model.geom(i).name
            for i in range(task.model.ngeom)
            if task.model.geom_bodyid[i] == task.model.body("task_table").id
        )
        region = PlacementRegion("table", (0.41, sign * 0.44, 0.7), (0.15, 0.16), (top,))
    else:
        raise ValueError("Collection destination must be table or tray")
    obstacles = []
    for i, obj in enumerate(task.layout.objects):
        if i == task.selected:
            continue
        body = task.data.body(obj.name)
        extent = object_extent(obj, body.xmat.reshape(3, 3))
        obstacles.append(
            PlacementObstacle(
                tuple(body.xpos[:2]),
                tuple(extent[:2]),
                float(body.xpos[2] - extent[2]),
                float(body.xpos[2] + extent[2]),
            )
        )
    if destination == "table":
        tray = task.data.body("task_bin")
        obstacles.append(
            PlacementObstacle(
                tuple(tray.xpos[:2]), (0.16, 0.235), float(tray.xpos[2]), float(tray.xpos[2] + 0.1)
            )
        )
    obj = task.layout.objects[task.selected]
    points = placement_candidates(
        region, radius=obj.radius, half_height=obj.half_size[2], obstacles=tuple(obstacles)
    )
    source = task.data.body(task.bottle_id).xpos
    points = tuple(p for p in points if np.linalg.norm(np.asarray(p[:2]) - source[:2]) >= 0.045)
    if not points:
        raise RuntimeError("No empty placement region with object and open-finger clearance")
    return np.asarray(points[int(np.random.default_rng(seed).integers(len(points)))]), region


def collect(
    output: Path, *, start_seed: int, layouts: int, images: bool, choices: int = 1
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifests: dict[tuple[Arm, Primitive], dict[str, Any]] = {}
    for arm in ARMS:
        for primitive in PRIMITIVES:
            folder = output / f"{primitive}-{arm}"
            folder.mkdir(exist_ok=True)
            profile = primitive_profile(primitive, arm)
            contract = dict(
                version=1,
                profile=profile.name,
                arm=arm,
                primitive=primitive,
                fps=R1PRO_PICK_PLACE_FPS,
                joints=list(profile.action.demonstration.joints),
                images=images,
                start_seed=start_seed,
                layouts=layouts,
                choices=choices,
            )
            path = folder / "manifest.json"
            manifest = (
                json.loads(path.read_text())
                if path.exists()
                else dict(**contract, episodes=[], rejected=[])
            )
            if any(manifest.get(k) != v for k, v in contract.items()):
                raise ValueError("Cannot resume with different collection settings")
            manifests[arm, primitive] = manifest
            save_manifest(path, manifest)
    for number, seed in enumerate(range(start_seed, start_seed + layouts)):
        original = sample_layout(seed, occupied=0 if number % 2 == 0 else 1 + number % 3)
        candidates = [
            i for i, obj in enumerate(original.objects) if obj.in_tray == bool(number % 2)
        ]
        selected_indices = list(
            map(int, np.random.default_rng(seed + 91).permutation(candidates)[:choices])
        )
        for arm in ARMS:
            scene, layout = prepare_primitive_scene(
                output / f"scene-{seed}-{arm}.xml", original, arm
            )
            for selected in selected_indices:
                finished = {
                    (row["seed"], row["selected"])
                    for row in manifests[arm, "place"]["episodes"]
                    + manifests[arm, "place"]["rejected"]
                }
                if (seed, selected) in finished:
                    continue
                phase = "initialize"
                primitive = "pick"
                started = time.monotonic()
                base_row = dict(
                    seed=seed,
                    selected=selected,
                    shape=layout.objects[selected].shape,
                    source="tray" if layout.objects[selected].in_tray else "table",
                    arm=arm,
                    layout=layout.to_dict(),
                    scene=str(scene.resolve()),
                )
                try:
                    with ObjectPrimitiveTask(scene, layout, arm=arm, images=images) as task:
                        task.select(selected)
                        for primitive in PRIMITIVES:
                            manifest = manifests[arm, primitive]
                            folder = output / f"{primitive}-{arm}"
                            stem = f"layout-{seed}-object-{selected}"
                            phase = "preposition"
                            if primitive == "pick":
                                target = task.data.body(task.bottle_id).xpos.copy()
                                region = None
                            else:
                                target, region = choose_placement(
                                    task,
                                    "table"
                                    if layout.objects[selected].in_tray or number % 4 == 0
                                    else "tray",
                                    seed + selected,
                                )
                            task.teacher_preposition(target)
                            if primitive == "place":
                                task.state.target = target.copy()
                            initial = task.inventory()
                            task.validate(initial)
                            np.savez_compressed(
                                folder / f"{stem}-initial.npz",
                                qpos=task.data.qpos,
                                qvel=task.data.qvel,
                                ctrl=task.data.ctrl,
                                act=task.data.act,
                                time=task.data.time,
                                initial_height=task.initial_height,
                            )
                            frames: dict[str, list[Any]] = {}
                            phases = []
                            cameras: dict[str, NDArray[Any]] = {}
                            teacher = (
                                task.teacher_pick()
                                if region is None
                                else task.teacher_place(target, region)
                            )
                            for frame, (phase, action) in enumerate(teacher):
                                obs = task.primitive_observation(
                                    primitive, render_images=frame % 2 == 0
                                )
                                cameras.update(
                                    {
                                        k: v
                                        for k, v in obs.items()
                                        if k.startswith("observation.images.")
                                    }
                                )
                                for key, value in {**cameras, **obs, "action": action}.items():
                                    frames.setdefault(key, []).append(value)
                                phases.append(phase)
                                task.primitive_step(action)
                                task.validate(initial)
                            arrays: dict[str, Any] = {k: np.stack(v) for k, v in frames.items()}
                            np.savez_compressed(
                                folder / f"{stem}.npz",
                                **arrays,
                                phase=np.asarray(phases),
                            )
                            row = dict(
                                **base_row,
                                file=f"{stem}.npz",
                                initial_state=f"{stem}-initial.npz",
                                primitive=primitive,
                                frames=len(phases),
                                success=True,
                                target=target.tolist(),
                                region=asdict(region) if region else None,
                                seconds=round(time.monotonic() - started, 2),
                            )
                            manifest["episodes"] = [
                                r
                                for r in manifest["episodes"]
                                if (r["seed"], r["selected"]) != (seed, selected)
                            ]
                            manifest["episodes"].append(row)
                            save_manifest(folder / "manifest.json", manifest)
                            print(
                                json.dumps({k: v for k, v in row.items() if k != "layout"}),
                                flush=True,
                            )
                except (RuntimeError, ValueError) as exc:
                    row = dict(
                        **base_row,
                        primitive=primitive,
                        phase=phase,
                        error=str(exc),
                        seconds=round(time.monotonic() - started, 2),
                    )
                    manifests[arm, primitive]["rejected"].append(row)
                    save_manifest(
                        output / f"{primitive}-{arm}" / "manifest.json", manifests[arm, primitive]
                    )
                    if primitive == "pick":
                        manifests[arm, "place"]["rejected"].append(
                            {**row, "primitive": "place", "phase": "pick prerequisite"}
                        )
                        save_manifest(
                            output / f"place-{arm}" / "manifest.json", manifests[arm, "place"]
                        )
                    print(json.dumps({k: v for k, v in row.items() if k != "layout"}), flush=True)
                report = {
                    f"{p}-{a}": dict(accepted=len(m["episodes"]), rejected=len(m["rejected"]))
                    for (a, p), m in manifests.items()
                }
                save_manifest(output / "status.json", dict(stage="collecting", seed=seed, **report))
    report = {
        f"{p}-{a}": dict(accepted=len(m["episodes"]), rejected=len(m["rejected"]))
        for (a, p), m in manifests.items()
    }
    save_manifest(output / "status.json", dict(stage="completed", **report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=320000)
    parser.add_argument("--layouts", type=int, default=8)
    parser.add_argument("--choices", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    if args.layouts < 1 or args.start_seed < 0:
        parser.error("Use positive layouts and a nonnegative seed")
    print(
        json.dumps(
            collect(
                args.output,
                start_seed=args.start_seed,
                layouts=args.layouts,
                choices=args.choices,
                images=not args.no_images,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
