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
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_FPS
from dimos.robot.galaxea.r1pro.object_bimanual_task import BimanualPrimitiveTask
from dimos.robot.galaxea.r1pro.object_packing_scene import (
    sample_layout,
)
from dimos.robot.galaxea.r1pro.object_primitive_task import ObjectPrimitiveTask
from dimos.robot.galaxea.r1pro.object_primitives import (
    ARMS,
    PRIMITIVES,
    Arm,
    Primitive,
    primitive_profile,
)
from dimos.robot.galaxea.r1pro.primitive_scene import (
    bilateral_layout,
    choose_placement,
    prepare_primitive_scene,
)


def collect(
    output: Path,
    *,
    start_seed: int,
    layouts: int,
    images: bool,
    choices: int = 1,
    interactive_context: bool = False,
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
            if interactive_context:
                contract["interactive_context"] = True
                contract["context_version"] = 2
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
        if interactive_context:
            original = bilateral_layout(original)
        candidates = [
            i for i, obj in enumerate(original.objects) if obj.in_tray == bool(number % 2)
        ]
        selected_indices = list(
            map(int, np.random.default_rng(seed + 91).permutation(candidates)[:choices])
        )
        for arm in ARMS:
            scene, layout = prepare_primitive_scene(
                output / f"scene-{seed}-{arm}.xml",
                original,
                "right" if interactive_context else arm,
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
                base_row: dict[str, Any] = dict(
                    seed=seed,
                    selected=selected,
                    shape=layout.objects[selected].shape,
                    source="tray" if layout.objects[selected].in_tray else "table",
                    arm=arm,
                    layout=layout.to_dict(),
                    scene=str(scene.resolve()),
                )
                try:
                    task_type = (
                        BimanualPrimitiveTask if interactive_context else ObjectPrimitiveTask
                    )
                    with task_type(scene, layout, arm=arm, images=images) as task:
                        if interactive_context:
                            assert isinstance(task, BimanualPrimitiveTask)
                            base_row["other_hand_object"] = None
                            # Half the layouts keep the other hand holding an object.
                            # Its setup grasp is teacher-only and never mislabeled ACT.
                            if number % 4 >= 2:
                                other_arm: Arm = "left" if arm == "right" else "right"
                                others = [
                                    i
                                    for i, o in enumerate(layout.objects)
                                    if i != selected and not o.in_tray
                                ]
                                if not others:
                                    raise RuntimeError("No supported object for the other hand")
                                other = min(
                                    others,
                                    key=lambda i: abs(
                                        layout.objects[i].position[1]
                                        - (0.4 if other_arm == "left" else -0.4)
                                    ),
                                )
                                task.switch_arm(other_arm)
                                task.select(other)
                                task.teacher_preposition(task.data.body(task.bottle_id).xpos.copy())
                                protected = task.inventory()
                                for setup_phase, action in task.teacher_pick():
                                    phase = f"other_hand/{setup_phase}"
                                    task.primitive_step(action)
                                    task.validate(protected)
                                base_row["other_hand_object"] = layout.objects[other].name
                                task.switch_arm(arm)
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
                                    ("worktable" if interactive_context else "table")
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
    parser.add_argument("--interactive-context", action="store_true")
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
                interactive_context=args.interactive_context,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
