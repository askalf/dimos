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

"""Record corrective teachers from actual ACT approaches, including other-hand holds."""

import argparse
from collections.abc import Iterator
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from dimos_lerobot.runtime import LeRobotBackend
import numpy as np
from numpy.typing import NDArray
import torch

from dimos.imitation.policy.lerobot.module import LeRobotPolicyConfig
from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest
from dimos.robot.galaxea.r1pro.object_bimanual_task import BimanualPrimitiveTask
from dimos.robot.galaxea.r1pro.object_packing_scene import sample_layout
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


def approach(
    task: BimanualPrimitiveTask, backend: LeRobotBackend, action_steps: int = 30
) -> dict[str, Any]:
    """Run ACT until it is about to close; labels begin only after teacher takeover."""
    backend.reset()
    initial = task.inventory()
    actions: NDArray[np.float32] = np.empty((0, 8), dtype=np.float32)
    for frame in range(600):
        if len(actions) == 0:
            actions = backend.predict(task.primitive_observation("pick"), "pick")[:action_steps]
        row = task.geometry(task.selected)
        obj = task.layout.objects[task.selected]
        offset = task.data.site(f"{task.arm}_tcp").xpos - np.asarray(row["position"])
        grasp_height = min(0.03, obj.half_size[2] * 0.45)
        opening = float(task.data.qpos[task.qids[task.active[-1]]])
        if (
            actions[0, -1] < 0.04
            and opening >= 0.04
            and row["upright"]
            and row["released"]
            and row["support_geoms"]
            and row["settled"]
            and np.linalg.norm(offset[:2]) <= 0.05
            and abs(offset[2] - grasp_height) <= 0.04
        ):
            return dict(frames=frame, tcp_offset=offset.tolist(), gripper=opening)
        if not row["upright"] or not row["support_geoms"]:
            raise RuntimeError("ACT moved the source before an eligible open-hand correction")
        task.primitive_step(actions[0])
        actions = actions[1:]
        task.validate(initial)
    raise RuntimeError("ACT never reached an eligible open-hand correction state")


def record(
    task: BimanualPrimitiveTask,
    primitive: Primitive,
    teacher: Iterator[tuple[str, NDArray[np.float32]]],
    folder: Path,
    stem: str,
) -> dict[str, Any]:
    """Keep labels strictly teacher-generated and retain the complete physical initial state."""
    initial = task.inventory()
    np.savez_compressed(
        folder / f"{stem}-initial.npz",
        qpos=task.data.qpos,
        qvel=task.data.qvel,
        ctrl=task.data.ctrl,
        act=task.data.act,
        time=task.data.time,
        initial_height=task.state.initial_height,
    )
    frames: dict[str, list[Any]] = {}
    phases = []
    cameras = {}
    for frame, (phase, action) in enumerate(teacher):
        obs = task.primitive_observation(primitive, render_images=frame % 2 == 0)
        cameras.update({k: v for k, v in obs.items() if ".images." in k})
        obs.update(cameras)
        for key, value in {**obs, "action": action}.items():
            frames.setdefault(key, []).append(value)
        phases.append(phase)
        task.primitive_step(action)
        task.validate(initial)
    # Retain the same post-stop hold verification used by learned evaluation.
    if primitive == "pick":
        hold = task.data.ctrl[task.aids][task.active].copy()
        for _ in range(100):
            task.primitive_step(hold)
            task.validate(initial)
            if not task.state.holding():
                raise RuntimeError("Corrected grasp did not retain a stable five-second hold")
    arrays: dict[str, Any] = {k: np.stack(v) for k, v in frames.items()}
    np.savez_compressed(folder / f"{stem}.npz", **arrays, phase=np.asarray(phases))
    return dict(
        frames=len(phases), file=f"{stem}.npz", initial_state=f"{stem}-initial.npz", success=True
    )


def collect(
    policies: Path, output: Path, start_seed: int, layouts: int, action_steps: int = 30
) -> None:
    contract = dict(
        source=str(policies.resolve()),
        pick_sha256={
            arm: hashlib.sha256(
                (policies / f"pick-{arm}/model.safetensors").read_bytes()
            ).hexdigest()
            for arm in ARMS
        },
        start_seed=start_seed,
        layouts=layouts,
        action_steps=action_steps,
        interactive_context=True,
        approach_corrections=True,
        table_preferred_reach=0.48,
    )
    settings = output / "settings.json"
    if settings.exists() and json.loads(settings.read_text()) != contract:
        raise ValueError("Cannot resume corrections with different settings or weights")
    if not settings.exists() and output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to reuse a correction collection without its settings")
    output.mkdir(parents=True, exist_ok=True)
    save_manifest(settings, contract)
    torch.set_num_threads(4)
    totals = {}
    for arm in ARMS:
        manifests: dict[Primitive, dict[str, Any]] = {}
        for primitive in PRIMITIVES:
            folder = output / f"{primitive}-{arm}"
            folder.mkdir(exist_ok=True)
            profile = primitive_profile(primitive, arm)
            path = folder / "manifest.json"
            manifests[primitive] = (
                json.loads(path.read_text())
                if path.exists()
                else dict(
                    version=1,
                    profile=profile.name,
                    arm=arm,
                    primitive=primitive,
                    fps=20,
                    joints=list(profile.action.demonstration.joints),
                    images=True,
                    episodes=[],
                    rejected=[],
                    **contract,
                )
            )
            save_manifest(path, manifests[primitive])
        completed = {
            row["seed"] for row in manifests["place"]["episodes"] + manifests["place"]["rejected"]
        }
        backend = LeRobotBackend(
            LeRobotPolicyConfig(artifact=str(policies / f"pick-{arm}"), device="cuda", task="pick")
        )
        backend.load(primitive_profile("pick", arm))
        for number, seed in enumerate(range(start_seed, start_seed + layouts)):
            if seed in completed:
                continue
            original = bilateral_layout(
                sample_layout(seed, occupied=1 + number % 3 if number % 2 else 0)
            )
            candidates = [
                i for i, o in enumerate(original.objects) if o.in_tray == bool(number % 2)
            ]
            selected = int(np.random.default_rng(seed + 91).choice(candidates))
            scene, layout = prepare_primitive_scene(
                output / f"scene-{seed}-{arm}.xml", original, "right"
            )
            row: dict[str, Any] = dict(
                seed=seed,
                arm=arm,
                selected=selected,
                shape=layout.objects[selected].shape,
                source="tray" if number % 2 else "table",
                layout=layout.to_dict(),
                scene=str(scene.resolve()),
                other_hand_object=None,
                approach_controller="ACT",
                grasp_controller="SDK teacher",
                success=False,
            )
            primitive = "pick"
            try:
                with BimanualPrimitiveTask(scene, layout, arm=arm, images=True) as task:
                    if number % 4 >= 2:
                        other_arm: Arm = "left" if arm == "right" else "right"
                        other = min(
                            (
                                i
                                for i, o in enumerate(layout.objects)
                                if i != selected and not o.in_tray
                            ),
                            key=lambda i: abs(
                                layout.objects[i].position[1]
                                - (0.4 if other_arm == "left" else -0.4)
                            ),
                        )
                        task.switch_arm(other_arm)
                        task.select(other)
                        task.teacher_preposition(task.data.body(task.bottle_id).xpos.copy())
                        protected = task.inventory()
                        for _, action in task.teacher_pick():
                            task.primitive_step(action)
                            task.validate(protected)
                        row["other_hand_object"] = layout.objects[other].name
                        task.switch_arm(arm)
                    task.select(selected)
                    task.teacher_preposition(
                        task.data.body(task.bottle_id).xpos.copy(),
                        preferred_reach=0.48 if not layout.objects[selected].in_tray else None,
                    )
                    row["act_approach"] = approach(task, backend, action_steps)
                    stem = f"layout-{seed}-object-{selected}"
                    pick = record(
                        task,
                        "pick",
                        task.teacher_pick(from_approach=True),
                        output / f"pick-{arm}",
                        stem,
                    )
                    manifests["pick"]["episodes"] = [
                        r for r in manifests["pick"]["episodes"] if r["seed"] != seed
                    ]
                    manifests["pick"]["episodes"].append({**row, **pick, "primitive": "pick"})
                    save_manifest(output / f"pick-{arm}/manifest.json", manifests["pick"])
                    primitive = "place"
                    target, region = choose_placement(
                        task,
                        "worktable" if number % 2 or number % 4 == 0 else "tray",
                        seed + selected,
                    )
                    task.teacher_preposition(target)
                    task.state.target = target.copy()
                    place = record(
                        task,
                        "place",
                        task.teacher_place(target, region),
                        output / f"place-{arm}",
                        stem,
                    )
                    manifests["place"]["episodes"].append(
                        {
                            **row,
                            **place,
                            "primitive": "place",
                            "target": target.tolist(),
                            "region": asdict(region),
                        }
                    )
            except (RuntimeError, ValueError) as exc:
                rejected = {**row, "primitive": primitive, "error": str(exc)}
                manifests[primitive]["rejected"].append(rejected)
                if primitive == "pick":
                    manifests["place"]["rejected"].append({**rejected, "primitive": "place"})
            for primitive in PRIMITIVES:
                manifest = manifests[primitive]
                save_manifest(output / f"{primitive}-{arm}/manifest.json", manifest)
                totals[f"{primitive}-{arm}"] = dict(
                    accepted=len(manifest["episodes"]),
                    rejected=len(manifest["rejected"]),
                    other_hand_held=sum(bool(r["other_hand_object"]) for r in manifest["episodes"]),
                )
            save_manifest(
                output / "status.json", dict(state="running", arm=arm, seed=seed, totals=totals)
            )
            print(json.dumps(dict(arm=arm, seed=seed, totals=totals)), flush=True)
    save_manifest(output / "status.json", dict(state="completed", totals=totals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=370000)
    parser.add_argument("--layouts", type=int, default=32)
    parser.add_argument("--action-steps", type=int, default=30)
    args = parser.parse_args()
    if args.layouts < 1 or args.start_seed < 0:
        parser.error("Use positive layouts and a nonnegative start seed")
    if not 1 <= args.action_steps <= 30:
        parser.error("Action steps must be between 1 and 30")
    collect(args.policies, args.output, args.start_seed, args.layouts, args.action_steps)


if __name__ == "__main__":
    main()
