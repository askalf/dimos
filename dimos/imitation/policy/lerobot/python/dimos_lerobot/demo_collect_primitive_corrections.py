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

"""Collect placement teachers from actual ACT-held states, preserving original demonstrations."""

import argparse
import json
from pathlib import Path
import time
from typing import Any

from dimos_lerobot.demo_object_primitives import rollout
from dimos_lerobot.runtime import LeRobotBackend
import numpy as np
import torch

from dimos.imitation.policy.lerobot.module import LeRobotPolicyConfig
from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest
from dimos.robot.galaxea.r1pro.object_packing_scene import sample_layout
from dimos.robot.galaxea.r1pro.object_primitive_task import ObjectPrimitiveTask
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, primitive_profile
from dimos.robot.galaxea.r1pro.primitive_scene import choose_placement, prepare_primitive_scene


def collect(policies: Path, output: Path, start_seed: int, layouts: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    for arm in ARMS:
        folder = output / f"place-{arm}"
        folder.mkdir()
        profile = primitive_profile("place", arm)
        manifest: dict[str, Any] = dict(
            version=1,
            profile=profile.name,
            arm=arm,
            primitive="place",
            fps=20,
            joints=list(profile.action.demonstration.joints),
            images=True,
            episodes=[],
            rejected=[],
            source_policies=str(policies.resolve()),
        )
        backend = LeRobotBackend(
            LeRobotPolicyConfig(artifact=str(policies / f"pick-{arm}"), device="cuda", task="pick")
        )
        backend.load(primitive_profile("pick", arm))
        for number, seed in enumerate(range(start_seed, start_seed + layouts)):
            layout = sample_layout(seed, occupied=1 + number % 3 if number % 2 else 0)
            selected = int(
                np.random.default_rng(seed + 91).choice(
                    [i for i, o in enumerate(layout.objects) if o.in_tray == bool(number % 2)]
                )
            )
            scene, layout = prepare_primitive_scene(output / f"scene-{seed}-{arm}.xml", layout, arm)
            stem = f"layout-{seed}-object-{selected}"
            row: dict[str, Any] = dict(
                seed=seed,
                selected=selected,
                shape=layout.objects[selected].shape,
                source="tray" if number % 2 else "table",
                arm=arm,
                primitive="place",
                layout=layout.to_dict(),
                scene=str(scene.resolve()),
                grasp_controller="ACT",
                success=False,
            )
            try:
                with ObjectPrimitiveTask(scene, layout, arm=arm, images=True) as task:
                    task.select(selected)
                    task.teacher_preposition(task.data.body(task.bottle_id).xpos.copy())
                    outcome = rollout(
                        task,
                        backend,
                        "pick",
                        None,
                        action_steps=20 if number % 2 else 30,
                        seconds=30,
                    )
                    row["act_pick"] = outcome
                    if not outcome["success"]:
                        raise RuntimeError(
                            "ACT did not supply a stable grasp; no teacher replacement"
                        )
                    target, region = choose_placement(
                        task, "table" if number % 2 or number % 4 == 0 else "tray", seed + selected
                    )
                    task.teacher_preposition(target)
                    task.state.target = target.copy()
                    initial = task.inventory()
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
                    cameras = {}
                    phases = []
                    for frame, (phase, action) in enumerate(task.teacher_place(target, region)):
                        obs = task.primitive_observation("place", render_images=frame % 2 == 0)
                        cameras.update({k: v for k, v in obs.items() if ".images." in k})
                        obs.update(cameras)
                        for key, value in {**obs, "action": action}.items():
                            frames.setdefault(key, []).append(value)
                        phases.append(phase)
                        task.primitive_step(action)
                        task.validate(initial)
                    row.update(
                        success=True,
                        frames=len(phases),
                        file=f"{stem}.npz",
                        initial_state=f"{stem}-initial.npz",
                        target=target.tolist(),
                        destination=region.name,
                        phases=phases,
                    )
                    arrays: dict[str, Any] = {k: np.stack(v) for k, v in frames.items()}
                    np.savez_compressed(folder / f"{stem}.npz", **arrays)
                    manifest["episodes"].append(row)
            except (RuntimeError, ValueError) as exc:
                row["error"] = str(exc)
                manifest["rejected"].append(row)
            save_manifest(folder / "manifest.json", manifest)
            save_manifest(
                output / "status.json",
                dict(
                    arm=arm,
                    seed=seed,
                    accepted=len(manifest["episodes"]),
                    rejected=len(manifest["rejected"]),
                    updated=time.time(),
                ),
            )
            print(
                json.dumps({k: v for k, v in row.items() if k not in ("phases", "layout")}),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=350000)
    parser.add_argument("--layouts", type=int, default=12)
    args = parser.parse_args()
    collect(args.policies, args.output, args.start_seed, args.layouts)


if __name__ == "__main__":
    main()
