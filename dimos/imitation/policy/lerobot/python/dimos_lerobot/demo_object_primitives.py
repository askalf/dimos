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

"""Physical chained ACT pick and place evaluation on fresh bench layouts."""

import argparse
import json
from pathlib import Path
from typing import Any

from dimos_lerobot.runtime import LeRobotBackend
import numpy as np
from numpy.typing import NDArray
import torch

from dimos.imitation.policy.lerobot.module import LeRobotPolicyConfig
from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest
from dimos.robot.galaxea.r1pro.demo_collect_primitives import (
    choose_placement,
    prepare_primitive_scene,
)
from dimos.robot.galaxea.r1pro.object_packing_scene import sample_layout
from dimos.robot.galaxea.r1pro.object_primitive_task import ObjectPrimitiveTask
from dimos.robot.galaxea.r1pro.object_primitives import (
    ARMS,
    PRIMITIVES,
    Primitive,
    primitive_profile,
)
from dimos.robot.galaxea.r1pro.placement_regions import PlacementRegion


def place_complete(task: ObjectPrimitiveTask, region: PlacementRegion) -> bool:
    row = task.geometry(task.selected)
    obj = task.layout.objects[task.selected]
    return bool(
        row["released"]
        and row["upright"]
        and row["settled"]
        and set(row["support_geoms"]) & set(region.support_geoms)
        and region.contains(tuple(row["position"]), obj.radius, obj.half_size[2])
        and task.data.site(f"{task.arm}_tcp").xpos[2]
        > row["position"][2] + obj.half_size[2] + 0.035
    )


def rollout(
    task: ObjectPrimitiveTask,
    backend: LeRobotBackend,
    primitive: Primitive,
    region: PlacementRegion | None,
    *,
    action_steps: int,
    seconds: float,
) -> dict[str, Any]:
    backend.reset()
    initial = task.inventory()
    actions: NDArray[np.float32] = np.empty((0, 8), dtype=np.float32)
    stable = 0
    frames = 0
    error = None
    try:
        for _ in range(round(seconds * 20)):
            frames += 1
            if len(actions) == 0:
                actions = backend.predict(task.primitive_observation(primitive), primitive)[
                    :action_steps
                ]
            task.primitive_step(actions[0])
            actions = actions[1:]
            task.validate(initial)
            complete = task.state.holding() if region is None else place_complete(task, region)
            stable = stable + 1 if complete else 0
            if stable >= 3:
                break
        if stable < 3:
            raise RuntimeError(f"{primitive} timed out without physical success")
        hold = task.data.ctrl[task.aids][task.active].copy()
        for _ in range(100 if primitive == "pick" else 20):
            task.primitive_step(hold)
            task.validate(initial)
            if not (task.state.holding() if region is None else place_complete(task, region)):
                raise RuntimeError("Physical success lost after stopping ACT")
    except (RuntimeError, ValueError) as exc:
        error = str(exc)
    return dict(
        success=error is None, error=error, frames=frames, geometry=task.geometry(task.selected)
    )


def evaluate(
    artifacts: Path, output: Path, *, start_seed: int, layouts: int, action_steps: int = 20
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    backends = {}
    for arm in ARMS:
        for primitive in PRIMITIVES:
            backend = LeRobotBackend(
                LeRobotPolicyConfig(
                    artifact=str(artifacts / f"{primitive}-{arm}"), device="cuda", task=primitive
                )
            )
            info = backend.load(primitive_profile(primitive, arm))
            if not 1 <= action_steps <= info.chunk_length:
                raise ValueError("Action steps must fit each trained chunk")
            backends[arm, primitive] = backend
    results = []
    for number, seed in enumerate(range(start_seed, start_seed + layouts)):
        original = sample_layout(seed, occupied=0 if number % 2 == 0 else 1 + number % 3)
        candidates = [
            i for i, obj in enumerate(original.objects) if obj.in_tray == bool(number % 2)
        ]
        selected = int(np.random.default_rng(seed + 71).choice(candidates))
        for arm in ARMS:
            scene, layout = prepare_primitive_scene(
                output / f"scene-{seed}-{arm}.xml", original, arm
            )
            row: dict[str, Any] = dict(
                seed=seed,
                arm=arm,
                selected=selected,
                shape=layout.objects[selected].shape,
                source="tray" if number % 2 else "table",
                success=False,
            )
            try:
                with ObjectPrimitiveTask(scene, layout, arm=arm) as task:
                    task.select(selected)
                    task.teacher_preposition(task.data.body(task.bottle_id).xpos.copy())
                    row["pick"] = rollout(
                        task,
                        backends[arm, "pick"],
                        "pick",
                        None,
                        action_steps=action_steps,
                        seconds=30,
                    )
                    if not row["pick"]["success"]:
                        raise RuntimeError("ACT pick failed; place was not attempted")
                    target, region = choose_placement(
                        task, "table" if number % 2 or number % 4 == 0 else "tray", seed + selected
                    )
                    task.teacher_preposition(target)
                    task.state.target = target.copy()
                    row["destination"] = region.name
                    row["place"] = rollout(
                        task,
                        backends[arm, "place"],
                        "place",
                        region,
                        action_steps=action_steps,
                        seconds=30,
                    )
                    row["success"] = row["place"]["success"]
                    np.savez_compressed(
                        output / f"final-{seed}-{arm}.npz",
                        qpos=task.data.qpos,
                        qvel=task.data.qvel,
                        ctrl=task.data.ctrl,
                    )
            except (RuntimeError, ValueError) as exc:
                row["error"] = str(exc)
            results.append(row)
            report = dict(
                artifacts=str(artifacts),
                start_seed=start_seed,
                action_steps=action_steps,
                total=len(results),
                successes=sum(r["success"] for r in results),
                classical_grasp_fallbacks=0,
                results=results,
            )
            save_manifest(output / "result.json", report)
            print(json.dumps(row), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=330000)
    parser.add_argument("--layouts", type=int, default=4)
    parser.add_argument("--action-steps", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.artifacts,
                args.output,
                start_seed=args.start_seed,
                layouts=args.layouts,
                action_steps=args.action_steps,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
