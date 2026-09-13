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

"""Fine-tune existing primitives on verified correction data with original-data rehearsal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, PRIMITIVES
from dimos.robot.galaxea.r1pro.primitive_training_job import TrainingStages


def merge_demonstrations(rehearsal: Path, corrections: Path, output: Path) -> None:
    original = json.loads((rehearsal / "manifest.json").read_text())
    fresh = json.loads((corrections / "manifest.json").read_text())
    for field in ("profile", "arm", "primitive", "fps", "joints", "images"):
        if original[field] != fresh[field]:
            raise ValueError(f"Rehearsal and correction contracts differ: {field}")
    if len(fresh["episodes"]) < 8:
        raise ValueError(
            "At least eight successful additional demonstrations per profile are required"
        )
    rows: list[dict[str, Any]] = []
    seen = set()
    for folder, manifest in ((rehearsal, original), (corrections, fresh)):
        for row in manifest["episodes"]:
            key = (row["seed"], row["selected"])
            if key in seen or not row["success"]:
                raise ValueError("Duplicate or failed primitive demonstration")
            seen.add(key)
            rows.append(
                {
                    **row,
                    "file": str((folder / row["file"]).resolve()),
                    "initial_state": str((folder / row["initial_state"]).resolve()),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    save_manifest(
        output / "manifest.json",
        {
            **original,
            "episodes": rows,
            "rehearsal_source": str(rehearsal.resolve()),
            "correction_source": str(corrections.resolve()),
            "rehearsal_episodes": len(original["episodes"]),
            "correction_episodes": len(fresh["episodes"]),
        },
    )


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[4]
    job = args.output.resolve()
    job.mkdir(parents=True, exist_ok=True)
    contract = dict(
        policies=str(args.policies.resolve()),
        hashes={
            f"{primitive}-{arm}": hashlib.sha256(
                (args.policies / f"{primitive}-{arm}/model.safetensors").read_bytes()
            ).hexdigest()
            for arm in ARMS
            for primitive in PRIMITIVES
        },
        rehearsal=str(args.rehearsal.resolve()),
        layouts=args.layouts,
        start_seed=args.start_seed,
        evaluation_seed=args.evaluation_seed,
        steps=args.steps,
        action_steps=args.action_steps,
    )
    if args.interactive_context:
        contract["interactive_context"] = True
    if args.approach_corrections:
        contract["approach_corrections"] = True
    train_all = args.interactive_context or args.approach_corrections
    settings = job / "settings.json"
    if settings.exists() and json.loads(settings.read_text()) != contract:
        raise ValueError("Resume settings or source policies differ")
    save_manifest(settings, contract)
    stage = TrainingStages(root, job)
    prepare = [*stage.learned, "-m", "dimos_lerobot.prepare_primitive_act"]
    try:
        collection = (
            [
                sys.executable,
                "-m",
                "dimos.robot.galaxea.r1pro.demo_collect_primitives",
                "--interactive-context",
            ]
            if args.interactive_context
            else [
                *stage.learned,
                "-m",
                "dimos_lerobot.demo_collect_primitive_approaches"
                if args.approach_corrections
                else "dimos_lerobot.demo_collect_primitive_corrections",
                "--policies",
                str(args.policies.resolve()),
                "--action-steps",
                str(args.action_steps),
            ]
        )
        stage(
            "collect",
            [
                *collection,
                "--output",
                str(job / "collection"),
                "--layouts",
                str(args.layouts),
                "--start-seed",
                str(args.start_seed),
            ],
        )
        for primitive, arm in (
            (p, a) for a in ARMS for p in (PRIMITIVES if train_all else ("place",))
        ):
            name = f"{primitive}-{arm}"
            if train_all:
                fresh = json.loads((job / "collection" / name / "manifest.json").read_text())
                held_examples = sum(bool(row.get("other_hand_object")) for row in fresh["episodes"])
                if held_examples < 4:
                    raise ValueError(
                        f"Need at least four other-hand-held demonstrations for {name}"
                    )
            merged = job / "merged" / name
            merge_demonstrations(args.rehearsal / name, job / "collection" / name, merged)
            dataset, initialization, training = (
                job / parent / name for parent in ("datasets", "initializations", "training")
            )
            stage(
                f"convert-{name}",
                [
                    *prepare,
                    "convert",
                    "--source",
                    str(merged),
                    "--output",
                    str(dataset),
                ],
            )
            stage(
                f"initialize-{name}",
                [
                    *prepare,
                    "initialize",
                    "--source",
                    str(args.policies.resolve() / name),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(initialization),
                    "--primitive",
                    primitive,
                    "--arm",
                    arm,
                ],
            )
            resume = training / "checkpoints/last/pretrained_model/train_config.json"
            source = (
                ["--resume=true", "--config_path=" + str(resume)]
                if resume.exists()
                else ["--policy.path=" + str(initialization)]
            )
            stage(
                f"train-{name}",
                [
                    *stage.learned,
                    "-m",
                    "lerobot.scripts.lerobot_train",
                    *source,
                    f"--dataset.repo_id=local/r1pro-{name}",
                    "--dataset.root=" + str(dataset),
                    "--dataset.eval_split=0.125",
                    "--policy.device=cuda",
                    "--policy.push_to_hub=false",
                    "--policy.optimizer_lr=.00003",
                    "--policy.optimizer_lr_backbone=.000006",
                    "--steps=" + str(args.steps),
                    "--batch_size=32",
                    "--num_workers=4",
                    "--env_eval_freq=0",
                    "--eval_steps=1000",
                    "--max_eval_samples=256",
                    "--log_freq=200",
                    "--save_freq=1000",
                    "--wandb.enable=false",
                    "--output_dir=" + str(training),
                ],
            )
            for export_primitive in (primitive,) if train_all else PRIMITIVES:
                name = f"{export_primitive}-{arm}"
                source_policy = (
                    training / "checkpoints/last/pretrained_model"
                    if export_primitive == primitive
                    else args.policies.resolve() / name
                )
                stage(
                    f"export-{name}",
                    [
                        *prepare,
                        "export",
                        "--source",
                        str(source_policy),
                        "--output",
                        str(job / "policies" / name),
                        "--primitive",
                        export_primitive,
                        "--arm",
                        arm,
                        "--action-steps",
                        str(args.action_steps),
                    ],
                )
        stage(
            "evaluate",
            [
                *stage.learned,
                "-m",
                "dimos_lerobot.demo_object_primitives",
                "--artifacts",
                str(job / "policies"),
                "--output",
                str(job / "evaluation"),
                "--start-seed",
                str(args.evaluation_seed),
                "--layouts",
                "4",
                "--action-steps",
                str(args.action_steps),
            ],
        )
        result = json.loads((job / "evaluation/result.json").read_text())
        save_manifest(
            job / "status.json",
            dict(
                stage="completed",
                state="completed",
                pid=os.getpid(),
                successes=result["successes"],
                total=result["total"],
                promoted=False,
                next="Review held-out failures; validate native interaction and both hands before promotion",
            ),
        )
    except Exception as exc:
        previous = (
            json.loads((job / "status.json").read_text()) if (job / "status.json").exists() else {}
        )
        save_manifest(job / "status.json", {**previous, "state": "failed", "error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=351000)
    parser.add_argument("--layouts", type=int, default=12)
    parser.add_argument("--evaluation-seed", type=int, default=352000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--action-steps", type=int, default=30)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--interactive-context", action="store_true")
    mode.add_argument("--approach-corrections", action="store_true")
    args = parser.parse_args()
    if args.layouts < 8 or args.steps < 1 or min(args.start_seed, args.evaluation_seed) < 0:
        parser.error("Use at least eight layouts, positive steps and nonnegative seeds")
    if not 1 <= args.action_steps <= 30:
        parser.error("Action steps must be between 1 and 30")
    if set(range(args.start_seed, args.start_seed + args.layouts)) & set(
        range(args.evaluation_seed, args.evaluation_seed + 4)
    ):
        parser.error("Evaluation seeds must be held out of correction collection")
    for primitive, arm in (
        (p, a)
        for a in ARMS
        for p in (
            PRIMITIVES if args.interactive_context or args.approach_corrections else ("place",)
        )
    ):
        seeds = {
            r["seed"]
            for r in json.loads((args.rehearsal / f"{primitive}-{arm}/manifest.json").read_text())[
                "episodes"
            ]
        }
        if seeds & set(range(args.evaluation_seed, args.evaluation_seed + 4)):
            parser.error("Evaluation seeds must be held out of rehearsal data")
    run(args)


if __name__ == "__main__":
    main()
