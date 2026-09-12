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

"""Persistent primitive collection, warm-start training and fresh-layout ACT evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest
from dimos.robot.galaxea.r1pro.object_primitives import ARMS, PRIMITIVES
from dimos.robot.galaxea.r1pro.primitive_training_job import TrainingStages


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[4]
    job = args.output.resolve()
    job.mkdir(parents=True, exist_ok=True)
    contract = dict(
        source=str(args.source.resolve()),
        source_sha256=hashlib.sha256((args.source / "model.safetensors").read_bytes()).hexdigest(),
        rehearsal=str(args.rehearsal.resolve()) if args.rehearsal else None,
        start_seed=args.start_seed,
        layouts=args.layouts,
        steps=args.steps,
        evaluation_seed=args.evaluation_seed,
    )
    settings = job / "settings.json"
    if settings.exists() and json.loads(settings.read_text()) != contract:
        raise ValueError("Resume settings differ from the original job")
    save_manifest(settings, contract)
    stage = TrainingStages(root, job)
    learned = stage.learned

    try:
        stage(
            "collect",
            [
                sys.executable,
                "-m",
                "dimos.robot.galaxea.r1pro.demo_collect_primitives",
                "--output",
                str(job / "collection"),
                "--layouts",
                str(args.layouts),
                "--start-seed",
                str(args.start_seed),
            ],
        )
        for arm in ARMS:
            for primitive in PRIMITIVES:
                name = f"{primitive}-{arm}"
                source = job / "collection" / name
                if args.rehearsal:
                    merged = job / "merged" / name
                    merged.mkdir(parents=True, exist_ok=True)
                    manifest = json.loads((source / "manifest.json").read_text())
                    rows = []
                    seen = set()
                    for folder in (args.rehearsal / name, source):
                        other = json.loads((folder / "manifest.json").read_text())
                        if other["profile"] != manifest["profile"]:
                            raise ValueError("Rehearsal primitive contract differs")
                        for row in other["episodes"]:
                            key = (row["seed"], row["selected"])
                            if key in seen:
                                raise ValueError("Duplicate rehearsal layout/object")
                            seen.add(key)
                            rows.append(
                                {
                                    **row,
                                    "file": str((folder / row["file"]).resolve()),
                                    "initial_state": str((folder / row["initial_state"]).resolve()),
                                }
                            )
                    manifest["episodes"] = rows
                    manifest["rehearsal_source"] = str(args.rehearsal.resolve())
                    save_manifest(merged / "manifest.json", manifest)
                    source = merged
                manifest = json.loads((source / "manifest.json").read_text())
                if len(manifest["episodes"]) < 16:
                    raise RuntimeError(f"Too few accepted {name} episodes for this pilot")
                dataset = job / "datasets" / name
                initialization = job / "initializations" / name
                training = job / "training" / name
                stage(
                    f"convert-{name}",
                    [
                        *learned,
                        "-m",
                        "dimos_lerobot.prepare_primitive_act",
                        "convert",
                        "--source",
                        str(source),
                        "--output",
                        str(dataset),
                    ],
                )
                stage(
                    f"initialize-{name}",
                    [
                        *learned,
                        "-m",
                        "dimos_lerobot.prepare_primitive_act",
                        "initialize",
                        "--source",
                        str(args.source),
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
                training_source = (
                    ["--resume=true", "--config_path=" + str(resume)]
                    if resume.exists()
                    else ["--policy.path=" + str(initialization)]
                )
                stage(
                    f"train-{name}",
                    [
                        *learned,
                        "-m",
                        "lerobot.scripts.lerobot_train",
                        *training_source,
                        f"--dataset.repo_id=local/r1pro-{name}",
                        "--dataset.root=" + str(dataset),
                        "--dataset.eval_split=0.125",
                        "--policy.device=cuda",
                        "--policy.push_to_hub=false",
                        "--policy.optimizer_lr=.00005",
                        "--policy.optimizer_lr_backbone=.00001",
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
                stage(
                    f"export-{name}",
                    [
                        *learned,
                        "-m",
                        "dimos_lerobot.prepare_primitive_act",
                        "export",
                        "--source",
                        str(training / "checkpoints/last/pretrained_model"),
                        "--output",
                        str(job / "policies" / name),
                        "--primitive",
                        primitive,
                        "--arm",
                        arm,
                    ],
                )
        stage(
            "evaluate",
            [
                *learned,
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
                next="Review physical failures and validate native interactive integration before promotion",
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rehearsal", type=Path)
    parser.add_argument("--layouts", type=int, default=24)
    parser.add_argument("--start-seed", type=int, default=321000)
    parser.add_argument("--evaluation-seed", type=int, default=330000)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    if args.layouts < 1 or args.steps < 1 or min(args.start_seed, args.evaluation_seed) < 0:
        parser.error("Use positive layouts/steps and nonnegative seeds")
    if args.start_seed <= args.evaluation_seed < args.start_seed + args.layouts:
        parser.error("Evaluation layouts must be held out of training")
    run(args)


if __name__ == "__main__":
    main()
