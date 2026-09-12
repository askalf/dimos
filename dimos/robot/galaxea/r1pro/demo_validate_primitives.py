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

"""Wait for a local policy job, then exercise both arms and two-hand holds through native MCP."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest
from dimos.robot.galaxea.r1pro.primitive_training_job import TrainingStages

CASES = {
    "right": [
        "pick:right:object_2",
        "place:right:tray",
        "pick:right:object_2",
        "place:right:table",
    ],
    "left": ["pick:left:object_1", "place:left:tray", "pick:left:object_1", "place:left:table"],
    "both": ["pick:right:object_2", "pick:left:object_1", "place:right:table", "place:left:table"],
}


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[4]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    contract = dict(
        wait_for=str(args.wait_for.resolve()),
        seed=args.seed,
        cases=CASES,
        mcp_port=args.mcp_port,
        zenoh_scout_addr=args.zenoh_scout_addr,
    )
    save_manifest(output / "settings.json", contract)
    status: dict[str, Any] = dict(
        state="waiting", stage="training", pid=os.getpid(), promoted=False
    )
    save_manifest(output / "status.json", status)
    try:
        while True:
            training = json.loads((args.wait_for / "status.json").read_text())
            if training["state"] == "failed":
                raise RuntimeError(f"Source policy job failed: {training.get('error')}")
            if training["state"] == "completed":
                break
            time.sleep(30)
        stage = TrainingStages(root, output)
        results = {}
        for case, commands in CASES.items():
            case_output = output / case
            try:
                stage(
                    case,
                    [
                        sys.executable,
                        "-m",
                        "dimos.robot.galaxea.r1pro.demo_primitive_interactive",
                        "--policies",
                        str(args.wait_for.resolve() / "policies"),
                        "--output",
                        str(case_output),
                        "--seed",
                        str(args.seed),
                        "--mcp-port",
                        str(args.mcp_port),
                        "--zenoh-scout-addr",
                        args.zenoh_scout_addr,
                        "--actions",
                        *commands,
                    ],
                )
                results[case] = json.loads((case_output / "result.json").read_text())["success"]
            except RuntimeError as exc:
                results[case] = False
                print(f"{case} failed: {exc}", flush=True)
            save_manifest(output / "results.json", results)
        status.update(
            state="completed",
            stage="completed",
            results=results,
            success=all(results.values()),
            promoted=False,
        )
    except Exception as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        save_manifest(output / "status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=340000)
    parser.add_argument("--mcp-port", type=int, required=True)
    parser.add_argument("--zenoh-scout-addr", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
