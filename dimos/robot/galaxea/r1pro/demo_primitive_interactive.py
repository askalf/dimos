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

"""Exercise independent ACT commands through the native MCP interface in MuJoCo."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import requests

from dimos.constants import RECORDINGS_DIR
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.robot.galaxea.r1pro.primitive_blueprint import build_primitive_blueprint
from dimos.robot.galaxea.r1pro.primitive_sim import R1ProPrimitiveSim
from dimos.robot.galaxea.r1pro.sim_session import reserve_demo_session


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    source = build_primitive_blueprint(args.policies)
    atoms = tuple(
        replace(
            a,
            kwargs={
                **a.kwargs,
                **(
                    dict(
                        output=args.output / "session",
                        seed=args.seed,
                        occupied=args.occupied,
                        headless=not args.viewer,
                        bilateral_layout=not args.right_layout,
                    )
                    if a.module is R1ProPrimitiveSim
                    else {}
                ),
            },
        )
        for a in source.blueprints
    )
    blueprint = replace(source, blueprints=atoms).global_config(
        zenoh_scout_addr=args.zenoh_scout_addr, mcp_port=args.mcp_port
    )
    report: dict[str, Any] = dict(seed=args.seed, actions=[], success=False)
    coordinator = None
    with reserve_demo_session(args.zenoh_scout_addr, args.output):
        try:
            coordinator = ModuleCoordinator.build(blueprint)
            with requests.Session() as client:
                client.headers["Connection"] = "close"

                def call(tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
                    response = client.post(
                        f"http://127.0.0.1:{args.mcp_port}/mcp",
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": tool, "arguments": arguments or {}},
                        },
                        timeout=45,
                    )
                    response.raise_for_status()
                    result = response.json()
                    if "error" in result or result["result"].get("isError"):
                        raise RuntimeError(result)
                    value: dict[str, Any] = json.loads(result["result"]["content"][0]["text"])
                    return value

                report["initial"] = call("get_scene")
                for command in args.actions:
                    primitive, arm, target = command.split(":", 2)
                    if primitive not in ("pick", "place"):
                        raise ValueError("Actions use pick:arm:object or place:arm:region")
                    accepted = call(
                        f"{primitive}_object",
                        {"arm": arm, "object" if primitive == "pick" else "region": target},
                    )
                    row = dict(command=command, accepted=accepted)
                    report["actions"].append(row)
                    if not accepted["accepted"]:
                        raise RuntimeError(f"Action refused: {accepted}")
                    deadline = time.monotonic() + 240
                    while True:
                        outcome = call("wait_for_action", {"seconds": 20})
                        if outcome["state"] != "running":
                            break
                        if time.monotonic() > deadline:
                            call("stop_action")
                            raise RuntimeError("Action did not finish")
                    row.update(outcome=outcome, after=call("get_scene"))
                    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                    if not outcome["success"]:
                        raise RuntimeError(f"Action failed: {outcome}")
                report["success"] = True
                if args.stay_open:
                    print("Actions complete. Close the viewer or press Ctrl-C to stop.", flush=True)
                    sim = coordinator.get_instance(R1ProPrimitiveSim)
                    while sim.is_simulation_running():
                        time.sleep(0.5)
        except Exception as exc:
            report["error"] = str(exc)
            raise
        finally:
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            if coordinator is not None:
                coordinator.stop()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies",
        type=Path,
        default=RECORDINGS_DIR / "r1pro-act-task/jobs/independent-primitives-act-v1/policies",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=340000)
    parser.add_argument("--occupied", type=int, default=0)
    parser.add_argument("--right-layout", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--stay-open", action="store_true")
    parser.add_argument("--mcp-port", type=int, required=True)
    parser.add_argument("--zenoh-scout-addr", required=True)
    parser.add_argument(
        "--actions",
        nargs="+",
        default=[
            "pick:right:object_2",
            "place:right:tray",
            "pick:right:object_2",
            "place:right:table",
        ],
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
