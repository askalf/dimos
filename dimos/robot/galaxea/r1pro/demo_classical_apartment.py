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

"""Exercise classical apartment commands through MCP, without an external language model."""

import argparse
from collections.abc import Callable
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import requests

from dimos.agents.mcp.mcp_client import McpClient
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import global_config
from dimos.core.transport_factory import make_transport
from dimos.robot.galaxea.r1pro.classical_blueprint import build_classical_apartment
from dimos.robot.galaxea.r1pro.classical_sim import R1ProClassicalSim
from dimos.robot.galaxea.r1pro.open_space_blueprint import build_classical_open_space
from dimos.robot.galaxea.r1pro.open_space_sim import R1ProOpenSpaceSim
from dimos.robot.galaxea.r1pro.sim_session import reserve_demo_session


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    sim_class = R1ProOpenSpaceSim if args.open_space else R1ProClassicalSim
    source = (
        build_classical_open_space(agent=args.agent)
        if args.open_space
        else build_classical_apartment(agent=args.agent)
    )
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
                        **(
                            {"scene_package": args.scene_package}
                            if getattr(args, "scene_package", None)
                            else {}
                        ),
                    )
                    if a.module is sim_class
                    else dict(
                        mcp_server_url=f"http://127.0.0.1:{args.mcp_port}/mcp",
                        **({"model": args.model} if args.model else {}),
                    )
                    if a.module is McpClient
                    else {}
                ),
            },
        )
        for a in source.blueprints
    )
    blueprint = replace(source, blueprints=atoms).global_config(
        zenoh_scout_addr=args.zenoh_scout_addr, mcp_port=args.mcp_port
    )
    sources = [
        *Path(__file__).parent.glob("classical*.py"),
        *Path(__file__).parent.glob("open_space*.py"),
        Path(__file__).with_name("primitive_sim.py"),
        Path(__file__).with_name("home_kinematics.py"),
        Path(__file__).with_name("object_primitive_state.py"),
        Path(__file__).with_name("primitive_scene.py"),
    ]
    report: dict[str, Any] = dict(
        seed=args.seed,
        actions=[],
        success=False,
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    )
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
                    calls = {
                        "pick": ("pick_object", {"arm": arm, "object": target}),
                        "place": ("place_object", {"arm": arm, "region": target}),
                        "go": ("go_to", {"destination": target}),
                        "tray_pick": ("pick_up_tray", {}),
                        "tray_place": ("put_down_tray", {"region": target}),
                    }
                    if primitive not in calls:
                        raise ValueError(
                            "Actions use pick:arm:object, place:arm:region, go:arm:region, "
                            "tray_pick::, or tray_place::region"
                        )
                    accepted = call(*calls[primitive])
                    row = dict(command=command, accepted=accepted)
                    report["actions"].append(row)
                    if not accepted["accepted"]:
                        raise RuntimeError(f"Action refused: {accepted}")
                    deadline = time.monotonic() + 600
                    while True:
                        outcome = call("wait_for_action", {"seconds": 20})
                        (args.output / "status.json").write_text(
                            json.dumps(dict(command=command, **outcome), indent=2) + "\n"
                        )
                        if outcome["state"] != "running":
                            break
                        if time.monotonic() > deadline:
                            call("stop_action")
                            raise RuntimeError("Action did not finish")
                    row.update(outcome=outcome, after=call("get_scene"))
                    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                    if not outcome["success"]:
                        if args.recover_on_failure:
                            report["recovery_accepted"] = call("recover_action")
                            if report["recovery_accepted"]["accepted"]:
                                deadline = time.monotonic() + 120
                                while True:
                                    recovery = call("wait_for_action", {"seconds": 20})
                                    if recovery["state"] != "running":
                                        break
                                    if time.monotonic() > deadline:
                                        call("stop_action")
                                        raise RuntimeError("Recovery did not finish")
                                report["recovery"] = recovery
                                report["after_recovery"] = call("get_scene")
                        raise RuntimeError(f"Action failed: {outcome}")
                if args.say:
                    report["conversation"] = run_conversation(args, call)
                    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                report["success"] = True
                if args.stay_open:
                    print("Actions complete. Close the viewer or press Ctrl-C to stop.", flush=True)
                    sim = coordinator.get_instance(sim_class)
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


def run_conversation(
    args: argparse.Namespace, call: Callable[..., dict[str, Any]]
) -> list[dict[str, Any]]:
    """Send each --say line through the language agent like HumanCLI, then record the outcome."""
    g = global_config.model_copy(update={"zenoh_scout_addr": args.zenoh_scout_addr})
    human = make_transport("/human_input", g=g)
    agent = make_transport("/agent", g=g)
    idle = make_transport("/agent_idle", g=g)
    replies: list[dict[str, Any]] = []
    state = dict(idle=True, busy_seen=False)

    def on_agent(message: Any) -> None:
        replies.append(
            dict(
                kind=type(message).__name__,
                content=str(getattr(message, "content", ""))[:2000],
                tool_calls=[
                    dict(name=c.get("name"), args=c.get("args"))
                    for c in getattr(message, "tool_calls", None) or []
                ],
            )
        )

    def on_idle(flag: Any) -> None:
        state["idle"] = bool(flag)
        if not flag:
            state["busy_seen"] = True

    agent.subscribe(on_agent)
    idle.subscribe(on_idle)
    time.sleep(2.0)
    rows = []
    for text in args.say:
        first = len(replies)
        state.update(idle=True, busy_seen=False)
        human.publish(text)
        print(f"Said: {text}", flush=True)
        deadline = time.monotonic() + 900
        while True:
            time.sleep(0.5)
            if state["busy_seen"] and state["idle"]:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"The agent did not finish handling: {text}")
        while call("get_scene")["action"].get("state") == "running":
            time.sleep(2.0)
        after = call("get_scene")
        rows.append(dict(say=text, replies=replies[first:], action=after["action"], after=after))
        print(f"Agent finished: {after['action']}", flush=True)
    for transport in (human, agent, idle):
        transport.stop()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=340000)
    parser.add_argument("--scene-package", type=Path)
    parser.add_argument("--open-space", action="store_true")
    parser.add_argument("--occupied", type=int, default=0)
    parser.add_argument("--right-layout", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--stay-open", action="store_true")
    parser.add_argument("--recover-on-failure", action="store_true")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--model", default=None, help="Language model for --agent runs")
    parser.add_argument("--say", nargs="*", default=[])
    parser.add_argument("--mcp-port", type=int, required=True)
    parser.add_argument("--zenoh-scout-addr", required=True)
    parser.add_argument(
        "--actions",
        nargs="*",
        default=["pick:right:object_2", "place:right:worktable"],
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
