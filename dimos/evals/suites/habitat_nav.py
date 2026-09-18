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

"""Go to a named object in a Habitat scene; graded on arrival, time, facing, bumps, path.

One scene file per scene under ``scenes/habitat/``: the ground-truth boxes
(``detections``, the ``detection3d_array_to_dict`` layout, ROS world frame) and the
``cases`` (label, spawn, end point beside the object, geodesic distance). Every arm
gets ``go to the <label> at (x, y)``; the boxes are published by ``demo-objects``
so text-only agents see them in ``world_state``.

    # the planner alone, the end point straight to /goal
    dimos evals run dimos.evals.suites.habitat_nav --agent dimos.evals.agents.topic \\
        --set send=goal --set send_type=point --set done=goal_reached --set done_type=Bool
    # the TypeSafe reactive agent
    dimos evals run dimos.evals.suites.habitat_nav --agent dimos.evals.agents.topic \\
        --set 'modules=["type-safe-agent"]' --set trace=TypeSafeAgent
    # a coding agent with dimOS (go_to / stop / finish tools) or without it (raw topics)
    dimos evals run dimos.evals.suites.habitat_nav --agent dimos.evals.agents.dimcode --set model=gpt-6-astra
    dimos evals run dimos.evals.suites.habitat_nav --agent dimos.evals.agents.pi --set no_dimos=true --set model=gpt-6-astra
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path

from dimos.evals.environments.habitat import HabitatEnvironment
from dimos.evals.nav_metrics import (
    box_of,
    read_cmds,
    read_declared,
    read_poses,
    score_navigation,
    write_metrics,
)
from dimos.evals.types import EvalCase, Outcome, Suite, recording

SCENES = Path(__file__).parent / "scenes" / "habitat"
BLUEPRINT = ["habitat-nav", "mcp-server", "demo-objects", "nav-skills"]
TIMEOUT_S = float(os.environ.get("DIMOS_EVAL_TIMEOUT_S", 1800))
# What grading and replay need; images and clouds stay out (1 GB per few minutes otherwise).
RECORD_TOPICS = ("odom", "cmd_vel", "goal", "path", "goal_reached", "stop_movement", "finished")
# The depth scan is in base_link with the floor at z = 0; keep the floor out of the sectors.
MODULE_ENV = {"TYPESAFEAGENT__LIDAR_Z_MIN": "0.1", "RAWROBOTBRIDGE__LIDAR_Z_MIN": "0.1"}


def grade_nav(
    end_xy: tuple[float, float], box: tuple[float, float, float, float]
) -> Callable[[Outcome], float]:
    def grade(o: Outcome) -> float:
        start = json.loads(o.artifacts["episode"].read_text()).get("task_start_ts", 0.0)
        with recording(o) as store:
            poses = [p for p in read_poses(store) if p[0] >= start]
            cmds = [c for c in read_cmds(store) if c[0] >= start]
            m = score_navigation(poses, cmds, end_xy, box, declared_at=read_declared(store))
        write_metrics(
            m, o.artifacts["recording"].parent / "nav_metrics.json", end_xy=end_xy, box=box
        )
        return m.score()

    return grade


def cases_for(scene_file: Path) -> list[EvalCase]:
    scene = json.loads(scene_file.read_text())
    boxes = {d["id"]: box_of(d["center_xyz"], d["size_xyz"]) for d in scene["detections"]}
    out = []
    for c in scene["cases"]:
        x, y = c["end_xy"]
        label = c["label"]
        out.append(
            EvalCase(
                id=f"{scene['scene_id']}_{label.replace(' ', '_')}",
                inputs=f"go to the {label} at ({x:.2f}, {y:.2f})",
                environment=HabitatEnvironment(
                    blueprint=BLUEPRINT,
                    scene_id=scene["scene_id"],
                    start_position_ros_override=tuple(c["spawn_xyz"]),
                    start_yaw_deg=c["spawn_yaw_deg"],
                    raw_bridge=True,  # inert unless an agent connects; identical launches per arm
                    raw_topics=("world_state", "cmd_vel", "finished"),
                    record_topics=RECORD_TOPICS,
                    tour=tuple((float(x), float(y)) for x, y in scene.get("tour", ())),
                    extra_env={"DEMOOBJECTS__SCENE_JSON": str(scene_file), **MODULE_ENV},
                ),
                grade=grade_nav((x, y), boxes[c["object_id"]]),
                timeout_s=TIMEOUT_S,
                threshold=0.5,  # passed == reached
                tags=frozenset({"habitat", "nav", scene["scene_id"], label}),
            )
        )
    return out


SUITE: Suite = [case for f in sorted(SCENES.glob("*.json")) for case in cases_for(f)]
