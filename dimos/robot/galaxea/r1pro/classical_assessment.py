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

"""Rank GraspGenX proposals in a separate process so planning cannot starve physics.

The simulator writes a request directory and starts this module; the skill polls
the simulator for the result. A runaway search is killed, not waited on.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import mujoco
import numpy as np
from pydantic import TypeAdapter

from dimos.robot.galaxea.r1pro.classical_planning import ClassicalGraspPlanner
from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout
from dimos.robot.galaxea.r1pro.object_primitive_state import PrimitiveSceneState


def rank_request(request: Path) -> list[dict[str, Any]]:
    spec = json.loads((request / "request.json").read_text())
    arrays = np.load(request / "request.npz")
    model = mujoco.MjModel.from_binary_path(spec["model"])
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["qpos"]
    data.qvel[:] = arrays["qvel"]
    data.ctrl[:] = arrays["ctrl"]
    mujoco.mj_forward(model, data)
    layout = TypeAdapter(ObjectLayout).validate_python(spec["layout"])
    scene = PrimitiveSceneState(model, data, layout, np.asarray(spec["home"], dtype=float))
    planner = ClassicalGraspPlanner(scene)
    plans = planner.rank(
        int(spec["index"]), arrays["poses"], arrays["scores"], arm=str(spec["arm"])
    )
    return [asdict(plan) for plan in plans]


def main() -> None:
    request = Path(sys.argv[1])
    try:
        result: dict[str, Any] = dict(plans=rank_request(request))
    except Exception as exc:
        result = dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    (request / "result.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
