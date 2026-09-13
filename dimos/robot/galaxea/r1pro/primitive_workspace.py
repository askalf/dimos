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

"""Audit the demonstrated ACT starting workspace without claiming learned success."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dimos.robot.galaxea.r1pro.grasping_task import HOME_TCP
from dimos.robot.galaxea.r1pro.object_primitives import (
    ARMS,
    PRIMITIVES,
    Arm,
    Primitive,
    primitive_profile,
)


@dataclass(frozen=True)
class PrimitiveWorkspace:
    """Observed starting-state bounds; coverage is necessary but not sufficient."""

    target_min: tuple[float, ...]
    target_max: tuple[float, ...]
    torso_min: tuple[float, ...]
    torso_max: tuple[float, ...]
    episodes: int
    manifest_sha256: str
    starts: tuple[tuple[float, ...], ...] = ()

    def preferred_torsos(self, target: NDArray[Any]) -> list[NDArray[np.float64]]:
        """Seed positioning from actual demonstrations near this target, not averaged postures."""
        if not self.starts:
            return [np.asarray(self.torso_min, dtype=float)]
        samples = np.asarray(self.starts, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 7 or not np.isfinite(samples).all():
            raise ValueError("Invalid target/torso samples in the policy workspace")
        order = np.argsort(np.linalg.norm((samples[:, :3] - target) / [0.05, 0.05, 0.02], axis=1))
        result: list[NDArray[np.float64]] = []
        for index in order:
            torso = samples[index, 3:]
            if all(np.linalg.norm(torso - q) > 0.025 for q in result):
                result.append(torso.copy())
            if len(result) == 3:
                break
        return result

    def covers(self, target: NDArray[Any], torso: NDArray[Any]) -> bool:
        # Allow measured servo error, not an invented expansion of the dataset.
        return bool(
            np.all(target >= np.asarray(self.target_min) - 0.005)
            and np.all(target <= np.asarray(self.target_max) + 0.005)
            and np.all(torso >= np.asarray(self.torso_min) - 0.01)
            and np.all(torso <= np.asarray(self.torso_max) + 0.01)
        )


def audit_workspace(manifest: Path, primitive: Primitive, arm: Arm) -> PrimitiveWorkspace:
    """Recover base-relative task targets from named, measured policy features."""
    raw = manifest.read_bytes()
    source = json.loads(raw)
    profile = primitive_profile(primitive, arm)
    if (
        source["profile"] != profile.name
        or source["primitive"] != primitive
        or source["arm"] != arm
    ):
        raise ValueError("Demonstration profile does not match the requested policy")
    home = np.asarray(HOME_TCP) * ([1, -1, 1] if arm == "left" else [1, 1, 1])
    targets, torsos = [], []
    for episode in source["episodes"]:
        if not episode["success"]:
            raise ValueError("A policy workspace must come from verified demonstrations")
        with np.load(manifest.parent / episode["file"], allow_pickle=False) as data:
            values = data["observation.environment_state"][episode.get("frame_start", 0)]
        expected = 61 if primitive == "pick" else 64
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError("Invalid primitive observation in workspace audit")
        home_index = 18 if primitive == "pick" else 21
        target_index = 0 if primitive == "pick" else 3
        targets.append(
            values[target_index : target_index + 3] + home - values[home_index : home_index + 3]
        )
        torsos.append(values[-12:-8])
    if not targets:
        raise ValueError("No demonstrations to audit")
    target, torso = np.asarray(targets), np.asarray(torsos)
    return PrimitiveWorkspace(
        tuple(map(float, target.min(axis=0))),
        tuple(map(float, target.max(axis=0))),
        tuple(map(float, torso.min(axis=0))),
        tuple(map(float, torso.max(axis=0))),
        len(targets),
        hashlib.sha256(raw).hexdigest(),
        tuple(tuple(map(float, row)) for row in np.column_stack((target, torso))),
    )


def save_workspaces(manifests: Path, output: Path) -> None:
    """Write provenance-backed bounds beside an experimental four-policy bundle."""
    result = {
        f"{primitive}-{arm}": asdict(
            audit_workspace(manifests / f"{primitive}-{arm}" / "manifest.json", primitive, arm)
        )
        for primitive in PRIMITIVES
        for arm in ARMS
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
