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

"""Author-only catalog: executable questions and their literal scorer references."""

import ast
from importlib import import_module
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from dimos.constants import DIMOS_PROJECT_ROOT


def reference(grade: ast.AST) -> dict[str, Any]:
    """Read only known literal scorer forms; never guess an unsupported answer."""
    for node in ast.walk(grade):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in {
            "exact",
            "numeric",
            "rank_order",
            "count",
            "boolean",
            "choice",
            "measurement",
            "order",
        }:
            continue
        answer = ast.literal_eval(node.args[0])
        result = {"answer": answer, "scorer": name}
        if name == "measurement":
            result.update(
                tolerance=ast.literal_eval(node.args[1]), band=ast.literal_eval(node.args[2])
            )
        elif name == "numeric":
            result.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
        return result
    raise ValueError("Unsupported scorer reference; update the catalog reader explicitly")


def split_question(prompt: str) -> tuple[str, str, str]:
    question = prompt.split("\n\n", 1)[-1]
    body, separator, response_format = question.rpartition(" Return ")
    if not separator:
        body, response_format = question, ""
    option = re.search(r"\bA\)\s", body)
    if option:
        return body[: option.start()].strip(), body[option.start() :].strip(), response_format
    return body, "", response_format


def dataset_path(family: str, configured: str, asset_root: Path) -> str:
    """Prefer explicit environment configuration, then installed authoring assets."""
    if configured == "default" or Path(configured).is_file():
        return configured
    if family == "hssd":
        candidate = asset_root / "hssd-hab/hssd-hab.scene_dataset_config.json"
    elif family == "replicacad":
        candidate = (
            asset_root / "versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json"
        )
    else:
        candidate = Path(configured)
    return str(candidate)


def build_catalog(
    root: Path = DIMOS_PROJECT_ROOT, asset_root: Path | None = None
) -> dict[str, Any]:
    asset_root = asset_root or Path(
        os.environ.get("HABITAT_REVIEW_ASSET_ROOT", Path.home() / "Documents/habitat-sim/data")
    )
    context_file = root / "dimos/evals/suites/habitat/SCENES.md"
    sections = re.split(r"^## ([A-Za-z0-9_]+)\s*$", context_file.read_text(), flags=re.MULTILINE)
    common_context = sections[0]
    contexts = {
        sections[i]: common_context + "## " + sections[i] + sections[i + 1]
        for i in range(1, len(sections), 2)
    }
    catalog = {}
    for path in sorted((root / "dimos/evals/suites/habitat").glob("*/*.py")):
        if path.name.startswith("_"):
            continue
        module = import_module(".".join(path.relative_to(root).with_suffix("").parts))
        suite = module.SUITE
        tree = ast.parse(path.read_text())
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "SUITE"
        )
        if not isinstance(assignment.value, ast.List) or len(assignment.value.elts) != len(suite):
            raise ValueError(f"Unsupported suite structure: {path}")
        family, slug = path.parent.name, path.stem
        questions = []
        for case, node in zip(suite, assignment.value.elts, strict=True):
            if not isinstance(node, ast.Call):
                raise ValueError(f"Unsupported case: {case.id}")
            grade = next((kw.value for kw in node.keywords if kw.arg == "grade"), None)
            if grade is None:
                grade = node.args[2]
            ref = reference(grade)
            # Confirm transcription against the actual grader, not only the AST.
            score = case.grade(
                SimpleNamespace(trajectory=SimpleNamespace(final_answer=str(ref["answer"])))
            )
            if score != 1:
                raise ValueError(f"Extracted reference does not earn full credit: {case.id}")
            wording, options, answer_format = split_question(case.inputs)
            questions.append(
                {
                    "id": case.id,
                    "question": wording,
                    "options": options,
                    "answer_format": answer_format,
                    "reference": ref,
                    "tags": sorted(case.tags),
                }
            )
        cfg = suite[0].environment.config
        dataset = dataset_path(family, cfg.scene_dataset_config, asset_root)
        scene_id = cfg.scene_id
        if family == "test" and not Path(scene_id).is_file():
            scene_id = str(asset_root / "versioned_data/habitat_test_scenes/apartment_1.glb")
        spawn = getattr(cfg, "start_position_ros_override", None)
        catalog[slug] = {
            "id": slug,
            "family": family,
            "scene_id": scene_id,
            "dataset": dataset,
            "asset_available": (dataset == "default" and Path(scene_id).is_file())
            or Path(dataset).is_file(),
            "start_position_ros": spawn,
            "start_yaw_deg": cfg.start_yaw_deg,
            "seed": cfg.seed,
            "questions": questions,
            "context": contexts[slug],
            "suite_path": str(path.relative_to(root)),
        }
    return catalog
