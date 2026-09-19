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

import ast
from pathlib import Path

from dimos.constants import DIMOS_PROJECT_ROOT

# The worker always injects the host's GlobalConfig under this name; see
# PythonWorker.deploy_module in dimos/core/coordination/python_worker.py.
GLOBAL_CONFIG_KWARG = "g"

MODULE_BASES = {"Module", "ModuleBase"}


def _base_names(node: ast.ClassDef) -> list[str]:
    """Base class names of *node*, without subscripts or module prefixes."""
    return [ast.unparse(base).split("[")[0].rsplit(".", maxsplit=1)[-1] for base in node.bases]


def _find_class_defs(dimos_dir: Path) -> dict[str, list[ast.ClassDef]]:
    """Every class defined under dimos/, keyed by class name."""
    class_defs: dict[str, list[ast.ClassDef]] = {}
    for path in sorted(dimos_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_defs.setdefault(node.name, []).append(node)
    return class_defs


def _is_module_subclass(
    name: str, class_defs: dict[str, list[ast.ClassDef]], seen: set[str]
) -> bool:
    """Whether *name* resolves to Module/ModuleBase through in-repo base classes."""
    if name in MODULE_BASES:
        return True
    if name in seen:
        return False
    seen.add(name)
    return any(
        _is_module_subclass(base, class_defs, seen)
        for node in class_defs.get(name, [])
        for base in _base_names(node)
    )


def _init_of(node: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "__init__":
                return statement
    return None


def _accepts_global_config(init: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if init.args.kwarg is not None:
        return True
    declared = {arg.arg for arg in (*init.args.args, *init.args.kwonlyargs)}
    return GLOBAL_CONFIG_KWARG in declared


def find_modules_rejecting_global_config() -> list[tuple[Path, int, str]]:
    """Return (file, line, class name) for modules whose __init__ would reject `g`."""
    dimos_dir = DIMOS_PROJECT_ROOT / "dimos"
    class_defs = _find_class_defs(dimos_dir)
    hits: list[tuple[Path, int, str]] = []
    for path in sorted(dimos_dir.rglob("*.py")):
        # Test modules are constructed in-process by their own test, not
        # deployed through a worker, so the injected `g` never reaches them.
        if path.name.startswith("test_") or path.name == "conftest.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(_is_module_subclass(base, class_defs, set()) for base in _base_names(node)):
                continue
            init = _init_of(node)
            if init is not None and not _accepts_global_config(init):
                hits.append((path, node.lineno, node.name))
    return hits


def _forwards_kwargs_to_super(init: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether *init* calls the parent constructor with ``**kwargs``."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__init__"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        and any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "kwargs"
            for keyword in node.keywords
        )
        for node in ast.walk(init)
    )


def test_module_init_accepts_global_config_kwarg() -> None:
    """Fail if a deployable module's __init__ cannot accept the injected `g`."""
    dimos_dir = DIMOS_PROJECT_ROOT / "dimos"
    hits = find_modules_rejecting_global_config()
    if hits:
        listing = "\n".join(
            f"  - {p.relative_to(dimos_dir)}:{lineno}: {name}" for p, lineno, name in hits
        )
        raise AssertionError(
            f"Found module(s) whose __init__ rejects the {GLOBAL_CONFIG_KWARG!r} kwarg:\n"
            f"{listing}\n\n"
            "PythonWorker.deploy_module always passes the host GlobalConfig as "
            f"{GLOBAL_CONFIG_KWARG!r}, so deploying such a module fails with "
            f'"__init__() got an unexpected keyword argument {GLOBAL_CONFIG_KWARG!r}". '
            "Accept `**kwargs: Any` and forward it to super().__init__(), as the "
            "other modules with explicit constructor parameters do."
        )


def test_vlm_stream_tester_forwards_constructor_kwargs() -> None:
    """Pin that VlmStreamTester stores coordinator kwargs in ModuleConfig."""
    source = DIMOS_PROJECT_ROOT / "dimos/agents/testing/vlm_stream_tester.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VlmStreamTester"
    )
    init = _init_of(cls)
    assert init is not None
    assert init.args.kwarg is not None and init.args.kwarg.arg == "kwargs"
    assert _forwards_kwargs_to_super(init)
