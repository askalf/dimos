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

from __future__ import annotations

from collections.abc import Callable
import inspect
import sys
from typing import Any, cast

import pytest

from dimos.core.coordination.blueprints import Blueprint
from dimos.core.global_config import global_config
from dimos.core.module import Module, ModuleBase
from dimos.robot.all_blueprints import all_blueprints, all_modules
from dimos.robot.get_all_blueprints import get_blueprint_by_name, get_module_by_name
from dimos.robot.test_all_blueprints import (
    OPTIONAL_DEPENDENCIES,
    OPTIONAL_ERROR_SUBSTRINGS,
    SELF_HOSTED_BLUEPRINTS,
)

# The worker always injects the host's GlobalConfig under this name; see
# PythonWorker.deploy_module in dimos/core/coordination/python_worker.py.
GLOBAL_CONFIG_KWARG = "g"


def _get_blueprint_or_skip(blueprint_name: str) -> Blueprint:
    try:
        return get_blueprint_by_name(blueprint_name)
    except ModuleNotFoundError as e:
        if e.name in OPTIONAL_DEPENDENCIES:
            pytest.skip(f"Skipping due to missing optional dependency: {e.name}")
        raise
    except Exception as e:
        message = str(e)
        if any(substring in message for substring in OPTIONAL_ERROR_SUBSTRINGS):
            pytest.skip(f"Skipping due to missing optional dependency: {message}")
        raise


def _get_module_or_skip(module_name: str) -> Blueprint:
    try:
        return get_module_by_name(module_name)
    except ModuleNotFoundError as e:
        if e.name in OPTIONAL_DEPENDENCIES:
            pytest.skip(f"Skipping due to missing optional dependency: {e.name}")
        raise
    except Exception as e:
        message = str(e)
        if any(substring in message for substring in OPTIONAL_ERROR_SUBSTRINGS):
            pytest.skip(f"Skipping due to missing optional dependency: {message}")
        raise


def _accepts_global_config(module: type[ModuleBase]) -> bool:
    """Whether ``module(**kwargs)`` can take the injected ``g``: a parameter of
    that name that binds by keyword, or ``**kwargs``."""
    parameters = inspect.signature(module.__init__).parameters
    named = parameters.get(GLOBAL_CONFIG_KWARG)
    if named is not None and named.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _rejecting_modules(blueprint: Blueprint) -> list[str]:
    return sorted(
        {
            atom.module.__name__
            for atom in blueprint.blueprints
            if not _accepts_global_config(atom.module)
        }
    )


def _rejection_message(name: str, rejecting: list[str]) -> str:
    return (
        f"{name!r} deploys module(s) whose __init__ rejects the "
        f"{GLOBAL_CONFIG_KWARG!r} kwarg: {', '.join(rejecting)}. PythonWorker.deploy_module "
        f"always passes the host GlobalConfig as {GLOBAL_CONFIG_KWARG!r}, so deploying them "
        "fails with TypeError. Accept **kwargs: Any and forward it to super().__init__()."
    )


def _self_hosted_params(names: list[str]) -> list[str | pytest.ParameterSet]:
    self_hosted = set(SELF_HOSTED_BLUEPRINTS)
    return [
        pytest.param(name, marks=pytest.mark.self_hosted) if name in self_hosted else name
        for name in sorted(names)
    ]


def _blueprint_params() -> list[str | pytest.ParameterSet]:
    return _self_hosted_params(list(all_blueprints))


def _module_params() -> list[str | pytest.ParameterSet]:
    return _self_hosted_params(list(all_modules))


@pytest.mark.parametrize("blueprint_name", _blueprint_params())
def test_blueprint_modules_accept_global_config_kwarg(
    blueprint_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail when a blueprint deploys a module whose __init__ rejects ``g``."""
    # The multi-robot blueprints read ROBOT_IPS at import time.
    monkeypatch.setattr(global_config, "robot_ips", "192.0.2.10,192.0.2.11")
    blueprint = _get_blueprint_or_skip(blueprint_name)

    rejecting = _rejecting_modules(blueprint)
    assert not rejecting, _rejection_message(blueprint_name, rejecting)


@pytest.mark.parametrize("module_name", _module_params())
def test_registered_modules_accept_global_config_kwarg(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Modules deployable on their own go through the same worker injection."""
    monkeypatch.setattr(global_config, "robot_ips", "192.0.2.10,192.0.2.11")
    blueprint = _get_module_or_skip(module_name)

    rejecting = _rejecting_modules(blueprint)
    assert not rejecting, _rejection_message(module_name, rejecting)


def test_vlm_stream_tester_forwards_constructor_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """VlmStreamTester passes every kwarg it does not consume on to Module."""
    from dimos.agents.testing.vlm_stream_tester import VlmStreamTester

    forwarded: dict[str, Any] = {}

    def record(self: Module, **kwargs: Any) -> None:
        forwarded.update(kwargs)

    # Module.__init__ builds streams and starts the RPC transport; the tester's
    # own constructor needs none of that, so recording what reaches Module is
    # enough to pin the forwarding.
    monkeypatch.setattr(Module, "__init__", record)
    sentinel = object()
    VlmStreamTester(prompt="unused", g=sentinel, instance_name="vlm-test-1")

    assert forwarded == {GLOBAL_CONFIG_KWARG: sentinel, "instance_name": "vlm-test-1"}


def test_vlm_stream_tester_keeps_its_own_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kwargs the tester consumes stay local and are not forwarded."""
    from dimos.agents.testing.vlm_stream_tester import VlmStreamTester

    forwarded: dict[str, Any] = {}

    def record(self: Module, **kwargs: Any) -> None:
        forwarded.update(kwargs)

    monkeypatch.setattr(Module, "__init__", record)
    tester = VlmStreamTester(
        prompt="describe",
        num_queries=0,
        query_interval_s=0.0,
        max_image_age_s=0.0,
        max_image_gap_s=0.0,
        g=object(),
    )

    assert forwarded.keys() == {GLOBAL_CONFIG_KWARG}
    assert tester._prompt == "describe"
    # Falsy-but-valid option values are kept verbatim, not replaced by defaults.
    assert (tester._num_queries, tester._query_interval_s) == (0, 0.0)
    assert (tester._max_image_age_s, tester._max_image_gap_s) == (0.0, 0.0)


def test_vlm_stream_tester_deploys_with_only_the_injected_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blueprint supplies no kwargs, so ``g`` alone must construct it."""
    from dimos.agents.testing.vlm_stream_tester import VlmStreamTester

    monkeypatch.setattr(Module, "__init__", lambda self, **kwargs: None)
    tester = VlmStreamTester(**{GLOBAL_CONFIG_KWARG: global_config})

    # Defaults survive the kwargs-only construction path.
    assert tester._prompt == "What do you see?"
    assert tester._num_queries == 10


def test_accepts_global_config_reads_the_binding_rules() -> None:
    """The signature check must agree with ``module_class(**kwargs)``."""

    def accepts(source: str) -> bool:
        namespace: dict[str, Any] = {}
        exec(f"class _M:\n    {source}", namespace)
        return _accepts_global_config(namespace["_M"])

    assert not accepts("def __init__(self, prompt: str = 'p') -> None: ...")
    assert not accepts("def __init__(self, *args) -> None: ...")
    assert accepts("def __init__(self, prompt: str = 'p', **kwargs) -> None: ...")
    assert accepts("def __init__(self, g=None) -> None: ...")
    assert accepts("def __init__(self, *, g=None) -> None: ...")
    # A positional-only ``g`` cannot be bound by keyword, so the worker's
    # ``module_class(**kwargs)`` still raises; it must not count as accepting.
    assert not accepts("def __init__(self, g, /) -> None: ...")
    # Only the parameter's kind decides; a variadic named ``g`` binds nothing
    # by keyword either.
    assert not accepts("def __init__(self, *g) -> None: ...")
    assert accepts("def __init__(self, *g, **kwargs) -> None: ...")


# Every constructor shape the check can meet, each with defaults throughout so
# that the only reason a call can fail is the ``g`` keyword itself.
BINDING_SHAPES = (
    "def __init__(self, prompt: str = 'p') -> None: ...",
    "def __init__(self, prompt: str = 'p', **kwargs) -> None: ...",
    "def __init__(self, g=None) -> None: ...",
    "def __init__(self, *, g=None) -> None: ...",
    "def __init__(self, g=None, /) -> None: ...",
    "def __init__(self, g=None, /, **kwargs) -> None: ...",
    "def __init__(self, *g) -> None: ...",
    "def __init__(self, *g, **kwargs) -> None: ...",
    "def __init__(self, **g) -> None: ...",
    "def __init__(self, *args) -> None: ...",
)


@pytest.mark.parametrize("source", BINDING_SHAPES)
def test_accepts_global_config_agrees_with_a_real_keyword_call(source: str) -> None:
    """Check the verdict against the binding the worker actually performs.

    ``_accepts_global_config`` reads a signature; ``PythonWorker`` calls
    ``module_class(**kwargs)``. Asserting the two agree keeps the check honest
    without restating CPython's binding rules a second time.
    """
    namespace: dict[str, Any] = {}
    exec(f"class _M:\n    {source}", namespace)
    module = namespace["_M"]

    try:
        module(**{GLOBAL_CONFIG_KWARG: global_config})
        binds = True
    except TypeError:
        binds = False

    assert _accepts_global_config(module) is binds, source


def test_module_subclass_without_constructor_accepts_the_kwarg() -> None:
    """A module that declares no __init__ inherits Module's ``**kwargs``."""

    class _InheritsInit(Module):
        pass

    assert _accepts_global_config(_InheritsInit)


def _outcome(lookup: Callable[[str], Blueprint], name: str) -> str:
    """Classify a lookup as ``skipped`` or as the exception type it raised.

    The skip is caught rather than allowed to propagate: a test that lets it
    through is reported as skipped, which would hide a triage that skips too
    much instead of failing.
    """
    try:
        lookup(name)
    except pytest.skip.Exception:
        return "skipped"
    except BaseException as e:
        return type(e).__name__
    return "returned"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # In OPTIONAL_DEPENDENCIES, so the extras are simply absent.
        (ModuleNotFoundError("No module named 'pyzed'", name="pyzed"), "skipped"),
        # Message in OPTIONAL_ERROR_SUBSTRINGS.
        (RuntimeError("ZED SDK not installed"), "skipped"),
        # A missing module that is not an optional extra is a real failure.
        (ModuleNotFoundError("No module named 'numba'", name="numba"), "ModuleNotFoundError"),
        # Anything else must propagate, or a broken blueprint reads as a skip.
        (RuntimeError("blueprint is broken"), "RuntimeError"),
    ],
)
def test_lookup_skips_only_optional_dependency_failures(
    error: Exception, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing optional dependencies skip; every other failure still fails.

    The triage is the one ``test_all_blueprints.py`` uses. On a runner with the
    extras installed nothing reaches the skip branch, so it is pinned here
    rather than by the parametrized checks above.
    """

    def raise_error(name: str) -> Blueprint:
        raise error

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "get_blueprint_by_name", raise_error)
    monkeypatch.setattr(module, "get_module_by_name", raise_error)

    assert _outcome(_get_blueprint_or_skip, "some-blueprint") == expected
    assert _outcome(_get_module_or_skip, "some-module") == expected


def test_rejecting_modules_are_named_once_each_in_a_stable_order() -> None:
    """The failure message lists every offender, deduplicated and sorted."""

    class _Rejects:
        def __init__(self, prompt: str = "p") -> None: ...

    class _AlsoRejects:
        def __init__(self, prompt: str = "p") -> None: ...

    class _Accepts:
        def __init__(self, **kwargs: Any) -> None: ...

    class _Atom:
        def __init__(self, module: type) -> None:
            self.module = module

    class _Blueprint:
        def __init__(self, modules: list[type]) -> None:
            self.blueprints = [_Atom(module) for module in modules]

    # A module repeated across atoms must be named once, not twice.
    blueprint = _Blueprint([_Rejects, _AlsoRejects, _Accepts, _Rejects])
    rejecting = _rejecting_modules(cast("Blueprint", blueprint))

    assert rejecting == ["_AlsoRejects", "_Rejects"]

    message = _rejection_message("some-blueprint", rejecting)
    assert "'some-blueprint'" in message
    assert "_AlsoRejects, _Rejects" in message
    assert "_Accepts" not in message
