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

import inspect
from typing import Any

import pytest

from dimos.core.coordination.blueprints import Blueprint
from dimos.core.global_config import global_config
from dimos.core.module import Module, ModuleBase
from dimos.robot.all_blueprints import all_blueprints
from dimos.robot.get_all_blueprints import get_blueprint_by_name
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


def _accepts_global_config(module: type[ModuleBase]) -> bool:
    """Whether ``module(**kwargs)`` can take the injected ``g``: a parameter of
    that name, or ``**kwargs``."""
    parameters = inspect.signature(module.__init__).parameters
    return GLOBAL_CONFIG_KWARG in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def _blueprint_params() -> list[str | pytest.ParameterSet]:
    self_hosted = set(SELF_HOSTED_BLUEPRINTS)
    return [
        pytest.param(name, marks=pytest.mark.self_hosted) if name in self_hosted else name
        for name in sorted(all_blueprints)
    ]


@pytest.mark.parametrize("blueprint_name", _blueprint_params())
def test_blueprint_modules_accept_global_config_kwarg(
    blueprint_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail when a blueprint deploys a module whose __init__ rejects ``g``."""
    # The multi-robot blueprints read ROBOT_IPS at import time.
    monkeypatch.setattr(global_config, "robot_ips", "192.0.2.10,192.0.2.11")
    blueprint = _get_blueprint_or_skip(blueprint_name)

    rejecting = sorted(
        {
            atom.module.__name__
            for atom in blueprint.blueprints
            if not _accepts_global_config(atom.module)
        }
    )
    assert not rejecting, (
        f"Blueprint {blueprint_name!r} deploys module(s) whose __init__ rejects the "
        f"{GLOBAL_CONFIG_KWARG!r} kwarg: {', '.join(rejecting)}. PythonWorker.deploy_module "
        f"always passes the host GlobalConfig as {GLOBAL_CONFIG_KWARG!r}, so deploying them "
        "fails with TypeError. Accept **kwargs: Any and forward it to super().__init__()."
    )


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
