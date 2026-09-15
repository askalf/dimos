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

import re

import pytest
from pytest_mock import MockerFixture
from typer.testing import CliRunner

from dimos.cli.dimos import main
import dimos.utils.cache as cache_utils


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--help"],
        ["run", "unitree-go2", "--help"],
        ["run", "--help", "unitree-go2"],
        ["run", "unitree-go2", "--daemon", "--help"],
        ["run", "unitree-go2", "--disable", "unknown-module", "--help"],
        ["run", "unitree-go2", "--n-workers", "invalid", "--help"],
        ["run", "unitree-go2", "--config", "missing.json", "--help"],
        ["run", "unitree-go2", "--config-help", "--help"],
        ["run", "unknown-blueprint", "--help"],
        ["run", "external-package.blueprint", "--help"],
    ],
)
def test_run_help_exits_before_runtime(args: list[str], mocker: MockerFixture) -> None:
    cache_guard = mocker.patch.object(cache_utils, "cache_usage_guard")

    result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, (result.output, result.exception)
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "Usage:" in output
    assert "Start a robot blueprint" in output
    assert "--daemon" in output
    cache_guard.assert_not_called()


def test_run_without_help_still_requires_a_blueprint() -> None:
    result = CliRunner().invoke(main, ["run"])

    assert result.exit_code == 2
    assert "Missing argument" in result.output
    assert "ROBOT_TYPES" in result.output
