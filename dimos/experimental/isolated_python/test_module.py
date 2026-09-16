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

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from dimos.core.core import rpc
from dimos.experimental.isolated_python.module import (
    IsolatedPythonModule,
    IsolatedPythonModuleConfig,
    isolated_python_run_command,
    prepare_isolated_python,
)


class Contract(IsolatedPythonModule):
    implementation = "runtime:Runtime"
    config: IsolatedPythonModuleConfig

    @rpc
    def value(self) -> int:
        raise NotImplementedError


def test_sibling_project_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "contract.py"
    source.touch()
    monkeypatch.setattr(
        "dimos.experimental.isolated_python.module.inspect.getfile", lambda _: str(source)
    )
    module = Contract()
    try:
        with pytest.raises(FileNotFoundError, match="sibling 'python/'"):
            module.runtime_project  # noqa: B018
    finally:
        module.stop()


def test_prepare_installs_host_code_without_changing_runtime_dependencies(tmp_path, mocker):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").touch()
    mocker.patch("dimos.experimental.isolated_python.module.DIMOS_PROJECT_ROOT", checkout)
    run = mocker.patch("dimos.experimental.isolated_python.module.subprocess.run")
    run.return_value.returncode = 0

    prepare_isolated_python(tmp_path, {})

    assert [call.args[0] for call in run.call_args_list] == [
        ["uv", "sync", "--frozen"],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(tmp_path / ".venv/bin/python"),
            "--no-deps",
            "--editable",
            str(checkout),
        ],
    ]
    assert isolated_python_run_command(tmp_path, "python", "script.py") == [
        "uv",
        "run",
        "--no-sync",
        "python",
        "script.py",
    ]


def test_installed_host_pins_runtime_to_host_version(tmp_path, mocker):
    mocker.patch("dimos.experimental.isolated_python.module.DIMOS_PROJECT_ROOT", tmp_path)
    mocker.patch("dimos.experimental.isolated_python.module.version", return_value="1.2.3")
    run = mocker.patch("dimos.experimental.isolated_python.module.subprocess.run")
    run.return_value.returncode = 0

    prepare_isolated_python(tmp_path, {})

    assert run.call_args.args[0][-2:] == ["--no-deps", "dimos==1.2.3"]


def test_failed_sync_does_not_install_host(tmp_path, mocker):
    run = mocker.patch("dimos.experimental.isolated_python.module.subprocess.run")
    run.return_value.returncode = 1
    run.return_value.stdout = ""
    run.return_value.stderr = "unsatisfiable dependencies"

    with pytest.raises(RuntimeError, match="unsatisfiable dependencies"):
        prepare_isolated_python(tmp_path, {})

    assert run.call_count == 1


def test_pixi_wraps_preparation_and_launch(tmp_path, mocker):
    (tmp_path / "pixi.toml").touch()
    run = mocker.patch("dimos.experimental.isolated_python.module.subprocess.run")
    run.return_value.returncode = 0

    prepare_isolated_python(tmp_path, {})

    assert all(
        call.args[0][:4] == ["pixi", "run", "--executable", "uv"] for call in run.call_args_list
    )
    assert isolated_python_run_command(tmp_path, "python") == [
        "pixi",
        "run",
        "--executable",
        "uv",
        "run",
        "--no-sync",
        "python",
    ]


def test_runtime_environment_uses_sibling_virtualenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/parent/.venv")
    module = Contract(extra_env={"EXAMPLE_SETTING": "configured"})
    try:
        env = module._runtime_env()

        assert "VIRTUAL_ENV" not in env
        assert env["EXAMPLE_SETTING"] == "configured"
    finally:
        module.stop()


def test_host_build_prepares_and_builds_runtime(mocker: MockerFixture) -> None:
    module = Contract()
    prepare = mocker.patch.object(module, "_run_prepare")
    spawn = mocker.patch.object(module, "_spawn_runtime")
    runtime_client = mocker.Mock()
    connect = mocker.patch.object(
        module,
        "_connect_runtime",
        side_effect=lambda: setattr(module, "_runtime_client", runtime_client),
    )
    try:
        module.build()

        prepare.assert_called_once_with()
        spawn.assert_called_once_with()
        connect.assert_called_once_with()
        runtime_client.build.assert_called_once_with()
    finally:
        module.stop()


def test_runtime_build_skips_environment_preparation(mocker: MockerFixture) -> None:
    module = Contract(_isolated_python_runtime=True)
    prepare = mocker.patch.object(module, "_run_prepare")
    spawn = mocker.patch.object(module, "_spawn_runtime")
    try:
        module.build()

        prepare.assert_not_called()
        spawn.assert_not_called()
    finally:
        module.stop()
