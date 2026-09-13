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

import asyncio
from contextlib import aclosing
import signal
import sys

import pytest
import pytest_asyncio

from dimos.core.global_config import GlobalConfig
from dimos.teleop.pico.client import PCServiceClient
from dimos.teleop.pico.install import service_executable
from dimos.teleop.pico.module import PicoTeleopModule
from dimos.teleop.pico.service import pc_service_lifecycle


@pytest.fixture
def installation(tmp_path, monkeypatch):
    executable = tmp_path / "installation" / "RoboticsServiceProcess"
    executable.parent.mkdir()
    executable.touch(mode=0o755)
    monkeypatch.setattr("dimos.teleop.pico.service.STATE_DIR", tmp_path / "state")
    return executable.parent


@pytest_asyncio.fixture
async def children(mocker):
    """Substitute the vendor binary only; exercise actual subprocess teardown."""
    processes = []
    create_process = asyncio.create_subprocess_exec

    async def launch(*args, **kwargs):
        # Signal readiness after installing the handler, avoiding timing sleeps.
        process = await create_process(
            sys.executable,
            "-u",
            "-c",
            "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); signal.pause()",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        assert await asyncio.wait_for(process.stdout.readline(), 3) == b"ready\n"
        return process

    spawn = mocker.patch(
        "dimos.teleop.pico.service.asyncio.create_subprocess_exec", side_effect=launch
    )
    try:
        yield processes, spawn
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.wait(), 3)


@pytest.fixture
def client(pc_service):
    return PCServiceClient(f"127.0.0.1:{pc_service.port}")


@pytest.mark.asyncio
async def test_reuses_responding_service_without_starting_or_stopping_it(client, children):
    _, spawn = children
    async with pc_service_lifecycle(client, manage=True) as process:
        assert process is None
    spawn.assert_not_called()
    assert await client.is_ready() is True


@pytest.mark.asyncio
async def test_external_mode_does_not_require_installation_or_wait_for_service(
    client, children, mocker
):
    _, spawn = children
    probe = mocker.patch.object(client, "is_ready", return_value=False)
    async with pc_service_lifecycle(client, manage=False) as process:
        assert process is None
    spawn.assert_not_called()
    probe.assert_not_called()


@pytest.mark.asyncio
async def test_module_owns_service_and_reaps_it_even_if_it_ignores_terminate(
    client, installation, children, pc_service, mocker
):
    processes, spawn = children
    mocker.patch.object(client, "is_ready", side_effect=[False, True])
    mocker.patch("dimos.teleop.pico.module.PCServiceClient", return_value=client)
    module = PicoTeleopModule(g=GlobalConfig(), pc_service_dir=installation)
    for stream in (module.body_tracking, module.teleop_buttons, module.cmd_vel):
        mocker.patch.object(stream, "publish")
    try:
        await asyncio.to_thread(module.start)
        pid, _ = await asyncio.to_thread(pc_service.subscriptions.get, timeout=3)
        assert module.tracking_status()["pc_service_pid"] == processes[0].pid
        assert spawn.call_args.args == (str(installation / "RoboticsServiceProcess"),)
        assert spawn.call_args.kwargs["cwd"] == installation
        assert spawn.call_args.kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(installation))
    finally:
        await asyncio.to_thread(module.stop)
    assert processes[0].returncode == -signal.SIGKILL
    assert pc_service.cancellations.get(timeout=3) == pid
    assert module.tracking_status()["pc_service_connected"] is False


@pytest.mark.asyncio
async def test_readiness_timeout_fails_startup_and_reaps_child(
    client, installation, children, mocker
):
    processes, _ = children
    mocker.patch.object(client, "is_ready", return_value=False)
    with pytest.raises(RuntimeError, match="did not become ready.*pc-service"):
        async with pc_service_lifecycle(
            client, manage=True, directory=installation, startup_timeout=0.02
        ):
            pytest.fail("unready service must not permit module startup")
    assert processes[0].returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_cancelling_startup_reaps_child(client, installation, children, mocker):
    processes, _ = children
    probing_child = asyncio.Event()

    async def probe():
        if processes:
            probing_child.set()
            await asyncio.Future()
        return False

    mocker.patch.object(client, "is_ready", side_effect=probe)

    async def start():
        async with pc_service_lifecycle(client, manage=True, directory=installation):
            pytest.fail("cancelled startup must not yield")

    task = asyncio.create_task(start())
    try:
        await asyncio.wait_for(probing_child.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert processes[0].returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_premature_exit_reports_failure_before_startup(
    client, installation, children, mocker
):
    processes, _ = children

    async def probe():
        if processes:
            processes[0].kill()
            await processes[0].wait()
        return False

    mocker.patch.object(client, "is_ready", side_effect=probe)
    with pytest.raises(RuntimeError, match="exited with code.*pc-service"):
        async with pc_service_lifecycle(client, manage=True, directory=installation):
            pytest.fail("dead service must not permit startup")
    assert processes[0].returncode == -signal.SIGKILL


def test_explicit_missing_installation_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--pc-service-dir"):
        service_executable(tmp_path)


@pytest.mark.asyncio
async def test_heartbeat_probe_does_not_replace_tracking_subscription(client, pc_service):
    async with aclosing(client.events()) as events:
        assert await anext(events) is None
        await asyncio.to_thread(pc_service.subscriptions.get, timeout=3)
        assert await client.is_ready() is True
        assert pc_service.subscriptions.empty()
        pc_service.fail_heartbeat.set()
        assert await client.is_ready() is False
