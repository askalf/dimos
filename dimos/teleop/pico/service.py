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

"""Own the vendor PC Service process for the lifetime of native PICO input."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path

from dimos.constants import STATE_DIR
from dimos.teleop.pico.client import PCServiceClient
from dimos.teleop.pico.install import service_executable
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()


@asynccontextmanager
async def pc_service_lifecycle(
    client: PCServiceClient,
    *,
    manage: bool,
    directory: Path | None = None,
    startup_timeout: float = 10.0,
) -> AsyncIterator[asyncio.subprocess.Process | None]:
    """Start and reap our child; leave an external service under its owner's control.

    Run the binary directly: the vendor shell launcher backgrounds it, which
    would detach the real service from module shutdown and startup error checks.
    """
    if not manage or await client.is_ready():
        logger.info("Using external XRoboToolkit PC Service", endpoint=client.target)
        yield None
        return

    executable = service_executable(directory)
    service_dir = executable.parent
    env = os.environ.copy()
    for key, paths in {
        "LD_LIBRARY_PATH": [service_dir, service_dir / "lib", service_dir / "SDK/x64"],
        "QT_PLUGIN_PATH": [service_dir / "plugins"],
        "QT_QML_PATH": [service_dir / "qml"],
    }.items():
        env[key] = os.pathsep.join([*(str(path) for path in paths), *filter(None, [env.get(key)])])

    log_path = STATE_DIR / "pico" / f"pc-service-{os.getpid()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        process = await asyncio.create_subprocess_exec(
            str(executable),
            cwd=service_dir,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            try:
                async with asyncio.timeout(startup_timeout):
                    while True:
                        ready = await client.is_ready()
                        if process.returncode is not None:
                            raise RuntimeError(
                                f"XRoboToolkit PC Service exited with code {process.returncode}; "
                                f"see {log_path}"
                            )
                        if ready:
                            break
                        await asyncio.sleep(0.1)
            except TimeoutError as exc:
                raise RuntimeError(
                    f"XRoboToolkit PC Service did not become ready within {startup_timeout}s; "
                    f"see {log_path}"
                ) from exc
            logger.info(
                "Started XRoboToolkit PC Service",
                pid=process.pid,
                endpoint=client.target,
                log_path=str(log_path),
            )
            yield process
        finally:
            await _stop_process(process)
            logger.info("Stopped XRoboToolkit PC Service", pid=process.pid)
