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

"""DimOS lifecycle owner of the emulator, including its disposable model snapshot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Protocol

import mujoco
from pydantic import InstanceOf

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.sim2.runtime import SimulationRuntime
from dimos.sim2.spec import WorldConfig
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SimulationSpec(Spec, Protocol):
    def describe(self) -> dict[str, Any]: ...


class SimulationModuleConfig(ModuleConfig):
    world: InstanceOf[WorldConfig]
    sim_id: str = "sim"
    viewer: bool = False


class SimulationModule(Module):
    config: SimulationModuleConfig
    dedicated_worker = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._runtime: SimulationRuntime | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._viewer: subprocess.Popen[bytes] | None = None
        self._lifecycle_lock = threading.RLock()
        self._failure: str | None = None

    @rpc
    def build(self) -> None:
        with self._lifecycle_lock:
            if self._runtime is not None:
                return
            runtime = SimulationRuntime(self.config.world, self.config.sim_id)
            directory = tempfile.TemporaryDirectory(prefix="dimos-sim2-")
            try:
                mujoco.mj_saveModel(runtime.model, str(Path(directory.name) / "world.mjb"), None)
            except BaseException:
                runtime.close()
                directory.cleanup()
                raise
            self._runtime, self._directory = runtime, directory
            (Path(directory.name) / "world.json").write_text(json.dumps(self.describe()))
            logger.info(
                "sim2 world prepared", robots=list(self.config.world.robots), nq=runtime.model.nq
            )

    @rpc
    def describe(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self._runtime is None or self._directory is None:
                raise RuntimeError("SimulationModule.build() has not completed")
            return {
                "model": str(Path(self._directory.name) / "world.mjb"),
                "snapshot": self._runtime.snapshot_descriptor.to_dict(),
            }

    @rpc
    def start(self) -> None:
        super().start()
        self.build()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sim2-physics", daemon=True)
        self._thread.start()
        if self.config.viewer:
            description = self.describe()
            executable = Path(sys.executable)
            if sys.platform == "darwin":
                executable = executable.with_name("mjpython")
            self._viewer = subprocess.Popen(
                [
                    str(executable),
                    "-m",
                    "dimos.sim2.viewer",
                    str(Path(description["model"]).with_suffix(".json")),
                ]
            )

    def _run(self) -> None:
        runtime = self._runtime
        assert runtime is not None
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                runtime.step()
                deadline += self.config.world.timestep
                now = time.monotonic()
                if now - deadline > 0.1:
                    deadline = now
                self._stop.wait(max(0.0, deadline - now))
        except Exception as error:
            logger.exception("sim2 physics stopped")
            with self._lifecycle_lock:
                self._failure = str(error)
            runtime.snapshots.set_lifecycle("faulted")
            for binding in runtime.robots.values():
                binding.channel.set_lifecycle("faulted")

    @rpc
    def status(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            runtime = self._runtime
            if runtime is None:
                return {"running": False, "error": self._failure}
            with runtime.lock:
                return {
                    "running": self._failure is None and not self._stop.is_set(),
                    "sim_time": float(runtime.data.time),
                    "steps": runtime.tick,
                    "episode": runtime.episode,
                    "error": self._failure,
                    "robots": {
                        name: runtime.data.xpos[b.root].tolist()
                        for name, b in runtime.robots.items()
                    },
                }

    @rpc
    def reset(self) -> None:
        with self._lifecycle_lock:
            if self._runtime is None:
                raise RuntimeError("simulation is not running")
            self._runtime.reset()

    @rpc
    def set_spawn(
        self,
        robot_id: str,
        xyz: tuple[float, float, float],
        rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        with self._lifecycle_lock:
            if self._runtime is None:
                raise RuntimeError("simulation is not running")
            self._runtime.set_spawn(robot_id, xyz, rpy)

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("sim2 physics did not stop")
        if self._viewer is not None:
            self._viewer.terminate()
            try:
                self._viewer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._viewer.kill()
                self._viewer.wait()
            self._viewer = None
        with self._lifecycle_lock:
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None
        super().stop()
