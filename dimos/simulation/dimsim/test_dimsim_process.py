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

import subprocess
import sys
import threading

from dimos.core.global_config import GlobalConfig
from dimos.simulation.dimsim import dimsim_process
from dimos.simulation.dimsim.dimsim_process import DimSimProcess


def test_stop_terminates_writer_before_closing_log_pipe(mocker):
    ready = threading.Event()
    mocker.patch.object(dimsim_process.logger, "info", side_effect=lambda *_: ready.set())
    simulator = DimSimProcess(GlobalConfig())
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time; print('ready', file=sys.stderr, flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        simulator.process = process
        simulator._start_log_reader()
        stop = threading.Thread(target=simulator.stop)
        try:
            assert ready.wait(5)
            stop.start()
            stop.join(timeout=3)
            assert not stop.is_alive(), "stop blocked closing a pipe before terminating its writer"
            assert process.poll() is not None
            assert simulator.process is None
            simulator.stop()  # repeated cleanup is harmless
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            if stop.ident is not None:
                stop.join(timeout=5)
