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

import os
import signal
import subprocess

import pytest

from dimos.e2e_tests.dimos_cli_call import DimosCliCall


def test_stop_allows_parent_to_clean_up_workers_and_is_idempotent(mocker):
    call = DimosCliCall()
    process = mocker.Mock()
    call.process = process
    kill_group = mocker.patch.object(os, "killpg")
    call.stop()
    call.stop()
    process.send_signal.assert_called_once_with(signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=30)
    kill_group.assert_not_called()


def test_shutdown_timeout_terminates_the_owned_process_group(mocker):
    call = DimosCliCall()
    process = mocker.Mock(pid=123)
    process.wait.side_effect = [subprocess.TimeoutExpired("owned", 30), 0]
    process.poll.return_value = 0
    call.process = process
    kill_group = mocker.patch.object(os, "killpg")
    with pytest.raises(AssertionError, match="did not shut down"):
        call.stop()
    kill_group.assert_called_once_with(123, signal.SIGKILL)
