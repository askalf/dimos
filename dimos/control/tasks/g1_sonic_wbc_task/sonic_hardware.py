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

"""Read-only Jetson performance checks required before SONIC control."""

from __future__ import annotations

from pathlib import Path
import subprocess

CPU_FREQUENCY_ROOT = Path("/sys/devices/system/cpu/cpufreq")
DEVFREQ_ROOT = Path("/sys/class/devfreq")


def _output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run {' '.join(command)}: {exc}") from exc


def ensure_sonic_max_performance() -> None:
    """Fail unless the Jetson is in MAXN with CPU/GPU clocks locked."""
    nvpmodel = _output(["nvpmodel", "-q"])
    if "NV Power Mode: MAXN" not in nvpmodel:
        raise RuntimeError("SONIC requires Jetson MAXN mode. Run `sudo nvpmodel -m 0`, then retry.")

    # These kernel limits are readable without sudo, including in module workers.
    # Configuring the limits still requires the operator to run jetson_clocks.
    cpu_policies = list(CPU_FREQUENCY_ROOT.glob("policy[0-9]*"))
    gpu_devices = [
        path for path in DEVFREQ_ROOT.glob("*") if path.name.endswith((".gpu", ".ga10b", ".gv11b"))
    ]
    unlocked: list[str] = []
    if not cpu_policies or not all(
        _locked_limits(path, "scaling_min_freq", "scaling_max_freq") for path in cpu_policies
    ):
        unlocked.append("CPU")
    if not gpu_devices or not all(
        _locked_limits(path, "min_freq", "max_freq") for path in gpu_devices
    ):
        unlocked.append("GPU")
    if unlocked:
        raise RuntimeError(
            "SONIC requires locked Jetson clocks for CPU/GPU. Run `sudo jetson_clocks`, "
            f"then retry (unlocked: {', '.join(unlocked)})."
        )


def _locked_limits(directory: Path, minimum_name: str, maximum_name: str) -> bool:
    try:
        minimum = int((directory / minimum_name).read_text(encoding="utf-8"))
        maximum = int((directory / maximum_name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read Jetson clock limits in {directory}: {exc}") from exc
    return 0 < minimum == maximum
