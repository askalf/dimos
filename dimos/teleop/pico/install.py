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

"""Prepare the pinned vendor PC Service during the module's build phase."""

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from filelock import FileLock
import requests

from dimos.constants import CACHE_DIR
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

PC_SERVICE_VERSION = "1.0.0"
# GitHub release asset digests, pinned independently of the mutable release tag.
# https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0
_PACKAGES = {
    "ubuntu-22.04-amd64": (
        "XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb",
        "61961067eb4b41f81ed7cae35f4690dbb0ddfefb329a12b24e0b90ebc46ada91",
    ),
    "ubuntu-24.04-amd64": (
        "XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb",
        "bce661f0be0b8a246ceecb2e5f1675a81c26b834648dc7fdf23f8c0bfe2a5d19",
    ),
    "ubuntu-arm64-headless": (
        "XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb",
        "532c605dfa1a02b05b7c285b856c91771c78623cded30ef5b16ea371de49ed5f",
    ),
}
_SERVICE_PATH = Path("opt/apps/roboticsservice/RoboticsServiceProcess")
_SYSTEM_SERVICE_PATH = Path("/") / _SERVICE_PATH


def _package_variant() -> str | None:
    if platform.system() != "Linux":
        return None
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        return None
    version = release.get("VERSION_ID", "")
    if release.get("ID") != "ubuntu" or version not in {"22.04", "24.04"}:
        return None
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "ubuntu-arm64-headless"
    if machine in {"x86_64", "amd64"}:
        return f"ubuntu-{version}-amd64"
    return None


def _cache_directory(variant: str) -> Path:
    # A reviewed checksum update must not silently reuse the previous package.
    directory = f"{variant}-{_PACKAGES[variant][1][:16]}"
    return CACHE_DIR / "xrobotoolkit-pc-service" / PC_SERVICE_VERSION / directory


def _find_service(directory: Path | None) -> Path | None:
    if directory is not None:
        candidates = [directory.expanduser() / _SERVICE_PATH.name]
    else:
        data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        candidates = [
            _SYSTEM_SERVICE_PATH,
            data_home / "dimos/xrobotoolkit-pc-service" / _SERVICE_PATH,
            data_home / f"dimos/xrobotoolkit-pc-service-{PC_SERVICE_VERSION}" / _SERVICE_PATH,
        ]
        if variant := _package_variant():
            candidates.append(_cache_directory(variant) / _SERVICE_PATH)
    for executable in candidates:
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable.resolve()
    return None


def service_executable(directory: Path | None) -> Path:
    """Locate prepared files without doing setup during process startup."""
    executable = _find_service(directory)
    if executable is None:
        raise FileNotFoundError(
            "XRoboToolkit PC Service is missing or not executable. Run the blueprint to prepare "
            "it in PicoTeleopModule.build(), or pass --pc-service-dir PATH to the directory "
            "containing RoboticsServiceProcess."
        )
    return executable


def ensure_pc_service(directory: Path | None = None) -> Path:
    """Reuse local files or atomically cache a verified package, without sudo.

    Explicit paths are operator-managed: a typo must fail instead of downloading
    into or replacing that location. Only the default cache is provisioned here.
    """
    if directory is not None:
        return service_executable(directory)
    if executable := _find_service(None):
        return executable
    variant = _package_variant()
    if variant is None:
        raise RuntimeError(
            "Automatic XRoboToolkit PC Service setup supports Ubuntu 22.04/24.04 x86_64 "
            "and ARM64. "
            "Use --pc-service-dir for a compatible local service, or --manage-pc-service false "
            "with an external service (--xrobotoolkit-host/--xrobotoolkit-port)."
        )
    extractor = shutil.which("dpkg-deb")
    if extractor is None:
        raise RuntimeError("XRoboToolkit setup requires dpkg-deb (Ubuntu's dpkg package).")

    filename, expected_sha256 = _PACKAGES[variant]
    target = _cache_directory(variant)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target.parent / f"{target.name}.lock"), timeout=300):
        if executable := _find_service(None):
            return executable
        if target.exists():
            raise RuntimeError(f"Incomplete XRoboToolkit cache at {target}; remove it and retry.")
        url = (
            "https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/"
            f"v{PC_SERVICE_VERSION}/{filename}"
        )
        logger.info("Preparing XRoboToolkit PC Service", url=url, directory=str(target))
        with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as tmp:
            archive = Path(tmp) / filename
            digest = hashlib.sha256()
            try:
                with requests.get(url, stream=True, timeout=(10, 30)) as response:
                    response.raise_for_status()
                    with archive.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Failed to download XRoboToolkit PC Service from {url}: {exc}"
                ) from exc
            if digest.hexdigest() != expected_sha256:
                raise RuntimeError(
                    "XRoboToolkit PC Service checksum mismatch; package was not extracted."
                )

            extracted = Path(tmp) / "unpacked"
            try:
                # Extraction does not install a system package or run its maintainer scripts.
                subprocess.run(
                    [extractor, "--extract", str(archive), str(extracted)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"XRoboToolkit PC Service extraction failed: {exc.stderr}"
                ) from exc
            service_executable(extracted / _SERVICE_PATH.parent)
            extracted.rename(target)
        logger.info("Prepared XRoboToolkit PC Service", directory=str(target))
        return target / _SERVICE_PATH
