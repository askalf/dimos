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

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import threading

import pytest
import requests
from requests_mock import ANY

from dimos.teleop.pico import install

PACKAGE = b"test vendor archive"
SERVICE_BINARY = b"test executable"


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(install, "_SYSTEM_SERVICE_PATH", tmp_path / "system/RoboticsServiceProcess")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    release = {"ID": "ubuntu", "VERSION_ID": "22.04"}
    monkeypatch.setattr(install.platform, "freedesktop_os_release", lambda: release)
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        install,
        "_PACKAGES",
        {
            variant: (filename, hashlib.sha256(PACKAGE).hexdigest())
            for variant, (filename, _) in install._PACKAGES.items()
        },
    )
    return release


@pytest.fixture
def extraction(mocker):
    def extract(argv, **kwargs):
        assert argv[1] == "--extract"
        assert Path(argv[2]).read_bytes() == PACKAGE
        executable = Path(argv[3]) / "opt/apps/roboticsservice/RoboticsServiceProcess"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(SERVICE_BINARY)
        executable.chmod(0o755)
        return subprocess.CompletedProcess(argv, 0)

    return mocker.patch.object(install.subprocess, "run", side_effect=extract)


@pytest.mark.parametrize("version", ["22.04", "24.04"])
@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_download_verified_package_and_reuse_without_network(
    environment, extraction, requests_mock, monkeypatch, version, machine
):
    environment["VERSION_ID"] = version
    monkeypatch.setattr(install.platform, "machine", lambda: machine)
    filename = (
        "XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb"
        if machine == "aarch64"
        else f"XRoboToolkit_PC_Service_1.0.0_ubuntu_{version}_amd64.deb"
    )
    url = (
        "https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/"
        f"{filename}"
    )
    download = requests_mock.get(url, content=PACKAGE)

    executable = install.ensure_pc_service()
    assert executable.read_bytes() == SERVICE_BINARY
    assert executable.is_relative_to(install.CACHE_DIR)
    assert install.service_executable(None) == executable
    assert install.ensure_pc_service() == executable
    assert download.call_count == 1
    extraction.assert_called_once()


def test_checksum_failure_never_extracts_and_can_be_retried(environment, extraction, requests_mock):
    requests_mock.get(ANY, content=b"changed upstream package")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        install.ensure_pc_service()
    extraction.assert_not_called()
    with pytest.raises(FileNotFoundError):
        install.service_executable(None)

    requests_mock.get(ANY, content=PACKAGE)
    assert install.ensure_pc_service().read_bytes() == SERVICE_BINARY


def test_updated_pin_does_not_reuse_old_cached_package(
    environment, extraction, requests_mock, monkeypatch
):
    requests_mock.get(ANY, content=PACKAGE)
    previous = install.ensure_pc_service()
    variant = "ubuntu-22.04-amd64"
    filename, _ = install._PACKAGES[variant]
    monkeypatch.setitem(install._PACKAGES, variant, (filename, "0" * 64))

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        install.ensure_pc_service()

    assert previous.read_bytes() == SERVICE_BINARY
    extraction.assert_called_once()


def test_failed_download_leaves_no_usable_cache_and_can_be_retried(
    environment, extraction, requests_mock
):
    requests_mock.get(ANY, exc=requests.ReadTimeout("connection interrupted"))
    with pytest.raises(RuntimeError, match="Failed to download"):
        install.ensure_pc_service()
    extraction.assert_not_called()
    requests_mock.get(ANY, content=PACKAGE)
    assert install.ensure_pc_service().read_bytes() == SERVICE_BINARY


def test_partial_extraction_is_never_published(environment, extraction, requests_mock):
    extract = extraction.side_effect

    def fail_after_writing_executable(argv, **kwargs):
        extract(argv, **kwargs)
        raise subprocess.CalledProcessError(2, argv, stderr="truncated package")

    extraction.side_effect = fail_after_writing_executable
    requests_mock.get(ANY, content=PACKAGE)
    with pytest.raises(RuntimeError, match="extraction failed: truncated package"):
        install.ensure_pc_service()
    with pytest.raises(FileNotFoundError):
        install.service_executable(None)
    extraction.side_effect = extract
    assert install.ensure_pc_service().read_bytes() == SERVICE_BINARY


def test_concurrent_builds_publish_one_complete_installation(
    environment, extraction, requests_mock
):
    downloading = threading.Event()
    finish_download = threading.Event()

    def download(request, context):
        downloading.set()
        assert finish_download.wait(timeout=5)
        return PACKAGE

    requests_mock.get(ANY, content=download)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(install.ensure_pc_service)
        try:
            assert downloading.wait(timeout=5)
            second = executor.submit(install.ensure_pc_service)
        finally:
            finish_download.set()
        assert first.result(timeout=5) == second.result(timeout=5)
    assert requests_mock.call_count == 1
    extraction.assert_called_once()


@pytest.mark.parametrize("location", ["explicit", "system", "legacy"])
def test_existing_service_is_reused_without_download_or_extraction(
    environment, extraction, tmp_path, requests_mock, location
):
    paths = {
        "explicit": tmp_path / "custom/RoboticsServiceProcess",
        "system": tmp_path / "system/RoboticsServiceProcess",
        "legacy": tmp_path
        / "data/dimos/xrobotoolkit-pc-service-1.0.0/opt/apps/roboticsservice/RoboticsServiceProcess",
    }
    executable = paths[location]
    executable.parent.mkdir(parents=True)
    executable.write_bytes(SERVICE_BINARY)
    executable.chmod(0o755)
    directory = executable.parent if location == "explicit" else None

    assert install.ensure_pc_service(directory) == executable
    extraction.assert_not_called()
    assert requests_mock.call_count == 0


def test_invalid_explicit_path_does_not_download_or_write_there(
    environment, extraction, tmp_path, requests_mock
):
    directory = tmp_path / "mistyped-directory"
    with pytest.raises(FileNotFoundError, match="--pc-service-dir"):
        install.ensure_pc_service(directory)
    assert not directory.exists()
    assert requests_mock.call_count == 0


@pytest.mark.parametrize(
    ("system", "machine", "version"),
    [("Darwin", "arm64", "22.04"), ("Linux", "riscv64", "22.04"), ("Linux", "x86_64", "20.04")],
)
def test_unsupported_platform_has_actionable_error_before_download(
    environment, extraction, requests_mock, monkeypatch, system, machine, version
):
    environment["VERSION_ID"] = version
    monkeypatch.setattr(install.platform, "system", lambda: system)
    monkeypatch.setattr(install.platform, "machine", lambda: machine)
    with pytest.raises(RuntimeError, match="--manage-pc-service false"):
        install.ensure_pc_service()
    assert requests_mock.call_count == 0
    extraction.assert_not_called()


def test_missing_extractor_fails_before_download(environment, requests_mock, monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="requires dpkg-deb"):
        install.ensure_pc_service()
    assert requests_mock.call_count == 0
