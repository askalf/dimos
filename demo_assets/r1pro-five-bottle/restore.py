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

"""Restore the committed five-bottle demo assets without downloading them."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assets = Path(__file__).resolve().parent
    manifest = json.loads((assets / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = root / relative
        if (
            path.stat().st_size != expected["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise RuntimeError(f"Asset checksum mismatch: {relative}")
    if args.check_only:
        print("All bundled asset and checkpoint checksums match.")
        return
    scenes = root / "dimos/data/scene_packages"
    if not (scenes / "hssd_102344115").exists():
        scenes.mkdir(parents=True, exist_ok=True)
        with tarfile.open(assets / "scene-package.tar.gz") as archive:
            archive.extractall(scenes, filter="data")
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    robot = cache / "dimos/robot_assets/sources" / manifest["robot_cache_key"] / "URDF"
    if not robot.exists():
        robot.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(robot)], check=True)
        with tarfile.open(assets / "robot-description.tar.gz") as archive:
            archive.extractall(robot, filter="data")
        # Untracked snapshot files keep GitAssetCache from fetching over this
        # portable copy. Existing robot caches are left in place.
        (robot / "DEMO_SNAPSHOT.txt").write_text(
            f"Bundled vendor description at {manifest['robot_revision']}.\n"
        )
    print("Five-bottle policy, apartment and robot assets are ready.")
    print("Run: uv run dimos run r1pro-home-sim")


if __name__ == "__main__":
    main()
