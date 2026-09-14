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

"""The Rust estimator against the C++ Point-LIO's own trajectory on the same walk.

``mid360_athens_stairs`` is a 305 s handheld Mid-360 recording kept as two LFS
artifacts: the raw ``.pcap`` (the input) and a ``.db`` holding what the C++
module published while it was captured (``pointlio_odometry``, the golden).

Point-LIO is chaotic at the ULP level, so this is a band check, not equality --
see ``rust/PARITY.md`` for the C++'s own spread under benign perturbation
(0.17-0.80 m RMSE on this recording). The tight assertion is the first 150 s,
where every run of either implementation still agrees to centimetres; the walk
descends stairs at ~275 s and every run bifurcates there by a metre or more.

Measured: 144.9 m path against the golden's 143.8 m, 0.51 m APE RMSE overall,
34 mm median over the first 150 s.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.memory.store.sqlite import SqliteStore
from dimos.utils.data import get_data

# ~10 s of motion at the 10 Hz frame rate. Measured 64 mm; the p95 is 495 mm and
# the max 1.7 m, both dominated by the branch flip, which is why only the median
# is bounded.
RPE_WINDOW_FRAMES = 100
MAX_RPE_MEDIAN_M = 0.15
EXPECTED_FRAMES = 3041


def _golden() -> tuple[np.ndarray, np.ndarray]:
    store = SqliteStore(path=str(get_data("mid360_athens_stairs.db")), must_exist=True)
    store.start()
    try:
        poses = [
            (o.data.ts, o.data.x, o.data.y, o.data.z) for o in store.stream("pointlio_odometry")
        ]
    finally:
        store.dispose()
    a = np.array(poses)
    return a[:, 0] - a[0, 0], a[:, 1:4]


@pytest.mark.self_hosted
def test_rust_tracks_the_cpp_trajectory(tmp_path: Path) -> None:
    binary = DIMOS_PROJECT_ROOT / "target" / "release" / "pointlio_replay"
    if not binary.exists():
        pytest.fail(f"{binary} missing; run: cargo build --release -p dimos-pointlio")

    out = tmp_path / "trajectory.tum"
    # Frame by frame, so the result does not depend on how loaded the machine is.
    subprocess.run(
        [str(binary), "--pcap", str(get_data("mid360_athens_stairs.pcap")), "--out", str(out)],
        check=True,
        timeout=600,
    )
    rust = np.loadtxt(out)
    assert len(rust) == EXPECTED_FRAMES, f"expected {EXPECTED_FRAMES} frames, got {len(rust)}"
    rt, rp = rust[:, 0] - rust[0, 0], rust[:, 1:4]

    gt, gp = _golden()
    # The golden publishes at IMU rate; take the nearest golden sample per frame.
    hi = np.searchsorted(gt, rt).clip(0, len(gt) - 1)
    lo = (hi - 1).clip(0)
    pick = np.where(np.abs(gt[lo] - rt) < np.abs(gt[hi] - rt), lo, hi)
    matched = gp[pick]

    # Total distance travelled still has to be right; a branch flip barely moves
    # it (144-152 m across every perturbation) but a real break would.
    path = np.linalg.norm(np.diff(rp, axis=0), axis=1).sum()
    golden_path = np.linalg.norm(np.diff(gp, axis=0), axis=1).sum()
    assert abs(path - golden_path) / golden_path < 0.05, f"path {path:.1f} m vs {golden_path:.1f} m"

    k = RPE_WINDOW_FRAMES
    drift = np.linalg.norm((rp[k:] - rp[:-k]) - (matched[k:] - matched[:-k]), axis=1)
    median = float(np.median(drift))
    assert median < MAX_RPE_MEDIAN_M, (
        f"RPE median {median * 1000:.0f} mm over {k}-frame windows; the port no "
        f"longer tracks the C++'s local motion"
    )
