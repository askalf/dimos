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

"""The Rust estimator against a recorded Point-LIO trajectory over the same walk.

``mid360_athens_stairs`` is a 305 s Mid-360 recording kept as two LFS artifacts:
the ``.pcap`` input, and a ``.db`` whose ``pointlio_odometry`` is the reference.
go2web captured it with its Point-LIO Lite sidecar, tuned identically to
``PointLioTuning`` but built for aarch64 and run in realtime.

Scored on relative pose error, because absolute error here is a property of the
recording rather than the port: on this walk's stairs descent one input float
changed by one ULP moves the final position anywhere in a 3.7 m band, in the C++
as much as the Rust. RPE median stays within centimetres across those branches.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.memory.store.sqlite import SqliteStore
from dimos.utils.data import get_data

# ~10 s of motion at 10 Hz. Measured 64 mm median; p95 and max are dominated by
# the branch flip, so only the median is bounded.
RPE_WINDOW_FRAMES = 100
MAX_RPE_MEDIAN_M = 0.15
EXPECTED_FRAMES = 3041


def _reference() -> tuple[np.ndarray, np.ndarray]:
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
def test_rust_tracks_the_reference_trajectory(tmp_path: Path) -> None:
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
    estimate = np.loadtxt(out)
    assert len(estimate) == EXPECTED_FRAMES, (
        f"expected {EXPECTED_FRAMES} frames, got {len(estimate)}"
    )
    et, ep = estimate[:, 0] - estimate[0, 0], estimate[:, 1:4]

    ref_t, ref_p = _reference()
    # The reference publishes at IMU rate; take its nearest sample per frame.
    hi = np.searchsorted(ref_t, et).clip(0, len(ref_t) - 1)
    lo = (hi - 1).clip(0)
    pick = np.where(np.abs(ref_t[lo] - et) < np.abs(ref_t[hi] - et), lo, hi)
    matched = ref_p[pick]

    # A branch flip barely moves the distance travelled; a real break would.
    path = np.linalg.norm(np.diff(ep, axis=0), axis=1).sum()
    ref_path = np.linalg.norm(np.diff(ref_p, axis=0), axis=1).sum()
    assert abs(path - ref_path) / ref_path < 0.05, f"path {path:.1f} m vs {ref_path:.1f} m"

    k = RPE_WINDOW_FRAMES
    drift = np.linalg.norm((ep[k:] - ep[:-k]) - (matched[k:] - matched[:-k]), axis=1)
    median = float(np.median(drift))
    assert median < MAX_RPE_MEDIAN_M, (
        f"RPE median {median * 1000:.0f} mm over {k}-frame windows; the port no "
        f"longer tracks the reference's local motion"
    )
