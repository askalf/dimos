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

"""A verified tilted hold can be maintained, but not tipped farther during planning."""

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.classical_planning import preserves_cargo_tilt


@pytest.mark.parametrize(
    ("initial", "current", "expected"),
    [(8, 8, True), (8, 14, False), (14, 16, False), (0, 4, True), (0, 8, False)],
)
def test_carry_gate_preserves_existing_tilt_and_rejects_new_tipping(initial, current, expected):
    assert (
        preserves_cargo_tilt(np.cos(np.deg2rad(initial)), np.cos(np.deg2rad(current))) is expected
    )
