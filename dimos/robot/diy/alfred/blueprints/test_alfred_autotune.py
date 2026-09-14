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

"""Alfred autotune: the world-pose projection, and the whole path end to end.

The autotune package's own test covers the fit/tune/emit math against synthetic
segments. What is Alfred-specific, and what these cover, is the step between a
Point-LIO world pose and the scalar those fitters read.
"""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dimos.control.autotune.excitation import step_battery
from dimos.control.autotune.fit.synth import MeasurementModel, synth_step
from dimos.control.autotune.runner import autotune_offline
from dimos.control.benchmarking.gate import GATE_ADVANCE, GATE_QUIT, GATE_SKIP
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.diy.alfred.blueprints.alfred_autotune import (
    alfred_autotune,
    alfred_autotune_profile,
    project_to_axis,
)

ODOM_HZ = 50.0


def _samples(xs, ys, yaws, *, rate_hz=ODOM_HZ, t0=1000.0):
    """World pose samples in the driver's capture shape."""
    return [
        (t0 + i / rate_hz, float(x), float(y), float(yaw))
        for i, (x, y, yaw) in enumerate(zip(xs, ys, yaws, strict=True))
    ]


def test_forward_travel_is_measured_in_the_heading_the_run_started_in() -> None:
    # Facing +y in the world. A vx step moves the robot along world +y, and that
    # must read as positive FORWARD travel, not zero.
    yaw0 = math.pi / 2
    ys = np.linspace(0.0, 2.0, 40)
    samples = _samples(np.zeros(40), ys, np.full(40, yaw0))

    t, measured = project_to_axis(samples, "vx")

    assert measured[0] == pytest.approx(0.0, abs=1e-9)
    assert measured[-1] == pytest.approx(2.0, abs=1e-9)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(39 / ODOM_HZ)


def test_strafe_reads_on_vy_and_not_on_vx() -> None:
    # Facing +x, moving along world +y is pure lateral travel.
    ys = np.linspace(0.0, 1.5, 30)
    samples = _samples(np.zeros(30), ys, np.zeros(30))

    _, lateral = project_to_axis(samples, "vy")
    _, forward = project_to_axis(samples, "vx")

    assert lateral[-1] == pytest.approx(1.5, abs=1e-9)
    assert forward[-1] == pytest.approx(0.0, abs=1e-9), "a strafe leaked into the forward axis"


def test_yaw_crossing_pi_is_not_a_two_pi_jump() -> None:
    # A run that crosses the wrap would otherwise fit as an enormous gain.
    yaws = np.linspace(math.pi - 0.2, math.pi + 0.3, 25)
    wrapped = np.arctan2(np.sin(yaws), np.cos(yaws))  # what an estimator reports
    samples = _samples(np.zeros(25), np.zeros(25), wrapped)

    _, measured = project_to_axis(samples, "wz")

    assert measured[-1] == pytest.approx(0.5, abs=1e-6)
    assert np.all(np.diff(measured) > 0), "unwrap left a discontinuity"


def test_unknown_channel_is_refused() -> None:
    samples = _samples(np.zeros(5), np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="unknown channel"):
        project_to_axis(samples, "vz")


def _world_run(displacement: np.ndarray, yaw0: float, channel: str):
    """Put an axis displacement into the world at a non-trivial heading."""
    if channel == "wz":
        return np.zeros_like(displacement), np.zeros_like(displacement), yaw0 + displacement
    cos0, sin0 = math.cos(yaw0), math.sin(yaw0)
    if channel == "vx":
        return displacement * cos0, displacement * sin0, np.full_like(displacement, yaw0)
    return -displacement * sin0, displacement * cos0, np.full_like(displacement, yaw0)


def test_a_battery_of_world_poses_recovers_the_plant_it_was_generated_from() -> None:
    """End to end: world pose in, Alfred's artifact out.

    The driver captures world pose and hands projected segments straight to
    autotune_offline. This drives that whole path on a known plant so a
    projection error shows up as a wrong gain rather than a plausible one.
    """
    K_true, tau_true, L_true = 0.92, 0.45, 0.08
    yaw0 = 0.7  # deliberately not axis-aligned
    profile = alfred_autotune_profile()

    segments: dict[str, list] = {}
    for channel in profile.channel_names:
        vmax = profile.channel(channel).vmax
        per_channel = []
        for fraction in profile.battery.amplitude_fractions:
            for direction in (1, -1):
                amp = direction * fraction * vmax
                _, _, pose = synth_step(
                    K_true,
                    tau_true,
                    L_true,
                    amp=amp,
                    duration_s=4.0,
                    model=MeasurementModel(rate_hz=ODOM_HZ, noise_std=0.0),
                )
                xs, ys, yaws = _world_run(pose, yaw0, channel)
                t, measured = project_to_axis(_samples(xs, ys, yaws), channel)
                per_channel.append((t, measured, amp))
        segments[channel] = per_channel

    outputs = autotune_offline(profile, segments, robot_id="alfred", sim_or_hw="sim")

    for channel in profile.channel_names:
        fopdt = outputs.profile.measured.fopdt[channel]
        assert fopdt.K == pytest.approx(K_true, rel=0.15), f"{channel} gain came back wrong"
        assert fopdt.tau == pytest.approx(tau_true, rel=0.35), f"{channel} tau came back wrong"

    # The artifact indexes these exact slots; the follower reads them by name.
    assert set(outputs.artifact["plant"]) == {"vx", "vy", "wz"}
    # Synthetic data must never be deployable, whatever the fit quality.
    assert outputs.artifact["valid_for_tuning"] is False


def test_blueprint_composes_with_pose_feedback_reaching_the_driver() -> None:
    names = {a.instance_name or a.module.__name__ for a in alfred_autotune.blueprints}
    assert {"AlfredHighLevel", "PointLio", "StartRelay", "AlfredAutotuneDriver"} <= names

    (driver,) = [
        a for a in alfred_autotune.blueprints if a.module.__name__ == "AlfredAutotuneDriver"
    ]
    streams = {s.name: s.direction for s in driver.streams}
    # Without start_pose reaching the driver there is nothing to fit.
    assert streams.get("start_pose") == "in"
    assert streams.get("cmd_vel") == "out"


def _driver():
    from dimos.robot.diy.alfred.blueprints.alfred_autotune import (
        AlfredAutotuneDriver,
        AlfredAutotuneDriverConfig,
    )

    d = AlfredAutotuneDriver.__new__(AlfredAutotuneDriver)
    d.config = AlfredAutotuneDriverConfig()
    d._gate = threading.Event()
    d._gate_action = GATE_ADVANCE
    d._relaying = False
    d._stop = threading.Event()
    d._published = []
    d.cmd_vel = SimpleNamespace(publish=d._published.append)
    return d


def test_the_gate_waits_and_a_click_advances_it() -> None:
    """Nothing moves until the operator says so - there is no arm flag, the gate is it."""
    from dimos.control.autotune.excitation import step_battery
    from dimos.robot.diy.alfred.blueprints.alfred_autotune import AlfredAutotuneDriverConfig

    # 0 means wait forever: a battery that marched on by itself would be driving
    # an unattended robot.
    assert AlfredAutotuneDriverConfig().gate_timeout_s == 0.0

    d = _driver()
    run = step_battery(alfred_autotune_profile(), duration_s=1.0)[0]

    released = []
    threading.Timer(0.05, lambda: (d._on_click(None), released.append(True))).start()
    assert d._wait_for_gate(run, 1, 1) == GATE_ADVANCE
    assert released, "the gate returned before anything advanced it"


def test_skip_and_abort_carry_their_own_verdicts() -> None:
    d = _driver()
    run = step_battery(alfred_autotune_profile(), duration_s=1.0)[0]

    threading.Timer(0.05, d.skip).start()
    assert d._wait_for_gate(run, 1, 1) == GATE_SKIP

    threading.Timer(0.05, d.abort).start()
    assert d._wait_for_gate(run, 1, 1) == GATE_QUIT


def test_teleop_drives_the_base_only_while_the_gate_is_open() -> None:
    """A stray key during an excitation would corrupt the step being identified."""
    d = _driver()
    twist = Twist(linear=Vector3(0.2, 0.0, 0.0), angular=Vector3())

    d._relaying = False  # a run is playing
    d._on_teleop(twist)
    assert d._published == [], "teleop reached the base mid-run"

    d._relaying = True  # waiting at the gate
    d._on_teleop(twist)
    assert d._published == [twist]


def test_the_gate_leaves_the_base_at_rest_before_a_run() -> None:
    """Whatever the operator left on the stick, the excitation starts from zero."""
    d = _driver()
    run = step_battery(alfred_autotune_profile(), duration_s=1.0)[0]

    threading.Timer(0.05, d.advance).start()
    d._wait_for_gate(run, 1, 1)

    assert d._relaying is False
    assert d._published and d._published[-1].linear.x == 0.0


def test_the_viewer_supplies_both_the_gate_and_the_hands() -> None:
    """Headless over ssh: the click and the driving both come from the viewer."""
    names = {a.instance_name or a.module.__name__ for a in alfred_autotune.blueprints}
    assert "RerunWebSocketServer" in names, "no viewer, so no gate and no way to reposition"

    (driver,) = [
        a for a in alfred_autotune.blueprints if a.module.__name__ == "AlfredAutotuneDriver"
    ]
    streams = {s.name: s.direction for s in driver.streams}
    assert streams.get("clicked_point") == "in"
    assert streams.get("tele_cmd_vel") == "in"
