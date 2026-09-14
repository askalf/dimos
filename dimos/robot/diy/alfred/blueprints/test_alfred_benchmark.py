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

"""What has to hold for a benchmark score to mean anything."""

from __future__ import annotations

from dimos.control.benchmarking.benchmark import Benchmarker
from dimos.robot.diy.alfred.blueprints.alfred_benchmark import (
    ALFRED_BASE_HARDWARE_ID,
    NAV_FOLLOWER_TASK_NAME,
    alfred_benchmark,
    alfred_holonomic_controller,
)


def _atoms(bp, module):
    return [a for a in bp.blueprints if a.module is module]


def _streams(bp, name):
    (atom,) = [a for a in bp.blueprints if a.module.__name__ == name]
    return {s.name: s.direction for s in atom.streams}


def test_the_controller_has_no_second_thing_steering() -> None:
    """A planner publishing path alongside the Benchmarker would fight it."""
    names = {a.module.__name__ for a in alfred_holonomic_controller.blueprints}
    assert "MLSPlannerNative" not in names
    assert "DanLocalPlanner" not in names
    assert "ManipulationModule" not in names
    # But it does need the pose it closes the loop on.
    assert {"PointLio", "StartRelay", "AlfredHighLevel"} <= names


def test_the_gate_reaches_the_benchmarker_from_the_viewer() -> None:
    """Headless: a click stands in for the Go2's pygame ENTER."""
    assert _streams(alfred_benchmark, "RerunWebSocketServer")["clicked_point"] == "out"
    gate = _streams(alfred_benchmark, "AlfredViewerGate")
    assert gate["clicked_point"] == "in"
    assert gate["operator_command"] == "out"
    assert _streams(alfred_benchmark, "Benchmarker")["operator_command"] == "in"


def test_the_benchmarker_scores_the_command_the_robot_received() -> None:
    """Scoring the operator's teleop nudges instead would be meaningless."""
    topics = {name: spec.args[0] for (name, _t), spec in alfred_benchmark.transport_map.items()}
    assert topics["cmd_vel"] == f"/{ALFRED_BASE_HARDWARE_ID}/cmd_vel"
    # The Benchmarker calls the robot's pose `odom`; StartRelay publishes start_pose.
    assert alfred_benchmark.remapping_map[("benchmarker", "odom")] == "start_pose"


def test_the_follower_is_the_only_thing_claiming_the_base_below_teleop() -> None:
    (coordinator,) = [
        a for a in alfred_benchmark.blueprints if a.instance_name == "ControlCoordinator"
    ]
    tasks = coordinator.kwargs["tasks"]
    followers = [t for t in tasks if t.type == "holonomic_pose_follower"]
    assert len(followers) == 1, "two followers in one coordinator would fight over the base"
    (follower,) = followers
    assert follower.name == NAV_FOLLOWER_TASK_NAME
    (teleop,) = [t for t in tasks if t.type == "velocity"]
    assert teleop.priority > follower.priority, "teleop must preempt the follower"


def test_the_battery_is_paced_by_the_operator() -> None:
    """auto would march an unattended robot through the whole battery."""
    (bench,) = _atoms(alfred_benchmark, Benchmarker)
    assert bench.kwargs["gate_source"] == "stream"
    assert bench.kwargs["robot"] == "alfred"


def test_the_operator_can_actually_reposition_between_runs() -> None:
    """The gate tells them to drive the base; the wiring has to let them.

    The viewer publishes tele_cmd_vel and the coordinator hears twist_command.
    Nothing joins those by name, so without the remap the instruction is a lie.
    """
    assert alfred_benchmark.remapping_map[("ControlCoordinator", "twist_command")] == "tele_cmd_vel"
    assert _streams(alfred_benchmark, "RerunWebSocketServer")["tele_cmd_vel"] == "out"


def test_the_speed_ladder_stays_inside_alfreds_envelope() -> None:
    """The Benchmarker's default ladder is the Go2's and tops out past Alfred's vmax."""
    from dimos.robot.diy.alfred.alfred_model import ALFRED_BASE_VELOCITY_LIMITS
    from dimos.robot.diy.alfred.blueprints.alfred_benchmark import ALFRED_BENCHMARK_SPEEDS

    vx_max = ALFRED_BASE_VELOCITY_LIMITS[0]
    speeds = [float(s) for s in ALFRED_BENCHMARK_SPEEDS.split(",")]
    assert speeds, "an empty ladder benchmarks nothing"
    assert max(speeds) < vx_max, "a run past vmax scores saturation, not tracking"
    assert speeds == sorted(speeds), "the ladder should climb"

    (bench,) = _atoms(alfred_benchmark, Benchmarker)
    assert bench.kwargs["speeds"] == ALFRED_BENCHMARK_SPEEDS


def test_nothing_arbitrates_the_base_but_the_coordinator() -> None:
    """No mux upstream of the hardware - the whole point of the rework."""
    names = {a.module.__name__ for a in alfred_benchmark.blueprints}
    assert "MovementManager" not in names
    assert "DanHolonomicTC" not in names
