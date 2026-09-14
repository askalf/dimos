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

"""Separate GraspGenX/classical apartment blueprint with no ACT policy modules."""

from dataclasses import replace
from pathlib import Path

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.constants import RECORDINGS_DIR
from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.manipulation.grasping.grasp_gen_x import GraspGenXModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.planning.planners.config import RRTConnectPlannerConfig
from dimos.navigation.nav_3d.mls_planner.mls_planner_native import MLSPlannerNative
from dimos.robot.assets.model import PlanarBaseDefinition
from dimos.robot.galaxea.r1pro.apartment_coordinator import R1ProApartmentCoordinator
from dimos.robot.galaxea.r1pro.apartment_navigation import (
    APARTMENT_FRAME,
    APARTMENT_NAV_TASK,
    ApartmentNavigation,
)
from dimos.robot.galaxea.r1pro.apartment_route import CLASSICAL_NAVIGATION_CLEARANCE_M
from dimos.robot.galaxea.r1pro.classical_gripper import (
    R1PRO_GRASP_FRAME_TO_TCP,
    R1PRO_GRIPPER_SWEEP,
)
from dimos.robot.galaxea.r1pro.classical_sim import R1ProClassicalSim
from dimos.robot.galaxea.r1pro.classical_skills import R1ProClassicalSkills
from dimos.robot.galaxea.r1pro.config import (
    R1PRO_MODEL,
    R1PRO_PLANAR_BASE,
    make_r1pro_planar_model_config,
)
from dimos.robot.galaxea.r1pro.grasping_blueprint import build_r1pro_manipulation
from dimos.robot.galaxea.r1pro.learning import R1PRO_PICK_PLACE_JOINTS
from dimos.robot.galaxea.r1pro.object_primitives import active_indices

CLASSICAL_BASE_ID = "r1pro_classical_base"
CLASSICAL_BASE_TASK = "base_trajectory"
CONTROLLER_ARTIFACT = Path(__file__).with_name("classical_navigation_controller.json")
CLASSICAL_PROMPT = """You control a simulated R1Pro in an apartment using classical DimOS manipulation and GraspGenX.
Start idle. Read get_scene before each request. Object IDs, kind, color and robot-relative coordinates
come from the current simulation. Never substitute another object or hand.
Pass the user's object description INCLUDING color, kind and left/right qualifiers to pick_object.
Do not resolve 'blue carton on the left' into just 'carton' or an ID yourself: the skill checks all attributes.
The arm argument describes the requested hand, separately from the object's location.
If no hand is specified, use arm=auto. If an item is ambiguous or absent, ask for clarification.
pick_object means grasp, lift and HOLD. Never add placement to a pick request.
place_object releases only an already held object at the explicitly requested support region.
go_to navigates while preserving all held items; it does not release them.
Call wait_for_action until an accepted action completes before starting the next action.
If an action fails, report its phase and error; recover_action once if recovery_required.
Never reset the scene or silently retry a failed physical grasp. Reset only on explicit request.
GraspGenX proposes learned grasps; DimOS classical planning executes picks and placements.
No ACT policy is running. Keep answers brief."""


def build_classical_apartment(*, agent: bool = False) -> Blueprint:
    base_joints = make_twist_base_joints(CLASSICAL_BASE_ID)
    source = build_r1pro_manipulation(
        scene_path=RECORDINGS_DIR / "r1pro-classical/scene.xml",
        artifact="unused",
        device="cuda",
        headless=False,
        simulator=R1ProClassicalSim,
        policy_module=None,
        task_description="",
        prepare_scene_on_build=True,
        background_camera_rendering=True,
        coordinator_type=R1ProApartmentCoordinator,
        simulator_options={
            "workspace_file": None,
            "policy_neighbor_distance": None,
            "navigation_clearance_m": CLASSICAL_NAVIGATION_CLEARANCE_M,
        },
        velocity_base=HardwareComponent(
            hardware_id=CLASSICAL_BASE_ID,
            hardware_type=HardwareType.BASE,
            joints=base_joints,
            adapter_type="transport_lcm",
            auto_enable=True,
        ),
        navigation_task=TaskConfig(
            name=CLASSICAL_BASE_TASK,
            type="base_trajectory",
            joint_names=base_joints,
            priority=30,
            params={
                "max_linear": 0.3,
                "max_angular": 0.4,
                "goal_tolerance": 0.005,
                "orientation_tolerance": 0.005,
                "settle_timeout": 25.0,
                "stop_hold_s": 0.5,
            },
        ),
        viewer_lookat=(0.35, 0, 0.8),
        viewer_distance=2.4,
        viewer_azimuth=145,
        viewer_elevation=-35,
    )
    atoms = []
    for atom in source.blueprints:
        kwargs = dict(atom.kwargs)
        if atom.module is R1ProClassicalSim:
            kwargs.update(width=320, height=240, fps=10, extra_cameras=[])
        if atom.module is R1ProApartmentCoordinator:
            tasks = [
                replace(task, name="joint_trajectory") if task.name == "tray_manipulation" else task
                for task in kwargs["tasks"]
                if task.name != "policy_rollout"
            ]
            for side in ("left", "right"):
                tasks.append(
                    TaskConfig(
                        name=f"primitive_{side}",
                        type="trajectory",
                        priority=30,
                        joint_names=[R1PRO_PICK_PLACE_JOINTS[i] for i in active_indices(side)],
                    )
                )
            tasks.append(
                TaskConfig(
                    name=APARTMENT_NAV_TASK,
                    type="holonomic_pose_follower",
                    joint_names=base_joints,
                    priority=30,
                    params={
                        "artifact_path": str(CONTROLLER_ARTIFACT),
                        "speed": 0.3,
                        "lookahead": 0.025,
                        "regulate_horizon": 0.2,
                        "goal_tolerance": 0.003,
                        "orientation_tolerance": 0.003,
                        "approach_decel": 0.2,
                        "stop_hold_s": 0.5,
                    },
                )
            )
            kwargs["tasks"] = tasks
        atoms.append(replace(atom, kwargs=kwargs))
    model = make_r1pro_planar_model_config()
    base = PlanarBaseDefinition(
        root_link=R1PRO_PLANAR_BASE.root_link,
        joint_names=R1PRO_PLANAR_BASE.joint_names,
        velocity_limits=(0.3, 0.3, 0.4),
        acceleration_limits=(0.2, 0.2, 0.4),
    )
    model.model = R1PRO_MODEL.with_planar_base(base)
    for side in ("left", "right"):
        model.model = model.model.with_fixed_frame(
            f"{side}_tcp", f"{side}_gripper_link", xyz=(0, 0, -0.085)
        )
    model.planning_groups = [
        replace(group, tip_link=group.name.replace("_arm", "_tcp"))
        if group.name.endswith("_arm")
        else group
        for group in model.planning_groups
    ]
    modules = [
        replace(source, blueprints=tuple(atoms)),
        R1ProClassicalSkills.blueprint(),
        GraspGenXModule.blueprint(
            gripper=R1PRO_GRIPPER_SWEEP,
            grasp_frame_to_tcp=R1PRO_GRASP_FRAME_TO_TCP,
            num_samples=600,
            max_candidates=600,
        ),
        ManipulationModule.blueprint(
            model=model,
            planner=RRTConnectPlannerConfig(),
            joint_state_aliases=dict(zip(base_joints, R1PRO_PLANAR_BASE.joint_names, strict=True)),
            base_trajectory_task=CLASSICAL_BASE_TASK,
            visualization={"backend": "none"},
        ),
        McpServer.blueprint(),
        ApartmentNavigation.blueprint(),
        MLSPlannerNative.blueprint(
            world_frame="world",
            base_frame=APARTMENT_FRAME,
            voxel_size=0.06,
            robot_height=1.70,
            surface_closing_radius=0.12,
            node_spacing_m=0.3,
            wall_clearance_m=0.35,
            wall_buffer_m=0.65,
            wall_buffer_weight=20.0,
            step_threshold_m=0.07,
            goal_tolerance=0.01,
            viz_publish_hz=1.0,
            worker_threads=2,
        ),
    ]
    if agent:
        modules.append(McpClient.blueprint(system_prompt=CLASSICAL_PROMPT))
    return (
        autoconnect(*modules)
        .remappings(
            [
                (R1ProClassicalSim, "base_cmd_vel", f"/{CLASSICAL_BASE_ID}/cmd_vel"),
                (R1ProClassicalSim, "base_odom", f"/{CLASSICAL_BASE_ID}/odom"),
                (MLSPlannerNative, "path", "planned_path"),
                (MLSPlannerNative, "tf", "navigation_tf"),
                (ApartmentNavigation, "base_odom", f"/{CLASSICAL_BASE_ID}/odom"),
                (R1ProApartmentCoordinator, "path", "execution_path"),
            ]
        )
        .global_config(transport="zenoh", viewer="none", simulation="mujoco", n_workers=7)
    )


r1pro_classical_apartment_sim = build_classical_apartment().global_config()
r1pro_classical_apartment_sim_agent = build_classical_apartment(agent=True).global_config()
