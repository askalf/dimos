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

"""Python 3.9 renderer worker. All EGL calls stay on this process's main thread."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import socket
import sys
import traceback
from typing import Any

import habitat_sim as hs
import magnum as mn
import numpy as np
from PIL import Image
from view_math import ceiling_cutaway, perspective


class Renderer:
    def __init__(self) -> None:
        self.sim = None
        self.scene = None
        self.floor_index = 0
        self.cut_height = 1.8
        self.mode = "topdown"
        self.cutaway = True
        self.orbit_yaw = 35.0
        self.orbit_elevation = 60.0
        self.orbit_zoom = 1.0
        self.pan_x = self.pan_z = 0.0

    def load(self, scene: dict[str, Any]) -> dict[str, Any]:
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        self.scene = None
        cfg = hs.SimulatorConfiguration()
        cfg.scene_dataset_config_file = scene["dataset"]
        cfg.scene_id = scene["scene_id"]
        # ReplicaCAD needs Bullet to instantiate its articulated furniture.
        # We never advance physics: authored placements/joints remain fixed.
        cfg.enable_physics = scene["family"] == "replicacad"
        cfg.random_seed = scene["seed"]
        sensor = hs.CameraSensorSpec()
        sensor.uuid = "rgb"
        sensor.sensor_type = hs.SensorType.COLOR
        sensor.resolution = [1536, 2048]
        sensor.sensor_subtype = hs.SensorSubType.ORTHOGRAPHIC
        sensor.ortho_scale = 1.0
        sensor.position = mn.Vector3(0, 0, 0)
        sensor.clear_color = mn.Color4(0.055, 0.075, 0.105, 1)
        agent = hs.agent.AgentConfiguration()
        agent.sensor_specifications = [sensor]
        agent.action_space = {}
        self.sim = hs.Simulator(hs.Configuration(cfg, [agent]))
        self.agent = self.sim.initialize_agent(0)
        pf = self.sim.pathfinder
        self.nav_note = "Floor elevations detected from supplied navmesh"
        if scene["family"] in {"hssd", "replicacad"} or not pf.is_loaded:
            # The review mesh supplies floor elevations; map bounds use full geometry.
            for manager in [
                self.sim.get_rigid_object_manager(),
                self.sim.get_articulated_object_manager(),
            ]:
                for obj in manager.get_objects_by_handle_substring().values():
                    if obj is not None:
                        obj.motion_type = hs.physics.MotionType.STATIC
            settings = hs.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = 0.25
            settings.agent_height = 0.6
            settings.include_static_objects = True
            rebuilt = self.sim.recompute_navmesh(pf, settings)
            self.nav_note = (
                "Floor elevations detected from an in-memory review navmesh"
                if rebuilt
                else "Floor elevation estimated from scene geometry"
            )
        self.scene = scene["id"]
        pf.seed(scene["seed"])
        bounds = self.sim.get_active_scene_graph().get_root_node().cumulative_bb
        self.low, self.high = np.array(bounds.min, dtype=float), np.array(bounds.max, dtype=float)
        if not np.isfinite([self.low, self.high]).all():
            raise ValueError("Scene has no finite geometry bounds")
        # Match the rendered 4:3 aspect ratio, adding a border on every side.
        self.map_width = (
            max(self.high[0] - self.low[0], (self.high[2] - self.low[2]) * 4 / 3) * 1.08
        )
        self.map_width = max(self.map_width, 1.0)
        self.floors = [float(self.low[1])]
        if pf.is_loaded:
            heights = np.array([pf.get_random_navigable_point()[1] for _ in range(800)])
            heights = heights[np.isfinite(heights)]
            if len(heights):
                bins, counts = np.unique(np.round(heights / 0.25), return_counts=True)
                levels = []
                for i in np.argsort(counts)[::-1]:
                    level = float(np.median(heights[np.round(heights / 0.25) == bins[i]]))
                    if counts[i] >= max(12, counts.max() * 0.04) and all(
                        abs(level - h) > 1.5 for h in levels
                    ):
                        levels.append(level)
                if levels:
                    self.floors = sorted(levels)
        self.floor_index = 0
        self.cut_height = 1.8
        self.mode = "topdown"
        self.cutaway = True
        self.orbit_yaw, self.orbit_elevation, self.orbit_zoom = 35.0, 60.0, 1.0
        self.pan_x = self.pan_z = 0.0
        return self.frame()

    def frame(self) -> dict[str, Any]:
        state = self.agent.get_state()
        floor = self.floors[self.floor_index]
        camera = self.agent._sensors["rgb"]
        if self.mode == "topdown":
            eye_height = max(float(self.high[1]) + 10.0, floor + self.cut_height + 1)
            state.position = np.array(
                [(self.low[0] + self.high[0]) / 2, eye_height, (self.low[2] + self.high[2]) / 2],
                dtype=np.float32,
            )
            state.rotation = hs.utils.common.quat_from_angle_axis(-math.pi / 2, np.array([1, 0, 0]))
            self.agent.set_state(state, reset_sensors=True)
            # Restore the sensor's orthographic matrix after a perspective frame.
            camera.near_plane_dist = eye_height - floor - self.cut_height
            camera.far_plane_dist = eye_height - floor + 0.5
            camera.reset_zoom()
            camera.zoom(1.0 / self.map_width)
        else:
            target = (self.low + self.high) / 2
            target[1] = floor + 0.7
            radius = float(
                np.linalg.norm(np.maximum(np.abs(self.low - target), np.abs(self.high - target)))
            )
            yaw, elevation = math.radians(self.orbit_yaw), math.radians(self.orbit_elevation)
            fov_y = math.radians(48)
            distance = max(0.3, radius * 1.1 / math.sin(fov_y / 2) / self.orbit_zoom)
            if self.cutaway:
                # Oblique near clipping requires the eye to remain above the cut.
                distance = max(distance, (self.cut_height - 0.7 + 0.15) / math.sin(elevation))
            target += np.array([self.pan_x, 0, self.pan_z])
            eye = target + distance * np.array(
                [
                    math.sin(yaw) * math.cos(elevation),
                    math.sin(elevation),
                    math.cos(yaw) * math.cos(elevation),
                ]
            )
            state.position = eye.astype(np.float32)
            state.rotation = hs.utils.common.quat_from_angle_axis(
                yaw, np.array([0, 1, 0])
            ) * hs.utils.common.quat_from_angle_axis(-elevation, np.array([1, 0, 0]))
            self.agent.set_state(state, reset_sensors=True)
            projection = perspective(fov_y, 4 / 3, 0.05, distance + radius * 5 + self.map_width)
            if self.cutaway:
                view = np.array(camera.render_camera.camera_matrix, dtype=np.float64)
                projection = ceiling_cutaway(projection, view, floor + self.cut_height)
            camera.render_camera.projection_matrix = mn.Matrix4(projection)
        rgb = self.sim.get_sensor_observations()["rgb"][:, :, :3]
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, "JPEG", quality=94 if self.mode == "topdown" else 85)
        return {"jpeg": base64.b64encode(output.getvalue()).decode(), "state": self.state()}

    def state(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "mode": self.mode,
            "cutaway": self.cutaway,
            "orbit_yaw": self.orbit_yaw,
            "orbit_elevation": self.orbit_elevation,
            "orbit_zoom": self.orbit_zoom,
            "pan_x": self.pan_x,
            "pan_z": self.pan_z,
            "floor_index": self.floor_index,
            "floors": self.floors,
            "cut_height": self.cut_height,
            "map_width_m": self.map_width,
            "map_height_m": self.map_width * 3 / 4,
            "bounds_min": self.low.tolist(),
            "bounds_max": self.high.tolist(),
            "nav_note": self.nav_note,
        }

    def control(self, cmd: dict[str, Any]) -> dict[str, Any]:
        if self.scene != cmd["scene"]:
            raise ValueError("Scene changed; reload the scene view")
        index = cmd.get("floor", self.floor_index)
        if not 0 <= index < len(self.floors):
            raise ValueError("Unknown floor")
        self.floor_index = index
        self.cut_height = cmd.get("cut_height", self.cut_height)
        self.mode = cmd.get("mode", self.mode)
        self.cutaway = cmd.get("cutaway", self.cutaway)
        self.orbit_yaw = cmd.get("orbit_yaw", self.orbit_yaw)
        self.orbit_elevation = cmd.get("orbit_elevation", self.orbit_elevation)
        self.orbit_zoom = cmd.get("orbit_zoom", self.orbit_zoom)
        self.pan_x = cmd.get("pan_x", self.pan_x)
        self.pan_z = cmd.get("pan_z", self.pan_z)
        return self.frame()


def main() -> None:
    connection = socket.socket(fileno=int(sys.argv[1]))
    stream = connection.makefile("rwb")
    renderer = Renderer()
    try:
        for line in stream:
            try:
                cmd = json.loads(line)
                result = (
                    renderer.load(cmd["scene"])
                    if cmd["action"] == "load"
                    else renderer.control(cmd)
                )
                result["ok"] = True
            except Exception as exc:
                traceback.print_exc()
                result = {"ok": False, "error": str(exc)}
            stream.write(json.dumps(result).encode() + b"\n")
            stream.flush()
    finally:
        if renderer.sim is not None:
            renderer.sim.close()


if __name__ == "__main__":
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    main()
