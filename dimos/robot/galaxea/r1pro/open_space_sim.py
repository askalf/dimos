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

"""Classical interactive manipulation on widely separated named platforms."""

from pathlib import Path

from dimos.robot.galaxea.r1pro.apartment_sim import R1ProApartmentSimConfig
from dimos.robot.galaxea.r1pro.classical_sim import R1ProClassicalSim
from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout
from dimos.robot.galaxea.r1pro.open_space_scene import OPEN_PLATFORMS, prepare_open_space_scene


class R1ProOpenSpaceSimConfig(R1ProApartmentSimConfig):
    scene_package: Path | None = None
    randomize_locations: bool = False
    occupied: int = 0


class R1ProOpenSpaceSim(R1ProClassicalSim):
    config: R1ProOpenSpaceSimConfig

    def _prepare_scene(
        self, output: Path, layout: ObjectLayout, package: Path | None
    ) -> tuple[Path, ObjectLayout]:
        scene, layout, self._regions = prepare_open_space_scene(output, layout)
        return scene, layout

    def _navigation_heading(self, region: str) -> float:
        if region not in {platform.name for platform in OPEN_PLATFORMS}:
            raise ValueError(f"Unknown open-space platform: {region}")
        return 0.0
