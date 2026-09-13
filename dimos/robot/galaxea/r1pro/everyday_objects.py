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

"""Seeded everyday-object variants within the existing ACT grasp envelope."""

from dataclasses import replace

import numpy as np

from dimos.robot.galaxea.r1pro.object_packing import ObjectShape
from dimos.robot.galaxea.r1pro.object_packing_scene import ObjectLayout, sample_layout


def sample_everyday_layout(seed: int, *, count: int = 5, occupied: int = 0) -> ObjectLayout:
    """Vary identity, dimensions, color and pose while retaining separated physical supports.

    This first catalog includes a cup, drink carton, glue stick, toy block and
    bottle. It deliberately retains the trained grasp envelope; success on these
    different appearances/contact surfaces must still be measured with ACT.
    """
    layout = sample_layout(seed, count=count, occupied=occupied)
    rng = np.random.default_rng(seed + 419)
    kinds = rng.permutation(["cup", "drink_carton", "glue_stick", "toy_block", "bottle"])
    objects = []
    shape: ObjectShape
    for obj, kind in zip(layout.objects, kinds, strict=False):
        radius = obj.radius
        if kind in ("drink_carton", "toy_block"):
            shape = "box"
            half = (radius / np.sqrt(2), radius / np.sqrt(2), obj.half_size[2])
        else:
            shape = "bottle" if kind == "bottle" else "cylinder"
            half = (radius, radius, obj.half_size[2])
        objects.append(
            replace(
                obj,
                shape=shape,
                half_size=(float(half[0]), float(half[1]), float(half[2])),
                kind=str(kind),
            )
        )
    return ObjectLayout(seed, tuple(objects))
