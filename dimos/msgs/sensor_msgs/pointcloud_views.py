# Copyright 2025-2026 Dimensional Inc.
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

"""Deterministic native-axis projections of finite point-cloud returns."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


def _label(value: float, span: float) -> str:
    scale = max(abs(value), abs(span), np.finfo(float).tiny)
    digits = max(
        5, min(17, math.ceil(math.log10(scale)) - math.floor(math.log10(span or scale)) + 4)
    )
    return f"{value:.{digits}g}"


def add_pointcloud_views(
    summary: dict[str, Any], points: NDArray[np.float64], max_bytes: int
) -> dict[str, Any]:
    """Add one calibrated PNG, reducing pixel resolution to fit the whole payload."""
    result = {**summary, "views": []}
    if not len(points):
        if len(json.dumps(result)) > max_bytes:
            raise ValueError("Point cloud metadata exceeds the encoder byte budget")
        return result
    lower, upper = points.min(axis=0), points.max(axis=0)
    spans = upper - lower
    for size in (256, 192, 128, 96, 64, 32):
        png, pixel_sizes = _projection_png(points, lower, spans, size)
        result["views"] = [
            {
                "planes": ["xy", "xz", "yz"],
                "pixel_m": pixel_sizes,
                "coordinate_min_m": lower.tolist(),
                "coordinate_max_m": upper.tolist(),
                "png_base64": base64.b64encode(png).decode("ascii"),
            }
        ]
        if len(json.dumps(result)) <= max_bytes:
            return result
    raise ValueError("Point cloud projections exceed the encoder byte budget")


def _projection_png(
    points: NDArray[np.float64],
    lower: NDArray[np.float64],
    spans: NDArray[np.float64],
    size: int,
) -> tuple[bytes, list[float]]:
    """Every projected pixel retains the highest third-axis coordinate bin."""
    # A fixed palette and bundled bitmap font avoid machine-dependent themes/fonts.
    colors = [
        (35, 55, 180),
        (25, 110, 200),
        (0, 160, 165),
        (40, 185, 105),
        (130, 200, 50),
        (215, 205, 35),
        (245, 145, 30),
        (220, 55, 40),
    ]
    palette = [255, 255, 255, 20, 20, 20, 225, 225, 225]
    palette += [component for color in colors for component in color]
    palette += [0] * (768 - len(palette))
    margin_x, margin_y = 48, 30
    panel_width, panel_height = size + 72, size + 100
    canvas = Image.new("P", (3 * panel_width, panel_height), color=0)
    canvas.putpalette(palette)
    draw = ImageDraw.Draw(canvas)
    pixel_sizes = []
    for panel, (horizontal, vertical, third) in enumerate(((0, 1, 2), (0, 2, 1), (1, 2, 0))):
        extent = float(max(spans[horizontal], spans[vertical]))
        # Degenerate projections still show their occupied pixel without inventing extent.
        denominator = extent if extent else 1.0
        pixel_sizes.append(extent / (size - 1))
        x = np.rint((points[:, horizontal] - lower[horizontal]) / denominator * (size - 1)).astype(
            int
        )
        y = np.rint((points[:, vertical] - lower[vertical]) / denominator * (size - 1)).astype(int)
        value = np.zeros(len(points), dtype=np.int64)
        if spans[third]:
            value = np.minimum(
                7,
                np.floor((points[:, third] - lower[third]) / spans[third] * 8).astype(np.int64),
            )
        pixels = np.zeros((size, size), dtype=np.uint8)
        np.maximum.at(pixels, (size - 1 - y, x), (value + 3).astype(np.uint8))
        tile = Image.fromarray(pixels, mode="P")
        tile.putpalette(palette)
        left, top = panel * panel_width + margin_x, margin_y
        canvas.paste(tile, (left, top))
        xname, yname, zname = "xyz"[horizontal], "xyz"[vertical], "xyz"[third]
        draw.text((left, 6), f"{xname.upper()}{yname.upper()} / native meters", fill=1)
        draw.line((left - 1, top, left - 1, top + size, left + size, top + size), fill=1)
        draw.text(
            (left, top + size + 4),
            f"{xname}: {_label(lower[horizontal], extent)} to {_label(lower[horizontal] + extent, extent)}",
            fill=1,
        )
        draw.text(
            (left, top + size + 18),
            f"{yname}: {_label(lower[vertical], extent)} to {_label(lower[vertical] + extent, extent)} (up)",
            fill=1,
        )
        draw.text(
            (left, top + size + 34),
            f"max {zname}: {_label(lower[third], spans[third])} to {_label(lower[third] + spans[third], spans[third])}",
            fill=1,
        )
        for index in range(8):
            start = left + index * max(1, size // 8)
            draw.rectangle(
                (start, top + size + 50, start + max(1, size // 8) - 1, top + size + 57),
                fill=index + 3,
            )
    stream = BytesIO()
    canvas.save(stream, format="PNG", optimize=False, compress_level=9, bits=4)
    return stream.getvalue(), pixel_sizes
