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

"""Guard against contaminating placement fine-tuning with incompatible or repeated data."""

import json

import pytest

from dimos.robot.galaxea.r1pro.demo_refine_primitives import merge_demonstrations
from dimos.robot.galaxea.r1pro.object_packing_scene import sample_layout
from dimos.robot.galaxea.r1pro.primitive_scene import bilateral_layout


@pytest.fixture
def collections(tmp_path):
    folders = [tmp_path / "original", tmp_path / "corrections"]
    for source, first_seed in zip(folders, (100, 200), strict=True):
        source.mkdir()
        manifest = dict(
            profile="place-right",
            arm="right",
            primitive="place",
            fps=20,
            joints=["joint1"],
            images=True,
            episodes=[
                dict(
                    seed=seed,
                    selected=0,
                    success=True,
                    file=f"{seed}.npz",
                    initial_state=f"{seed}-initial.npz",
                )
                for seed in range(first_seed, first_seed + 8)
            ],
        )
        (source / "manifest.json").write_text(json.dumps(manifest))
    return folders


def test_merge_preserves_original_data_and_references_both_sources(collections, tmp_path):
    original, corrections = collections
    before = (original / "manifest.json").read_bytes()
    output = tmp_path / "merged"
    merge_demonstrations(original, corrections, output)
    merged = json.loads((output / "manifest.json").read_text())
    assert merged["rehearsal_episodes"] == merged["correction_episodes"] == 8
    assert merged["episodes"][0]["file"] == str(original / "100.npz")
    assert merged["episodes"][-1]["file"] == str(corrections / "207.npz")
    assert (original / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("problem", ["wrong_arm", "duplicate", "failed", "insufficient"])
def test_merge_rejects_contaminated_training_data(collections, tmp_path, problem):
    original, corrections = collections
    manifest = json.loads((corrections / "manifest.json").read_text())
    if problem == "wrong_arm":
        manifest["arm"] = "left"
    elif problem == "duplicate":
        manifest["episodes"][0]["seed"] = 100
    elif problem == "failed":
        manifest["episodes"][0]["success"] = False
    else:
        manifest["episodes"].pop()
    (corrections / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        merge_demonstrations(original, corrections, tmp_path / "merged")
    assert not (tmp_path / "merged/manifest.json").exists()


def test_interactive_layout_keeps_tray_occupants_and_identity_when_spreading_table_objects():
    original = sample_layout(360001, occupied=2)
    arranged = bilateral_layout(original)
    assert arranged.seed == original.seed
    for index, (before, after) in enumerate(zip(original.objects, arranged.objects, strict=True)):
        assert after.name == before.name
        if before.in_tray or index % 2:
            assert after == before
        else:
            assert after.position == (before.position[0], -before.position[1], before.position[2])
            assert after.yaw == -before.yaw
    assert any(o.position[1] > 0 for o in arranged.objects if not o.in_tray)
    assert any(o.position[1] < 0 for o in arranged.objects if not o.in_tray)
