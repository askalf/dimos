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

"""Convert arm-local primitive demonstrations and warm-start ACT from the trained model."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dimos_lerobot.prepare_r1pro_dataset import stable_joint_statistics
from dimos_lerobot.prepare_r1pro_deployment import prepare as export_deployment
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
import numpy as np
from safetensors.torch import load_file
import torch

from dimos.imitation.profile import ImageSource, JointPositionSource, VectorSource
from dimos.robot.galaxea.r1pro.object_packing import OBJECT_PACKING_IO
from dimos.robot.galaxea.r1pro.object_primitives import (
    MIRROR_ARM_SIGNS,
    Arm,
    Primitive,
    active_indices,
    goal_indices,
    primitive_profile,
)


def convert(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "manifest.json").read_text())
    primitive, arm = manifest["primitive"], manifest["arm"]
    profile = primitive_profile(primitive, arm)
    if (
        manifest["profile"] != profile.name
        or not manifest["images"]
        or manifest["fps"] != 20
        or manifest["joints"] != list(profile.action.demonstration.joints)
        or not manifest["episodes"]
    ):
        raise ValueError("Expected nonempty verified RGB primitive demonstrations")
    features: dict[str, Any] = {}
    for key, spec in profile.observations.items():
        if isinstance(spec, ImageSource):
            features[key] = dict(
                dtype="image", shape=spec.shape, names=["height", "width", "channels"]
            )
        elif isinstance(spec, JointPositionSource):
            features[key] = dict(
                dtype="float32", shape=(len(spec.joints),), names=list(spec.joints)
            )
        else:
            features[key] = dict(
                dtype="float32", shape=(len(spec.features),), names=list(spec.features)
            )
    features["action"] = dict(
        dtype="float32", shape=(8,), names=list(profile.action.demonstration.joints)
    )
    dataset = LeRobotDataset.create(
        repo_id=f"local/r1pro-{primitive}-{arm}",
        root=output,
        fps=20,
        robot_type=profile.robot_type,
        features=features,
        use_videos=False,
        image_writer_threads=4,
    )
    states, actions, goals = [], [], []
    try:
        for row in manifest["episodes"]:
            if not row["success"] or row["primitive"] != primitive or row["arm"] != arm:
                raise ValueError("Refusing failed or mismatched primitive demonstrations")
            with np.load(source / row["file"], allow_pickle=False) as data:
                arrays = {key: data[key] for key in features}
                if any(
                    len(a) != row["frames"] or a.shape[1:] != features[key]["shape"]
                    for key, a in arrays.items()
                ):
                    raise ValueError("Frame/feature dimensions do not match the primitive contract")
                for index in range(row["frames"]):
                    dataset.add_frame(
                        {
                            **{key: a[index] for key, a in arrays.items()},
                            "task": "Grasp, lift and hold the selected object."
                            if primitive == "pick"
                            else "Place the held object at the assigned supported location, release and retreat.",
                        }
                    )
                dataset.save_episode()
                states.append(arrays["observation.state"])
                actions.append(arrays["action"])
                goals.append(arrays["observation.environment_state"])
            print(
                f"Converted {primitive}-{arm} seed={row['seed']} object={row['selected']}",
                flush=True,
            )
    finally:
        dataset.finalize()  # type: ignore[no-untyped-call]
    stats_path = output / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats.update(stable_joint_statistics(np.concatenate(states), np.concatenate(actions)))
    env = np.concatenate(goals).astype(np.float64)
    stats["observation.environment_state"] = dict(
        mean=env.mean(0).tolist(),
        std=np.maximum(env.std(0), 0.01).tolist(),
        min=env.min(0).tolist(),
        max=env.max(0).tolist(),
        count=[len(env)],
    )
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    (output / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def geometry_mirror() -> torch.Tensor:
    signs = torch.ones(52)
    signs[[1, 4, 22]] = -1
    signs[6:15] = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1])
    for i in range(4):
        signs[24 + 7 * i + 2] = -1
    return signs


def migrate_weights(
    weights: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    primitive: Primitive,
    arm: Arm,
) -> dict[str, torch.Tensor]:
    """Retain compatible learned weights; map physical left-arm signs explicitly.

    Moving inactive joints into the ENV token and removing the destination from
    pick changes the architecture. This is initialization, not an equivalence
    claim or evidence of left-arm policy transfer.
    """
    if weights.keys() != target.keys():
        raise ValueError("Unexpected ACT architecture change")
    indices = list(active_indices("right"))
    columns = list(goal_indices(primitive))
    joint_sign = (
        torch.as_tensor(MIRROR_ARM_SIGNS, dtype=torch.float32) if arm == "left" else torch.ones(8)
    )
    env_sign = geometry_mirror()[columns] if arm == "left" else torch.ones(len(columns))
    for key, original in weights.items():
        if key in (
            "model.encoder_robot_state_input_proj.weight",
            "model.vae_encoder_robot_state_input_proj.weight",
            "model.vae_encoder_action_input_proj.weight",
        ):
            value = original[:, indices] * joint_sign
        elif key in ("model.action_head.weight", "model.action_head.bias"):
            value = original[indices] * (joint_sign[:, None] if original.ndim == 2 else joint_sign)
        elif key == "model.encoder_env_state_input_proj.weight":
            value = torch.zeros_like(target[key])
            value[:, : len(columns)] = original[:, columns] * env_sign
        else:
            value = original
        if value.shape != target[key].shape:
            raise ValueError(
                f"Unexpected checkpoint dimension at {key}: {value.shape} != {target[key].shape}"
            )
        target[key] = value.clone()
    return target


def prepare(source: Path, dataset: Path, output: Path, primitive: Primitive, arm: Arm) -> None:
    if output.exists():
        raise FileExistsError(output)
    profile = primitive_profile(primitive, arm)
    manifest = json.loads((dataset / "collection_manifest.json").read_text())
    if manifest["profile"] != profile.name or source.resolve() == dataset.resolve():
        raise ValueError("Dataset must be a new derivative with the exact primitive profile")
    config = PreTrainedConfig.from_pretrained(source)
    if (
        not isinstance(config, ACTConfig)
        or tuple(config.input_features["observation.state"].shape) != (20,)
        or tuple(config.input_features["observation.environment_state"].shape) != (52,)
        or tuple(config.output_features["action"].shape) != (20,)
    ):
        raise ValueError("Expected the existing 20-joint/52-goal object ACT checkpoint")
    if json.loads((source / "deployment.json").read_text())["profile"] != OBJECT_PACKING_IO.name:
        raise ValueError("Source must declare the trained object profile")
    goal_spec = profile.observations["observation.environment_state"]
    assert isinstance(goal_spec, VectorSource)
    config.input_features["observation.images.wrist"] = config.input_features.pop(
        "observation.images.right_wrist"
    )
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    config.input_features["observation.environment_state"] = PolicyFeature(
        type=FeatureType.ENV, shape=(len(goal_spec.features),)
    )
    config.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(8,))
    config.pretrained_path = None
    config.pretrained_backbone_weights = None
    config.device = "cpu"
    torch.manual_seed(173)
    policy = ACTPolicy(config)
    policy.load_state_dict(
        migrate_weights(
            load_file(source / "model.safetensors"), policy.state_dict(), primitive, arm
        )
    )
    normalizers = list(source.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if len(normalizers) != 1:
        raise ValueError("Expected one saved source normalizer")
    saved = load_file(normalizers[0])
    stats_path = dataset / "meta/stats.json"
    raw_stats = json.loads(stats_path.read_text())
    (dataset / "normalization-before-warm-start.json").write_text(
        json.dumps(raw_stats, indent=2) + "\n"
    )
    indices = list(active_indices("right"))
    columns = list(goal_indices(primitive))
    sign = (
        torch.as_tensor(MIRROR_ARM_SIGNS, dtype=torch.float32) if arm == "left" else torch.ones(8)
    )
    env_sign = geometry_mirror()[columns] if arm == "left" else torch.ones(len(columns))
    for key in ("observation.state", "action"):
        raw_stats[key]["mean"] = (saved[f"{key}.mean"][indices] * sign).tolist()
        raw_stats[key]["std"] = saved[f"{key}.std"][indices].tolist()
    for stat in ("mean", "std"):
        values = saved[f"observation.environment_state.{stat}"][columns]
        if stat == "mean":
            values = values * env_sign
        raw_stats["observation.environment_state"][stat][: len(columns)] = values.tolist()
        for key, original in (
            ("observation.images.head", "observation.images.head"),
            ("observation.images.wrist", "observation.images.right_wrist"),
        ):
            raw_stats[key][stat] = saved[f"{original}.{stat}"].tolist()
    stats_path.write_text(json.dumps(raw_stats, indent=2) + "\n")
    stats = {
        key: {name: torch.tensor(value) for name, value in values.items()}
        for key, values in raw_stats.items()
    }
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    policy.save_pretrained(output)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    provenance = dict(
        source=str(source.resolve()),
        source_sha256=hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest(),
        profile=profile.name,
        primitive=primitive,
        arm=arm,
        source_action_columns=indices,
        source_goal_columns=columns,
        context_columns_initialized_to_zero=12,
        compatible_normalization_preserved=True,
        fine_tuning_pending=True,
        behavior_equivalence_claimed=False,
    )
    (output / "initialization.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("convert", "initialize", "export"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--primitive", choices=("pick", "place"))
    parser.add_argument("--arm", choices=("left", "right"))
    args = parser.parse_args()
    if args.operation == "convert":
        convert(args.source, args.output)
    elif args.operation == "export":
        if args.primitive is None or args.arm is None:
            parser.error("Export requires --primitive and --arm")
        export_deployment(args.source, args.output, 20, primitive_profile(args.primitive, args.arm))
    else:
        if args.dataset is None or args.primitive is None or args.arm is None:
            parser.error("Initialization requires --dataset, --primitive and --arm")
        prepare(args.source, args.dataset, args.output, args.primitive, args.arm)


if __name__ == "__main__":
    main()
