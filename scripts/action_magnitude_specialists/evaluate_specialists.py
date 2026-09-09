#!/usr/bin/env python3
"""Evaluate action-bin specialists on episode-disjoint matching-bin samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


SEED = 20260905
ROOT = Path("/mnt/pfs/yuchen/action_mag_specialists_20260905")


def _device(value):
    if torch.is_tensor(value):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: _device(item) for key, item in value.items()}
    return value


def _balanced_candidates(
    bins: np.ndarray,
    is_train: np.ndarray,
    tasks: np.ndarray,
    bin_index: int,
    per_task: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = []
    for task in np.unique(tasks):
        candidates = np.flatnonzero((bins == bin_index) & ~is_train & (tasks == task))
        count = min(per_task, len(candidates))
        selected.extend(rng.choice(candidates, count, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def _save(path: Path, metadata: dict, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".writing.npz")
    np.savez_compressed(temporary, metadata=np.asarray(json.dumps(metadata)), **arrays)
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def evaluate_pi05(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    checkpoint = ROOT / "runs" / f"pi05_q{args.bin}" / "checkpoints" / "005000" / "pretrained_model"
    manifest_path = ROOT / "manifests" / "pi05" / "pi05.npz"
    data_root = Path("/mnt/pfs/yuchen/pi05/libero_lerobot")
    model = PI05Policy.from_pretrained(str(checkpoint)).cuda().eval()
    if model.config.loss_type != "mse":
        raise ValueError(f"Expected MSE specialist, got {model.config.loss_type}")
    preprocessor, _ = make_pre_post_processors(model.config, pretrained_path=str(checkpoint))
    horizon = int(model.config.chunk_size)
    fps = json.loads((data_root / "meta" / "info.json").read_text())["fps"]
    dataset = LeRobotDataset(
        repo_id="HuggingFaceVLA/libero",
        root=data_root,
        delta_timestamps={"action": [step / fps for step in range(horizon)]},
    )
    with np.load(manifest_path) as manifest:
        bins = manifest["bins"]
        is_train = manifest["is_train"]
        tasks = manifest["task_ids"]
        magnitude = manifest["magnitude"]
        episodes = manifest["episode_ids"]
    rng = np.random.default_rng(SEED + args.bin)
    chosen = _balanced_candidates(bins, is_train, tasks, args.bin, args.per_task, rng)
    residuals, rms_values, action_magnitudes, task_values, episode_values, sample_indices = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for start in range(0, len(chosen), args.batch):
        indices = chosen[start : start + args.batch]
        raw = torch.utils.data.default_collate([dataset[int(index)] for index in indices])
        batch = _device(preprocessor(raw))
        target = batch["action"].float()
        with torch.inference_mode():
            prediction = model.predict_action_chunk(batch).float()
        residual = (prediction[..., :6] - target[..., :6]).cpu().numpy()
        residuals.append(residual)
        rms_values.append(np.sqrt(np.mean(np.square(residual, dtype=np.float64), axis=(1, 2))))
        action_magnitudes.append(magnitude[indices])
        task_values.append(tasks[indices])
        episode_values.append(episodes[indices])
        sample_indices.append(indices)
        if start % (10 * args.batch) == 0:
            print(f"pi05 Q{args.bin + 1}: {start}/{len(chosen)}", flush=True)
    residual_array = np.concatenate(residuals)
    metadata = {
        "stack": "pi05_libero",
        "bin": args.bin,
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
        "split": "held-out episodes, matching action-magnitude quintile",
        "continuous_dimensions": 6,
        "horizon": horizon,
        "samples": int(len(chosen)),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual_array, dtype=np.float64)))),
    }
    _save(
        Path(args.output),
        metadata,
        residual=residual_array,
        sample_rms=np.concatenate(rms_values),
        action_magnitude=np.concatenate(action_magnitudes),
        task=np.concatenate(task_values),
        episode=np.concatenate(episode_values),
        sample_index=np.concatenate(sample_indices),
    )


def _load_groot(checkpoint: Path, tag_name: str):
    from transformers import AutoModel, AutoProcessor

    import gr00t.model  # noqa: F401
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = EmbodimentTag.resolve(tag_name)
    backbone = Path(
        "/mnt/pfs/yuchen/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/"
        "snapshots/9ce19a195e423419c349abfc86fd07178b230561"
    )
    original = processing.build_processor
    processing.build_processor = lambda name, kwargs: original(str(backbone), kwargs)
    processor = AutoProcessor.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        model_name=str(backbone),
        transformers_loading_kwargs={"local_files_only": True},
    )
    processor.eval()
    model, loading = AutoModel.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        output_loading_info=True,
        transformers_loading_kwargs={"local_files_only": True},
    )
    if loading.get("missing_keys") or loading.get("unexpected_keys"):
        raise ValueError(loading)
    if model.config.loss_type != "mse":
        raise ValueError(f"Expected MSE specialist, got {model.config.loss_type}")
    return model.cuda().eval(), processor, tag


def evaluate_groot(args) -> None:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType

    stack = args.stack
    checkpoint = ROOT / "runs" / f"{stack}_q{args.bin}" / "checkpoint-5000"
    if stack == "gr1":
        tag_name = "ROBOCASA_GR1_TABLETOP"
        data_paths = sorted(Path("/mnt/pfs/yuchen/groot/lerobot/LeRobot").glob("gr1_unified.*"))
        manifest_dir = ROOT / "manifests" / "gr1"
        continuous_dimensions = 29
    else:
        tag_name = "SIMPLER_ENV_WIDOWX"
        data_paths = [Path("/mnt/pfs/yuchen/groot/bridge_orig_lerobot")]
        manifest_dir = ROOT / "manifests" / "widowx"
        continuous_dimensions = 6
    model, processor, tag = _load_groot(checkpoint, tag_name)
    modalities = processor.modality_configs[tag.value]
    rng = np.random.default_rng(SEED + args.bin)
    residuals, rms_values, action_magnitudes, task_values, episode_values, step_values = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    pending_features = []
    pending_meta = []

    def flush() -> None:
        if not pending_features:
            return
        batch = _device(processor.collator(pending_features))
        inner = batch["inputs"] if "inputs" in batch else batch
        target_full = inner["action"].float()
        action_mask = inner["action_mask"].bool()
        valid_time = torch.nonzero(action_mask[0].any(dim=1)).flatten()
        valid_action = torch.nonzero(action_mask[0].any(dim=0)).flatten()[:continuous_dimensions]
        target = target_full[:, valid_time][:, :, valid_action]
        inputs = {key: value for key, value in inner.items() if key != "action"}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction_full = model.get_action(inputs)["action_pred"].float()
        prediction = prediction_full[:, valid_time][:, :, valid_action]
        target = target.to(prediction.device)
        residual = (prediction - target).cpu().numpy()
        residuals.append(residual)
        rms_values.append(np.sqrt(np.mean(np.square(residual, dtype=np.float64), axis=(1, 2))))
        action_magnitudes.append(np.asarray([item["magnitude"] for item in pending_meta]))
        task_values.append(np.asarray([item["task"] for item in pending_meta]))
        episode_values.append(np.asarray([item["episode"] for item in pending_meta]))
        step_values.append(np.asarray([item["step"] for item in pending_meta]))
        pending_features.clear()
        pending_meta.clear()

    for task_index, path in enumerate(data_paths):
        manifest_path = (
            manifest_dir / f"{path.name}.npz" if stack == "gr1" else manifest_dir / "wx.npz"
        )
        with np.load(manifest_path) as manifest:
            bins = manifest["bins"]
            is_train = manifest["is_train"]
            magnitude = manifest["magnitude"]
            offsets = manifest["offsets"]
        candidates = np.flatnonzero((bins == args.bin) & ~is_train)
        count = min(args.per_task, len(candidates))
        chosen = rng.choice(candidates, count, replace=False)
        episode_ordinals = np.searchsorted(offsets[1:], chosen, side="right")
        steps = chosen - offsets[episode_ordinals]
        loader = LeRobotEpisodeLoader(path, modalities)
        for ordinal, step, flat_index in zip(episode_ordinals, steps, chosen, strict=True):
            episode = loader[int(ordinal)]
            data = extract_step_data(
                episode, int(step), modalities, tag, allow_padding=False
            )
            pending_features.append(
                processor([{"type": MessageType.EPISODE_STEP.value, "content": data}])
            )
            pending_meta.append(
                {
                    "magnitude": float(magnitude[flat_index]),
                    "task": task_index,
                    "episode": int(ordinal),
                    "step": int(step),
                }
            )
            if len(pending_features) == args.batch:
                flush()
        print(f"{stack} Q{args.bin + 1}: task {task_index + 1}/{len(data_paths)}", flush=True)
    flush()
    residual_array = np.concatenate(residuals)
    metadata = {
        "stack": stack,
        "bin": args.bin,
        "checkpoint": str(checkpoint),
        "manifest_dir": str(manifest_dir),
        "split": "held-out episodes, matching action-magnitude quintile",
        "continuous_dimensions": continuous_dimensions,
        "horizon": int(residual_array.shape[1]),
        "samples": int(len(residual_array)),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual_array, dtype=np.float64)))),
    }
    _save(
        Path(args.output),
        metadata,
        residual=residual_array,
        sample_rms=np.concatenate(rms_values),
        action_magnitude=np.concatenate(action_magnitudes),
        task=np.concatenate(task_values),
        episode=np.concatenate(episode_values),
        step=np.concatenate(step_values),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stack", choices=["gr1", "widowx", "pi05"])
    parser.add_argument("--bin", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--per-task", type=int, default=40)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    if args.stack == "pi05":
        evaluate_pi05(args)
    else:
        evaluate_groot(args)


if __name__ == "__main__":
    main()
