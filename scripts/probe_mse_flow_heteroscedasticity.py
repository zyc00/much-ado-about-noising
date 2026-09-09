#!/usr/bin/env python3
"""Matched held-out probe of MSE residual scale and Flow sample radius.

For each stack, select one state per task and episode-progress decile from the
held-out episodes defined by the action-magnitude manifests.  On exactly those
states, measure (1) the RMS residual of the converged deterministic MSE policy
and (2) the within-state RMS radius of K deployed Flow samples.  The script is
torchrun-friendly but does not require collective communication; every rank
writes an independent part file that is merged by the plotting script.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


SEED = 20260906
ROOT = Path("/mnt/pfs/yuchen")
PROBE_ROOT = ROOT / "hetero_scatter_20260906"
MANIFEST_ROOT = ROOT / "action_mag_specialists_20260905" / "manifests"


def to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def select_one_per_progress_cell(records: list[dict], rng: np.random.Generator) -> list[dict]:
    selected: list[dict] = []
    tasks = sorted({int(row["task_id"]) for row in records})
    for task_id in tasks:
        task_rows = [row for row in records if int(row["task_id"]) == task_id]
        for decile in range(10):
            candidates = [row for row in task_rows if int(row["progress_decile"]) == decile]
            if candidates:
                selected.append(candidates[int(rng.integers(0, len(candidates)))])
    selected.sort(key=lambda row: (int(row["task_id"]), int(row["progress_decile"])))
    return selected


def gr1_records() -> list[dict]:
    records: list[dict] = []
    files = sorted((MANIFEST_ROOT / "gr1").glob("*.npz"))
    for task_id, manifest_path in enumerate(files):
        with np.load(manifest_path) as manifest:
            episode_ids = manifest["episode_ids"].astype(np.int64)
            offsets = manifest["offsets"].astype(np.int64)
            is_train = manifest["is_train"].astype(bool)
            magnitude = manifest["magnitude"].astype(np.float64)
        source = Path(json.loads(manifest_path.with_suffix(".json").read_text())["source"])
        for ep_pos, episode in enumerate(episode_ids):
            lo, hi = int(offsets[ep_pos]), int(offsets[ep_pos + 1])
            length = hi - lo
            if length < 8 or bool(is_train[lo]):
                continue
            for step in range(0, length - 7):
                flat = lo + step
                progress = step / max(length - 1, 1)
                records.append(
                    {
                        "task_id": task_id,
                        "task": source.name,
                        "source": str(source),
                        "episode": int(episode),
                        "step": step,
                        "length": length,
                        "progress": progress,
                        "progress_decile": min(9, int(10 * progress)),
                        "label_scale": float(magnitude[flat]),
                    }
                )
    return select_one_per_progress_cell(records, np.random.default_rng(SEED))


def pi05_records() -> list[dict]:
    import pandas as pd

    root = ROOT / "pi05" / "libero_lerobot"
    with np.load(MANIFEST_ROOT / "pi05" / "pi05.npz") as manifest:
        indices = manifest["indices"].astype(np.int64)
        episode_ids = manifest["episode_ids"].astype(np.int64)
        task_ids = manifest["task_ids"].astype(np.int64)
        is_train = manifest["is_train"].astype(bool)
        magnitude = manifest["magnitude"].astype(np.float64)
    episodes = pd.concat(
        [pd.read_parquet(path) for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    episode_meta = {int(row.episode_index): row for _, row in episodes.iterrows()}
    records: list[dict] = []
    for pos in np.flatnonzero(~is_train):
        episode = int(episode_ids[pos])
        row = episode_meta[episode]
        start = int(row.dataset_from_index)
        length = int(row.length)
        step = int(indices[pos] - start)
        if step < 0 or step >= length:
            continue
        progress = step / max(length - 1, 1)
        records.append(
            {
                "task_id": int(task_ids[pos]),
                "task": str(row.tasks[0]),
                "source": str(root),
                "sample_index": int(indices[pos]),
                "episode": episode,
                "step": step,
                "length": length,
                "progress": progress,
                "progress_decile": min(9, int(10 * progress)),
                "label_scale": float(magnitude[pos]),
            }
        )
    return select_one_per_progress_cell(records, np.random.default_rng(SEED))


def load_gr1_model(checkpoint: Path):
    from transformers import AutoModel, AutoProcessor
    import gr00t.model  # noqa: F401 -- registers custom classes
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing

    backbone = (
        ROOT
        / "hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
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
    model, info = AutoModel.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        output_loading_info=True,
        transformers_loading_kwargs={"local_files_only": True},
    )
    if info.get("missing_keys") or info.get("unexpected_keys"):
        raise RuntimeError(info)
    return model.cuda().eval(), processor


def gr1_inputs(row: dict, processor, loader_cache: dict):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType

    source = row["source"]
    if source not in loader_cache:
        tag = EmbodimentTag.resolve("ROBOCASA_GR1_TABLETOP")
        loader_cache[source] = (LeRobotEpisodeLoader(Path(source), processor.modality_configs[tag.value]), tag)
    loader, tag = loader_cache[source]
    episode = loader[int(row["episode"])]
    modalities = processor.modality_configs[tag.value]
    data = extract_step_data(episode, int(row["step"]), modalities, tag, allow_padding=False)
    feature = processor([{"type": MessageType.EPISODE_STEP.value, "content": data}])
    batch = to_device(processor.collator([feature]), torch.device("cuda"))
    inner = batch["inputs"] if "inputs" in batch else batch
    mask = inner["action_mask"].bool()
    tv = torch.nonzero(mask[0].any(dim=1)).flatten()
    av = torch.nonzero(mask[0].any(dim=0)).flatten()
    if len(tv) != 8 or len(av) != 29:
        raise RuntimeError((len(tv), len(av)))
    target = inner["action"][:, tv][:, :, av].float().cuda()
    inputs = {key: value for key, value in inner.items() if key != "action"}
    return inputs, target, tv, av


def run_gr1(rows: list[dict], k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoints = {
        "mse": ROOT / "groot/ft_mse/checkpoint-60000",
        "flow": ROOT / "groot/ft_flow/checkpoint-60000",
    }
    mse_residual = np.empty(len(rows), dtype=np.float64)
    model, processor = load_gr1_model(checkpoints["mse"])
    if getattr(model.config, "loss_type", None) != "mse":
        raise RuntimeError(f"unexpected MSE loss_type={getattr(model.config, 'loss_type', None)}")
    cache: dict = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, row in enumerate(rows):
            inputs, target, tv, av = gr1_inputs(row, processor, cache)
            pred = model.get_action(inputs)["action_pred"][:, tv][:, :, av].float()
            mse_residual[i] = torch.mean((pred - target) ** 2).sqrt().item()
            if i % 10 == 0:
                print(f"MSE {i}/{len(rows)} rms={mse_residual[i]:.4f}", flush=True)
    del model, processor, cache
    gc.collect()
    torch.cuda.empty_cache()

    flow_radius = np.empty(len(rows), dtype=np.float64)
    flow_residual = np.empty(len(rows), dtype=np.float64)
    model, processor = load_gr1_model(checkpoints["flow"])
    if getattr(model.config, "loss_type", None) != "flow":
        raise RuntimeError(f"unexpected Flow loss_type={getattr(model.config, 'loss_type', None)}")
    cache = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, row in enumerate(rows):
            inputs, target, tv, av = gr1_inputs(row, processor, cache)
            samples = []
            for draw in range(k):
                torch.manual_seed(
                    SEED
                    + 1000003 * int(row["task_id"])
                    + 1009 * int(row["episode"])
                    + 101 * int(row["step"])
                    + draw
                )
                samples.append(model.get_action(inputs)["action_pred"][:, tv][:, :, av].float())
            samples_t = torch.cat(samples, dim=0)
            mean = samples_t.mean(dim=0, keepdim=True)
            flow_radius[i] = torch.var(samples_t, dim=0, unbiased=True).mean().sqrt().item()
            flow_residual[i] = torch.mean((mean - target) ** 2).sqrt().item()
            if i % 10 == 0:
                print(f"FLOW {i}/{len(rows)} radius={flow_radius[i]:.4f}", flush=True)
    return mse_residual, flow_radius, flow_residual


def load_pi05(checkpoint: Path):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies import make_pre_post_processors

    model = PI05Policy.from_pretrained(str(checkpoint)).cuda().eval()
    pre, _ = make_pre_post_processors(model.config, pretrained_path=str(checkpoint))
    return model, pre


def run_pi05(rows: list[dict], k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    checkpoints = {
        "mse": ROOT / "pi05/run_mse/checkpoints/030000/pretrained_model",
        "flow": ROOT / "pi05/libero_ft",
    }
    data_root = ROOT / "pi05/libero_lerobot"
    base = LeRobotDataset(repo_id="HuggingFaceVLA/libero", root=data_root)
    dataset = LeRobotDataset(
        repo_id="HuggingFaceVLA/libero",
        root=data_root,
        delta_timestamps={"action": [step / base.fps for step in range(50)]},
    )
    mse_residual = np.empty(len(rows), dtype=np.float64)
    model, pre = load_pi05(checkpoints["mse"])
    if getattr(model.config, "loss_type", None) != "mse":
        raise RuntimeError(f"unexpected MSE loss_type={getattr(model.config, 'loss_type', None)}")
    with torch.inference_mode():
        for i, row in enumerate(rows):
            raw = torch.utils.data.default_collate([dataset[int(row["sample_index"])]])
            batch = to_device(pre(raw), torch.device("cuda"))
            target = batch["action"][:, :10, :6].float()
            pred = model.predict_action_chunk(batch)[:, :10, :6].float()
            mse_residual[i] = torch.mean((pred - target) ** 2).sqrt().item()
            if i % 10 == 0:
                print(f"MSE {i}/{len(rows)} rms={mse_residual[i]:.4f}", flush=True)
    del model, pre
    gc.collect()
    torch.cuda.empty_cache()

    flow_radius = np.empty(len(rows), dtype=np.float64)
    flow_residual = np.empty(len(rows), dtype=np.float64)
    model, pre = load_pi05(checkpoints["flow"])
    with torch.inference_mode():
        for i, row in enumerate(rows):
            raw = torch.utils.data.default_collate([dataset[int(row["sample_index"])]])
            batch = to_device(pre(raw), torch.device("cuda"))
            target = batch["action"][:, :10, :6].float()
            samples = []
            for draw in range(k):
                torch.manual_seed(
                    SEED
                    + 1000003 * int(row["task_id"])
                    + 1009 * int(row["episode"])
                    + 101 * int(row["step"])
                    + draw
                )
                # Compiled PI0.5 inference may reuse CUDA-graph output storage on
                # the next invocation. Clone each draw before requesting another.
                samples.append(model.predict_action_chunk(batch)[:, :10, :6].float().clone())
            samples_t = torch.cat(samples, dim=0)
            mean = samples_t.mean(dim=0, keepdim=True)
            flow_radius[i] = torch.var(samples_t, dim=0, unbiased=True).mean().sqrt().item()
            flow_residual[i] = torch.mean((mean - target) ** 2).sqrt().item()
            if i % 10 == 0:
                print(f"FLOW {i}/{len(rows)} radius={flow_radius[i]:.4f}", flush=True)
    return mse_residual, flow_radius, flow_residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stack", choices=("gr1", "pi05"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output", type=Path, default=PROBE_ROOT / "parts")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.set_num_threads(4)

    all_rows = globals()[f"{args.stack}_records"]()
    rows = all_rows[rank::world]
    print(f"stack={args.stack} rank={rank}/{world} selected={len(rows)}/{len(all_rows)}", flush=True)
    mse_residual, flow_radius, flow_residual = globals()[f"run_{args.stack}"](rows, args.k)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.stack}_rank{rank:02d}.npz"
    np.savez_compressed(
        path,
        task=np.asarray([row["task"] for row in rows]),
        task_id=np.asarray([row["task_id"] for row in rows], dtype=np.int64),
        episode=np.asarray([row["episode"] for row in rows], dtype=np.int64),
        step=np.asarray([row["step"] for row in rows], dtype=np.int64),
        length=np.asarray([row["length"] for row in rows], dtype=np.int64),
        progress=np.asarray([row["progress"] for row in rows], dtype=np.float64),
        label_scale=np.asarray([row["label_scale"] for row in rows], dtype=np.float64),
        mse_residual=mse_residual,
        flow_radius=flow_radius,
        flow_residual=flow_residual,
        metadata=np.asarray(
            json.dumps(
                {
                    "stack": args.stack,
                    "rank": rank,
                    "world_size": world,
                    "seed": SEED,
                    "flow_draws_per_state": args.k,
                    "selection": "one held-out state per task and episode-progress decile",
                    "continuous_channels_only": True,
                }
            )
        ),
    )
    print(f"SAVED {path}", flush=True)


if __name__ == "__main__":
    main()
