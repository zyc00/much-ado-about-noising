#!/usr/bin/env python3
"""Dump model-free, split-half local-residual scales for paper analysis.

For each action chunk, the local conditional mean is approximated by the mean
action chunk of nearest proprioceptive states from *other episodes* of the same
task.  The output keeps the squared residual energy in the first and second
halves of the chunk.  Persistence across halves is evidence that the local
scale belongs to the input, rather than being an isolated large residual.

This script deliberately reuses the dataset loaders and preprocessing in
analysis/paper/scripts_gain_rule/resid_stats.py so its numbers are directly
comparable with the existing gain-rule audit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/paper/scripts_gain_rule/resid_stats.py"


def load_module():
    spec = importlib.util.spec_from_file_location("resid_stats", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=50000)
    parser.add_argument("--save-residuals", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rs = load_module()
    if args.dataset not in rs.LOADERS:
        raise ValueError(f"Unknown dataset {args.dataset!r}")

    # Make subsampling and the random dimension split deterministic.
    rs.RNG = np.random.default_rng(0)
    episodes, gripper_dims, task_names = rs.LOADERS[args.dataset]()
    # GR1 hand commands are continuous joint targets, not binary gripper
    # events.  Keep them in the continuous-action likelihood analysis.
    if args.dataset == "gr1":
        gripper_dims = []

    all_actions = np.concatenate([episode[1] for episode in episodes])
    lo, hi = np.percentile(all_actions, [1, 99], axis=0)
    center = (hi + lo) / 2
    radius = (hi - lo) / 2
    keep = radius > 1e-6
    radius = np.where(keep, radius, 1.0)
    episodes = [
        (state, ((action - center) / radius)[:, keep], task)
        for state, action, task in episodes
    ]
    kept_dims = np.flatnonzero(keep).tolist()
    gripper_cols = [kept_dims.index(dim) for dim in gripper_dims if dim in kept_dims]
    continuous_cols = np.array(
        [dim for dim in range(len(kept_dims)) if dim not in gripper_cols], dtype=int
    )

    chunks, states, _, episode_ids, task_ids, _ = rs.make_chunks(
        episodes, args.horizon, args.stride
    )
    if len(chunks) > args.max_chunks:
        chosen = np.sort(
            rs.RNG.choice(len(chunks), size=args.max_chunks, replace=False)
        )
        chunks = chunks[chosen]
        states = states[chosen]
        episode_ids = episode_ids[chosen]
        task_ids = task_ids[chosen]

    residual, valid = rs.knn_residual(
        chunks, states, episode_ids, task_ids, K=args.neighbors
    )
    residual = residual[valid][:, :, continuous_cols].astype(np.float32)
    episode_ids = episode_ids[valid]
    task_ids = task_ids[valid]
    split = args.horizon // 2
    first = np.mean(residual[:, :split] ** 2, axis=(1, 2))
    second = np.mean(residual[:, split:] ** 2, axis=(1, 2))
    full = np.mean(residual**2, axis=(1, 2))

    metadata = {
        "dataset": args.dataset,
        "episodes_loaded": len(episodes),
        "chunks_before_knn_filter": int(len(chunks)),
        "chunks": int(len(residual)),
        "horizon": args.horizon,
        "continuous_dimensions": int(len(continuous_cols)),
        "neighbors": args.neighbors,
        "neighbor_rule": "same task, different episode",
        "action_normalization": "dataset q01/q99 affine scale; no clipping",
        "task_names": {str(k): str(v) for k, v in task_names.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        first_energy=first.astype(np.float32),
        second_energy=second.astype(np.float32),
        full_energy=full.astype(np.float32),
        episode=episode_ids,
        task=task_ids,
        metadata=np.array(json.dumps(metadata)),
    )
    if args.save_residuals:
        # Float16 is sufficient for visualization and keeps cluster transfer
        # small; all reported energy statistics above remain float32.
        arrays["residual"] = residual.astype(np.float16)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
