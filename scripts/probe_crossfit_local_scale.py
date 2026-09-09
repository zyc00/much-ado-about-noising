#!/usr/bin/env python3
"""Cross-fit a model-free local scale estimate from proprioceptive state.

Episodes are split in two.  For every query chunk, both the local action mean
and the local residual scale are computed exclusively from chunks in the other
episode split.  The output can therefore test whether an input-only local scale
ranking predicts residual RMS on unseen episodes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/paper/scripts_gain_rule/resid_stats.py"


def load_module():
    spec = importlib.util.spec_from_file_location("resid_stats", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_state(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(train, axis=0)
    lo, hi = np.percentile(train, [10, 90], axis=0)
    scale = hi - lo
    keep = np.isfinite(scale) & (scale > 1e-6)
    if not keep.any():
        keep = np.ones(train.shape[1], dtype=bool)
        scale = np.ones(train.shape[1])
    return (train[:, keep] - center[keep]) / scale[keep], (query[:, keep] - center[keep]) / scale[keep]


def training_energy(
    states: np.ndarray,
    chunks: np.ndarray,
    episodes: np.ndarray,
    neighbors: int,
    continuous: np.ndarray,
) -> np.ndarray:
    """Leave-episode-out local residual energy for reference chunks."""
    sn, _ = normalize_state(states, states)
    tree = cKDTree(sn)
    _, near = tree.query(sn, k=min(neighbors * 8 + 1, len(sn)))
    energy = np.full(len(sn), np.nan, dtype=np.float64)
    for row in range(len(sn)):
        candidates = near[row]
        candidates = candidates[episodes[candidates] != episodes[row]][:neighbors]
        if len(candidates) < max(2, neighbors // 2):
            continue
        mean = chunks[candidates].mean(axis=0)
        energy[row] = np.mean((chunks[row][:, continuous] - mean[:, continuous]) ** 2)
    return energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rs = load_module()
    rs.RNG = np.random.default_rng(0)
    episodes, gripper_dims, task_names = rs.LOADERS[args.dataset]()
    if args.dataset == "gr1":
        gripper_dims = []

    all_actions = np.concatenate([episode[1] for episode in episodes])
    lo, hi = np.percentile(all_actions, [1, 99], axis=0)
    center, radius = (hi + lo) / 2, (hi - lo) / 2
    keep = radius > 1e-6
    radius = np.where(keep, radius, 1.0)
    episodes = [
        (state, ((action - center) / radius)[:, keep], task)
        for state, action, task in episodes
    ]
    kept_dims = np.flatnonzero(keep).tolist()
    gripper_cols = [kept_dims.index(dim) for dim in gripper_dims if dim in kept_dims]
    continuous = np.array([dim for dim in range(len(kept_dims)) if dim not in gripper_cols])

    chunks, states, _, episode_ids, task_ids, _ = rs.make_chunks(
        episodes, args.horizon, args.stride
    )
    if len(chunks) > args.max_chunks:
        chosen = np.sort(rs.RNG.choice(len(chunks), size=args.max_chunks, replace=False))
        chunks, states = chunks[chosen], states[chosen]
        episode_ids, task_ids = episode_ids[chosen], task_ids[chosen]

    predictions, observations, out_tasks, out_episodes = [], [], [], []
    skipped_tasks = []
    for task_id in np.unique(task_ids):
        task_index = np.flatnonzero(task_ids == task_id)
        unique_episodes = np.unique(episode_ids[task_index])
        if len(unique_episodes) < 6:
            skipped_tasks.append(int(task_id))
            continue
        # Deterministic episode-level split, balanced by alternating a shuffle.
        local_rng = np.random.default_rng(1009 + int(task_id))
        shuffled = unique_episodes.copy()
        local_rng.shuffle(shuffled)
        fold_of = {int(ep): i % 2 for i, ep in enumerate(shuffled)}
        for query_fold in (0, 1):
            is_query = np.array([fold_of[int(ep)] == query_fold for ep in episode_ids[task_index]])
            query_index = task_index[is_query]
            train_index = task_index[~is_query]
            train_eps = episode_ids[train_index]
            if len(np.unique(train_eps)) < 3 or len(query_index) < 10:
                continue

            train_state, query_state = normalize_state(states[train_index], states[query_index])
            ref_energy = training_energy(
                states[train_index], chunks[train_index], train_eps, args.neighbors, continuous
            )
            good_ref = np.isfinite(ref_energy)
            if good_ref.sum() < args.neighbors:
                continue
            tree = cKDTree(train_state[good_ref])
            _, near = tree.query(query_state, k=min(args.neighbors, good_ref.sum()))
            if near.ndim == 1:
                near = near[:, None]
            ref_index = train_index[good_ref]
            neighbors_global = ref_index[near]
            mean_chunk = chunks[neighbors_global].mean(axis=1)
            query_energy = np.mean(
                (chunks[query_index][:, :, continuous] - mean_chunk[:, :, continuous]) ** 2,
                axis=(1, 2),
            )
            # Geometric averaging is appropriate for a positive scale spanning
            # orders of magnitude and is robust to a few large reference errors.
            predicted_energy = np.exp(np.mean(np.log(ref_energy[good_ref][near] + 1e-12), axis=1))
            predictions.append(predicted_energy)
            observations.append(query_energy)
            out_tasks.append(np.full(len(query_index), task_id))
            out_episodes.append(episode_ids[query_index])

    if not predictions:
        raise RuntimeError("No task had enough episodes for cross-fitting")
    pred = np.concatenate(predictions).astype(np.float32)
    obs = np.concatenate(observations).astype(np.float32)
    out_task = np.concatenate(out_tasks)
    out_episode = np.concatenate(out_episodes)
    metadata = {
        "dataset": args.dataset,
        "queries": int(len(pred)),
        "tasks_used": int(len(np.unique(out_task))),
        "tasks_skipped": skipped_tasks,
        "continuous_dimensions": int(len(continuous)),
        "horizon": args.horizon,
        "neighbors": args.neighbors,
        "split": "two-fold, disjoint episodes within task",
        "mean_and_scale_sources": "opposite episode fold only",
        "scale_estimator": "geometric mean of leave-episode-out residual energy of nearest reference states",
        "task_names": {str(k): str(v) for k, v in task_names.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        predicted_energy=pred,
        observed_energy=obs,
        task=out_task,
        episode=out_episode,
        metadata=np.array(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
