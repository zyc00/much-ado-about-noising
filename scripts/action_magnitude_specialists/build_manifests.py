#!/usr/bin/env python3
"""Build deterministic train/held-out action-magnitude quintile manifests.

The manifests contain only integer bin assignments and split membership. They do
not modify the source LeRobot datasets.  Magnitudes are measured after the same
q01/q99 action normalization used by the corresponding policy.  Binary gripper
channels are excluded; GR1 hand joints remain included.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


QUANTILES = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64)


@lru_cache(maxsize=None)
def _load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_quantiles(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    denom = np.asarray(q99, dtype=np.float32) - np.asarray(q01, dtype=np.float32)
    if np.any(denom <= 1e-12):
        raise ValueError("Degenerate continuous-action normalization range")
    normalized = 2.0 * (values - q01) / denom - 1.0
    return np.clip(normalized, -1.0, 1.0)


def _episode_holdout_flags(episode_ids: np.ndarray, task_ids: np.ndarray) -> np.ndarray:
    """Select every tenth episode within task, deterministically."""
    held_out = np.zeros(len(episode_ids), dtype=bool)
    for task in np.unique(task_ids):
        loc = np.flatnonzero(task_ids == task)
        loc = loc[np.argsort(episode_ids[loc], kind="stable")]
        held_out[loc[::10]] = True
    return held_out


def _assign_within_task_quintiles(
    values: np.ndarray, task_ids: np.ndarray, is_train: np.ndarray
) -> tuple[np.ndarray, dict[str, list[float]]]:
    bins = np.empty(len(values), dtype=np.uint8)
    thresholds: dict[str, list[float]] = {}
    for task in np.unique(task_ids):
        task_mask = task_ids == task
        fit_values = values[task_mask & is_train]
        if len(fit_values) < 10:
            raise ValueError(f"Task {task} has only {len(fit_values)} training samples")
        cuts = np.quantile(fit_values, QUANTILES)
        bins[task_mask] = np.searchsorted(cuts, values[task_mask], side="right")
        thresholds[str(int(task))] = cuts.astype(float).tolist()
    return bins, thresholds


def _episode_file(root: Path, episode_id: int, info: dict) -> Path:
    chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
    pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    values = {
        "episode_chunk": episode_id // chunk_size,
        "chunk_index": episode_id // chunk_size,
        "episode_index": episode_id,
        "file_index": episode_id,
    }
    path = root / pattern.format(**values)
    if path.exists():
        return path
    matches = list((root / "data").glob(f"**/episode_{episode_id:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot resolve episode {episode_id} under {root}")
    return matches[0]


def _read_episode_arrays(root: Path, episode_id: int, info: dict) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        _episode_file(root, episode_id, info), columns=["observation.state", "action"]
    )
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    return state, action


def _gr1_episode_rms(root: Path, episode_id: int, info: dict, horizon: int) -> np.ndarray:
    state, action = _read_episode_arrays(root, episode_id, info)
    count = len(action) - horizon + 1
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    future = np.arange(count)[:, None] + np.arange(horizon)[None, :]
    relative_stats = _load_json(root / "meta" / "relative_stats.json")
    stats = _load_json(root / "meta" / "stats.json")["action"]
    groups = {
        "left_arm": slice(0, 7),
        "left_hand": slice(7, 13),
        "right_arm": slice(22, 29),
        "right_hand": slice(29, 35),
    }
    squared = np.zeros(count, dtype=np.float64)
    dimensions = 0
    for name, slc in groups.items():
        rel = action[future, slc] - state[:count, None, slc]
        q01 = np.asarray(relative_stats[name]["q01"], dtype=np.float32)[:horizon]
        q99 = np.asarray(relative_stats[name]["q99"], dtype=np.float32)[:horizon]
        normalized = _normalize_quantiles(rel, q01, q99)
        squared += np.square(normalized, dtype=np.float64).sum(axis=(1, 2))
        dimensions += normalized.shape[1] * normalized.shape[2]
    waist = action[future, 41:44]
    q01 = np.asarray(stats["q01"], dtype=np.float32)[41:44]
    q99 = np.asarray(stats["q99"], dtype=np.float32)[41:44]
    waist = _normalize_quantiles(waist, q01, q99)
    squared += np.square(waist, dtype=np.float64).sum(axis=(1, 2))
    dimensions += waist.shape[1] * waist.shape[2]
    if dimensions != horizon * 29:
        raise AssertionError(f"Expected {horizon * 29} GR1 targets, got {dimensions}")
    return np.sqrt(squared / dimensions).astype(np.float32)


def _continuous_episode_rms(
    action: np.ndarray, q01: np.ndarray, q99: np.ndarray, horizon: int, pad: bool
) -> np.ndarray:
    action = _normalize_quantiles(action[:, :6], q01[:6], q99[:6])
    count = len(action) if pad else len(action) - horizon + 1
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    future = np.arange(count)[:, None] + np.arange(horizon)[None, :]
    if pad:
        future = np.minimum(future, len(action) - 1)
    chunks = action[future]
    return np.sqrt(np.mean(np.square(chunks, dtype=np.float64), axis=(1, 2))).astype(np.float32)


def build_groot(root: Path, output: Path, kind: str, horizon: int) -> None:
    info = _load_json(root / "meta" / "info.json")
    episodes = _load_jsonl(root / "meta" / "episodes.jsonl")
    episode_ids = np.asarray([row["episode_index"] for row in episodes], dtype=np.int64)
    # Each GR1 directory is one task. Bridge/WidowX contains many free-form
    # language annotations with too few episodes per exact string, so it uses
    # global quintiles rather than conflating annotations with semantic tasks.
    task_ids = np.zeros(len(episodes), dtype=np.int64)
    heldout_episode = _episode_holdout_flags(episode_ids, task_ids)

    offsets = [0]
    values = []
    row_tasks = []
    row_train = []
    for ordinal, (episode_id, task_id) in enumerate(zip(episode_ids, task_ids, strict=True)):
        if kind == "gr1":
            rms = _gr1_episode_rms(root, int(episode_id), info, horizon)
        else:
            _, action = _read_episode_arrays(root, int(episode_id), info)
            stats = _load_json(root / "meta" / "stats.json")["action"]
            rms = _continuous_episode_rms(
                action,
                np.asarray(stats["q01"], dtype=np.float32),
                np.asarray(stats["q99"], dtype=np.float32),
                horizon,
                pad=False,
            )
        values.append(rms)
        row_tasks.append(np.full(len(rms), task_id, dtype=np.int64))
        row_train.append(np.full(len(rms), not heldout_episode[ordinal], dtype=bool))
        offsets.append(offsets[-1] + len(rms))
        if (ordinal + 1) % 1000 == 0:
            print(f"{root.name}: {ordinal + 1}/{len(episodes)} episodes", flush=True)
    values_array = np.concatenate(values)
    task_array = np.concatenate(row_tasks)
    train_array = np.concatenate(row_train)
    bins, thresholds = _assign_within_task_quintiles(values_array, task_array, train_array)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        episode_ids=episode_ids,
        offsets=np.asarray(offsets, dtype=np.int64),
        bins=bins,
        is_train=train_array,
        magnitude=values_array,
    )
    summary = {
        "kind": kind,
        "source": str(root),
        "horizon": horizon,
        "continuous_dimensions": 29 if kind == "gr1" else 6,
        "gripper_excluded": kind == "widowx",
        "split": "deterministic 10% held-out episodes within task",
        "binning": "training-set quintiles within task",
        "thresholds": thresholds,
        "samples": int(len(values_array)),
        "train_samples_per_bin": [int(np.sum(train_array & (bins == i))) for i in range(5)],
        "heldout_samples_per_bin": [int(np.sum(~train_array & (bins == i))) for i in range(5)],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_pi05(root: Path, output: Path, horizon: int) -> None:
    columns = ["action", "episode_index", "frame_index", "task_index", "index"]
    tables = [pq.read_table(path, columns=columns) for path in sorted((root / "data").glob("**/*.parquet"))]
    table = pa.concat_tables(tables)
    order = np.argsort(np.asarray(table["index"], dtype=np.int64), kind="stable")
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]
    episode = np.asarray(table["episode_index"], dtype=np.int64)[order]
    task = np.asarray(table["task_index"], dtype=np.int64)[order]
    index = np.asarray(table["index"], dtype=np.int64)[order]
    if not np.array_equal(index, np.arange(len(index))):
        raise ValueError("LIBERO global indices are not contiguous")
    episode_ids, starts = np.unique(episode, return_index=True)
    episode_tasks = task[starts]
    heldout_episode = _episode_holdout_flags(episode_ids, episode_tasks)
    heldout_lookup = dict(zip(episode_ids.tolist(), heldout_episode.tolist(), strict=True))
    is_train = np.asarray([not heldout_lookup[int(ep)] for ep in episode], dtype=bool)
    stats = _load_json(root / "meta" / "stats.json")["action"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    magnitude = np.empty(len(action), dtype=np.float32)
    ends = np.r_[starts[1:], len(action)]
    for start, end in zip(starts, ends, strict=True):
        magnitude[start:end] = _continuous_episode_rms(
            action[start:end], q01, q99, horizon, pad=True
        )
    bins, thresholds = _assign_within_task_quintiles(magnitude, task, is_train)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        indices=index,
        episode_ids=episode,
        task_ids=task,
        bins=bins,
        is_train=is_train,
        magnitude=magnitude,
    )
    summary = {
        "kind": "pi05",
        "source": str(root),
        "horizon": horizon,
        "continuous_dimensions": 6,
        "gripper_excluded": True,
        "split": "deterministic 10% held-out episodes within task",
        "binning": "training-set quintiles within task",
        "thresholds": thresholds,
        "samples": int(len(magnitude)),
        "train_samples_per_bin": [int(np.sum(is_train & (bins == i))) for i in range(5)],
        "heldout_samples_per_bin": [int(np.sum(~is_train & (bins == i))) for i in range(5)],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["gr1", "widowx", "pi05"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    if args.kind == "pi05":
        build_pi05(args.root, args.output, args.horizon or 50)
    else:
        build_groot(args.root, args.output, args.kind, args.horizon or 8)


if __name__ == "__main__":
    main()
