#!/usr/bin/env python3
"""Extract a compact, task-stratified Fractal action-chunk archive.

This script is intended to run where the LeRobot Fractal dataset is mounted.
It reads demonstrations only and excludes the binary gripper channel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SEED = 20260905


def episode_fold(episode: int) -> int:
    digest = hashlib.sha256(f"{SEED}:{episode}".encode()).digest()
    return int(digest[0] % 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--episodes-per-task", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    task_to_episodes: dict[str, list[tuple[int, int]]] = defaultdict(list)
    episodes_path = args.root / "meta" / "episodes.jsonl"
    with episodes_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            task = record["tasks"][0].strip()
            if task and record["length"] >= args.horizon + 2:
                task_to_episodes[task].append(
                    (int(record["episode_index"]), int(record["length"])))

    eligible = sorted(
        task for task, episodes in task_to_episodes.items()
        if len(episodes) >= args.episodes_per_task
    )
    # Hash ordering is deterministic but does not favor the most frequent tasks.
    eligible.sort(key=lambda task: hashlib.sha256(f"{SEED}:{task}".encode()).digest())
    selected_tasks = eligible[: args.tasks]

    chunks: list[np.ndarray] = []
    tasks: list[str] = []
    episodes: list[int] = []
    stages: list[int] = []
    folds: list[int] = []
    steps: list[int] = []

    rng = np.random.default_rng(SEED)
    for task in selected_tasks:
        candidates = task_to_episodes[task]
        chosen = rng.choice(len(candidates), size=args.episodes_per_task, replace=False)
        for chosen_position, candidate_index in enumerate(chosen):
            episode, expected_length = candidates[int(candidate_index)]
            path = (
                args.root
                / "data"
                / f"chunk-{episode // 1000:03d}"
                / f"episode_{episode:06d}.parquet"
            )
            table = pq.read_table(path, columns=["action"])
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if len(action) != expected_length:
                raise RuntimeError(f"length mismatch for {path}: {len(action)} != {expected_length}")
            # One complete action chunk from the center of each progress decile.
            latest = len(action) - args.horizon
            for stage in range(10):
                step = int(round((stage + 0.5) * latest / 10.0))
                chunks.append(action[step : step + args.horizon, :6])
                tasks.append(task)
                episodes.append(episode)
                stages.append(stage)
                # Exactly balance the two disjoint demonstration folds per task.
                folds.append(int(chosen_position % 2))
                steps.append(step)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": str(args.root),
        "seed": SEED,
        "task_count": len(selected_tasks),
        "episodes_per_task": args.episodes_per_task,
        "horizon": args.horizon,
        "continuous_channels": list(range(6)),
        "gripper_excluded": True,
        "sampling": "one complete chunk at the center of each episode-progress decile",
        "tasks": selected_tasks,
    }
    np.savez_compressed(
        args.out,
        gt=np.asarray(chunks, dtype=np.float32),
        task=np.asarray(tasks),
        episode=np.asarray(episodes, dtype=np.int64),
        stage=np.asarray(stages, dtype=np.int8),
        split=np.asarray(folds, dtype=np.int8),
        step=np.asarray(steps, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps({"out": str(args.out), "shape": list(np.asarray(chunks).shape), **metadata}, indent=2))


if __name__ == "__main__":
    main()
