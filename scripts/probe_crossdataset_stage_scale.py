#!/usr/bin/env python3
"""Cross-dataset, cross-demonstration probe of stage-dependent action scale.

For every task and episode-progress decile, estimate an action-chunk mean and
radius from one episode fold, then evaluate the residual spread on held-out
episodes.  Actions are never used to assign their own scale group.  Continuous
channels only; binary grippers are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


SEED = 20260905
HORIZON = 8
STAGES = 10


@dataclass
class TaskData:
    dataset: str
    task: str
    action: np.ndarray  # [N, H, C], continuous channels only
    episode: np.ndarray
    stage: np.ndarray
    split: np.ndarray


def stable_fold(namespace: str, episode: int) -> int:
    digest = hashlib.sha256(f"{SEED}:{namespace}:{episode}".encode()).digest()
    return int(digest[0] % 2)


def select_one_per_episode_stage(
    action: np.ndarray,
    task: np.ndarray,
    episode: np.ndarray,
    stage: np.ndarray,
    split: np.ndarray,
    step: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Balance archives to one chunk per demonstration and progress decile."""
    chosen: list[int] = []
    episode_keys = sorted(set(zip(task.tolist(), episode.tolist())))
    for task_name, ep in episode_keys:
        ep_mask = (task == task_name) & (episode == ep)
        for st in range(STAGES):
            ids = np.flatnonzero(ep_mask & (stage == st))
            if not len(ids):
                continue
            target = (st + 0.5) / STAGES
            progress = step[ids] / np.maximum(length[ids] - 1, 1)
            chosen.append(int(ids[np.argmin(np.abs(progress - target))]))
    chosen_array = np.asarray(chosen, dtype=np.int64)
    return (
        action[chosen_array], task[chosen_array], episode[chosen_array],
        stage[chosen_array], split[chosen_array]
    )


def load_npz(path: Path, label: str) -> list[TaskData]:
    archive = np.load(path, allow_pickle=True)
    metadata = json.loads(str(archive["metadata"].item()))
    action = archive["gt"].astype(np.float64)
    if "valid_time" in archive:
        start = int(metadata["executed_start"])
        horizon = min(HORIZON, int(metadata["executed_horizon"]))
        channels = np.asarray(metadata["continuous_channels"], dtype=int)
        valid = archive["valid_time"][:, start : start + horizon].all(axis=1)
        action = action[:, start : start + horizon, :][:, :, channels]
    else:
        valid = np.ones(len(action), dtype=bool)
    task = archive["task"].astype(str)
    episode = archive["episode"].astype(np.int64)
    stage = archive["stage"].astype(np.int8)
    split = archive["split"].astype(np.int8)
    step = archive["step"].astype(np.int64)
    if "length" in archive:
        length = archive["length"].astype(np.int64)
    else:
        # Extracted LeRobot archives contain exactly one point per stage.
        length = np.maximum(step + HORIZON, 1)
    action, task, episode, stage, split = select_one_per_episode_stage(
        action[valid], task[valid], episode[valid], stage[valid], split[valid],
        step[valid], length[valid]
    )
    result = []
    for name in np.unique(task):
        keep = task == name
        result.append(TaskData(label, name, action[keep], episode[keep], stage[keep], split[keep]))
    return result


def load_robomimic_hdf5(path: Path, task: str, label: str = "RoboMimic") -> TaskData:
    chunks: list[np.ndarray] = []
    episodes: list[int] = []
    stages: list[int] = []
    splits: list[int] = []
    with h5py.File(path, "r") as handle:
        names = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
        for episode_index, name in enumerate(names):
            action = np.asarray(handle["data"][name]["actions"], dtype=np.float64)
            latest = len(action) - HORIZON
            if latest < 1:
                continue
            for stage in range(STAGES):
                step = int(round((stage + 0.5) * latest / STAGES))
                chunks.append(action[step : step + HORIZON, :6])
                episodes.append(episode_index)
                stages.append(stage)
                splits.append(stable_fold(task, episode_index))
    return TaskData(
        label, task, np.asarray(chunks), np.asarray(episodes),
        np.asarray(stages), np.asarray(splits)
    )


def normalize_channels(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return active channels and conservative per-channel reference scales."""
    scale = reference.reshape(-1, reference.shape[-1]).std(axis=0, ddof=1)
    active = scale > max(1e-8, float(scale.max()) * 1e-4)
    positive = scale[active]
    floor = max(1e-8, float(np.median(positive)) * 0.05)
    return active, np.maximum(scale[active], floor)


def task_stage_records(data: TaskData) -> list[dict]:
    records: list[dict] = []
    for query_fold in (0, 1):
        reference_fold = 1 - query_fold
        reference_all = data.action[data.split == reference_fold]
        active, channel_scale = normalize_channels(reference_all)
        scaled = data.action[:, :, active] / channel_scale[None, None, :]
        for stage in range(STAGES):
            reference = scaled[(data.split == reference_fold) & (data.stage == stage)]
            query_mask = (data.split == query_fold) & (data.stage == stage)
            query = scaled[query_mask]
            if len(reference) < 3 or len(query) < 3:
                continue
            mean = reference.mean(axis=0)
            reference_radius = float(np.sqrt(np.var(reference, axis=0, ddof=1).mean()))
            residual = query - mean
            query_sumsq = float(np.sum(residual ** 2))
            query_count = int(residual.size)
            records.append({
                "dataset": data.dataset,
                "task": data.task,
                "query_fold": query_fold,
                "stage": stage,
                "reference_radius": reference_radius,
                "heldout_rms": float(np.sqrt(query_sumsq / query_count)),
                "query_sumsq": query_sumsq,
                "query_count": query_count,
                "query_episodes": int(len(np.unique(data.episode[query_mask]))),
                "active_channels": int(active.sum()),
            })
    return records


def ratio_from_records(records: list[dict]) -> tuple[float, list[dict]]:
    """Q5/Q1 held-out RMS after within-task/fold normalization."""
    enriched: list[dict] = []
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in records:
        groups.setdefault((row["task"], row["query_fold"]), []).append(row)
    for rows in groups.values():
        radius_median = float(np.median([row["reference_radius"] for row in rows]))
        rms_median = float(np.median([row["heldout_rms"] for row in rows]))
        for row in rows:
            enriched.append({
                **row,
                "normalized_radius": row["reference_radius"] / radius_median,
                "normalized_heldout_rms": row["heldout_rms"] / rms_median,
            })
    predicted = np.asarray([row["normalized_radius"] for row in enriched])
    actual = np.asarray([row["normalized_heldout_rms"] for row in enriched])
    edges = np.quantile(predicted, np.linspace(0, 1, 6))
    quintile = np.searchsorted(edges[1:-1], predicted, side="right")
    medians = [float(np.median(actual[quintile == q])) for q in range(5)]
    for row, q in zip(enriched, quintile):
        row["quintile"] = int(q + 1)
    return medians[-1] / medians[0], enriched


def summarize_dataset(records: list[dict], rng: np.random.Generator, demonstrations: int) -> dict:
    ratio, enriched = ratio_from_records(records)
    tasks = np.asarray(sorted({row["task"] for row in records}))
    boot = np.empty(4000, dtype=np.float64)
    for iteration in range(len(boot)):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        duplicated: list[dict] = []
        for copy_index, task in enumerate(sampled):
            for row in records:
                if row["task"] == task:
                    duplicated.append({**row, "task": f"{task}::boot{copy_index}"})
        boot[iteration] = ratio_from_records(duplicated)[0]

    predicted = np.asarray([row["normalized_radius"] for row in enriched])
    actual = np.asarray([row["normalized_heldout_rms"] for row in enriched])
    quintile = np.asarray([row["quintile"] for row in enriched])
    quintiles = []
    for q in range(1, 6):
        keep = quintile == q
        quintiles.append({
            "quintile": q,
            "predicted_radius_median": float(np.median(predicted[keep])),
            "heldout_rms_median": float(np.median(actual[keep])),
            "n_task_stage_folds": int(keep.sum()),
        })

    # Within-task/fold permutation: destroy stage--radius correspondence only.
    null = np.empty(4000, dtype=np.float64)
    group_keys = sorted({(row["task"], row["query_fold"]) for row in enriched})
    for iteration in range(len(null)):
        permuted = [dict(row) for row in enriched]
        for key in group_keys:
            ids = [i for i, row in enumerate(permuted) if (row["task"], row["query_fold"]) == key]
            radii = np.asarray([permuted[i]["reference_radius"] for i in ids])
            for i, value in zip(ids, rng.permutation(radii)):
                permuted[i]["reference_radius"] = float(value)
        null[iteration] = ratio_from_records(permuted)[0]
    permutation_p = float((1 + np.sum(null >= ratio)) / (1 + len(null)))

    task_ratios = []
    for task in tasks:
        task_rows = [row for row in records if row["task"] == task]
        task_ratios.append(ratio_from_records(task_rows)[0])
    return {
        "dataset": records[0]["dataset"],
        "tasks": int(len(tasks)),
        "task_stage_folds": int(len(records)),
        "demonstrations": int(demonstrations),
        "central_80_radius_span": float(np.quantile(predicted, 0.9) / np.quantile(predicted, 0.1)),
        "q5_over_q1_heldout_rms": float(ratio),
        "task_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "within_task_stage_permutation_p": permutation_p,
        "fraction_tasks_q5_over_q1_gt_one": float(np.mean(np.asarray(task_ratios) > 1)),
        "task_ratio_median": float(np.median(task_ratios)),
        "quintiles": quintiles,
        "normalized_radius": predicted.tolist(),
        "normalized_heldout_rms": actual.tolist(),
        "task_ratios": task_ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("analysis/paper/data_ht_motivation/cross_dataset")
    )
    args = parser.parse_args()
    task_data: list[TaskData] = []

    # Five RoboMimic manipulation tasks, including two dual-arm / long-horizon tasks.
    for file_name, task in (
        ("lift_ph_low_dim.hdf5", "Lift"),
        ("can_ph_low_dim.hdf5", "Can"),
        ("square_ph_low_dim.hdf5", "Square"),
    ):
        task_data.append(load_robomimic_hdf5(Path("data/hf") / file_name, task))
    task_data += load_npz(Path("analysis/paper/mse_scale/tool_hang.npz"), "RoboMimic")
    task_data += load_npz(Path("analysis/paper/mse_scale/transport.npz"), "RoboMimic")
    task_data += load_npz(Path("analysis/paper/mse_scale/gr1.npz"), "RoboCasa / GR1")
    task_data += load_npz(Path("analysis/paper/mse_scale/pi05.npz"), "LIBERO")
    task_data += load_npz(args.out / "bridge_stage_probe.npz", "Bridge / WidowX")
    task_data += load_npz(args.out / "fractal_stage_probe.npz", "Fractal")

    all_records = [row for task in task_data for row in task_stage_records(task)]
    rng = np.random.default_rng(SEED)
    summaries = []
    order = ["RoboMimic", "LIBERO", "RoboCasa / GR1", "Bridge / WidowX", "Fractal"]
    for dataset in order:
        rows = [row for row in all_records if row["dataset"] == dataset]
        demonstrations = sum(
            len(np.unique(task.episode)) for task in task_data if task.dataset == dataset
        )
        summaries.append(summarize_dataset(rows, rng, demonstrations))

    args.out.mkdir(parents=True, exist_ok=True)
    output = {
        "seed": SEED,
        "horizon": HORIZON,
        "stages": STAGES,
        "protocol": (
            "For each task and episode-progress decile, estimate the continuous-action "
            "chunk mean and radius on one demonstration fold; group stage bins by that "
            "reference radius and reveal residual RMS only on disjoint held-out episodes. "
            "Action channels are scaled using reference-fold, task-wide statistics."
        ),
        "gripper_excluded": True,
        "datasets": summaries,
    }
    (args.out / "cross_dataset_scale_summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        item["dataset"]: {
            "tasks": item["tasks"],
            "span": item["central_80_radius_span"],
            "ratio": item["q5_over_q1_heldout_rms"],
            "ci": item["task_bootstrap_95_ci"],
            "p": item["within_task_stage_permutation_p"],
            "fraction_tasks": item["fraction_tasks_q5_over_q1_gt_one"],
        } for item in summaries
    }, indent=2))


if __name__ == "__main__":
    main()
