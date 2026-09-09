#!/usr/bin/env python3
"""Exact-state executability probe for the continuous WidowX conditional mean.

The gripper channel is intentionally not averaged: it is treated as a
separate discrete/Bernoulli variable.  At each on-policy state we estimate the
mean of the six continuous arm channels from K same-observation Flow samples.
We then restore the exact simulator state and execute three matched branches:

  1. a fresh sampled arm chunk;
  2. the estimated conditional-mean arm chunk; and
  3. the sampled arm medoid closest to that mean.

Each branch in a matched triple receives the same sampled gripper sequence.
Thus differences between conditions isolate the continuous arm action.  If an
MSE target is invalid because it averages incompatible strategies, the mean
branch should leave the outcome cloud or behave much worse than the medoid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
from gr00t.policy.server_client import PolicyClient
from probe_widowx_branching import (
    ACTION_NAMES,
    execute_chunk,
    pose_from_obs,
    restore_snapshot,
    sample_same_state,
    save_snapshot,
    task_state,
)


HOST = os.environ.get("POLICY_HOST", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "5555"))
OUT_STEM = Path(os.environ.get("OUT", "/mnt/pfs/yuchen/groot/widowx_arm_mean_v1"))
TASKS = tuple(
    part.strip()
    for part in os.environ.get(
        "TASKS", "widowx_close_drawer,widowx_put_eggplant_in_basket"
    ).split(",")
    if part.strip()
)
SEED = int(os.environ.get("SEED", "20260908"))
K_MEAN = int(os.environ.get("K_MEAN", "64"))
K_REFERENCE = int(os.environ.get("K_REFERENCE", "8"))
EXECUTION_HORIZON = int(os.environ.get("EXECUTION_HORIZON", "4"))
CASES_PER_TASK = int(os.environ.get("CASES_PER_TASK", "20"))
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "8"))
MAX_MACRO_STEPS = int(os.environ.get("MAX_MACRO_STEPS", "60"))
TARGET_STEPS = tuple(
    int(part) for part in os.environ.get("TARGET_STEPS", "0,4,8,16,24,36").split(",")
)


def arm_flat(chunks: np.ndarray) -> np.ndarray:
    return chunks[:, :EXECUTION_HORIZON, :6].reshape(len(chunks), -1)


def paired_chunk(arm: np.ndarray, source: np.ndarray) -> np.ndarray:
    result = source.copy()
    result[:EXECUTION_HORIZON, :6] = arm[:EXECUTION_HORIZON]
    return result


def execute_from_snapshot(
    env: gym.Env,
    task: str,
    snapshot: dict[str, Any],
    chunk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    observation = restore_snapshot(env, snapshot)
    initial_pose = pose_from_obs(observation)
    initial_metric, _ = task_state(env, task)
    _, poses, metrics, _, success, _ = execute_chunk(
        env, task, chunk, EXECUTION_HORIZON
    )
    return (
        np.asarray([initial_pose, *poses], dtype=np.float64),
        np.asarray([initial_metric, *metrics], dtype=np.float64),
        bool(success),
    )


def path_centrality(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Candidate/reference arrays have shape branch x time x pose-dimension."""
    ref_xyz = reference[..., :3]
    cand_xyz = candidate[..., :3]
    ref_rpy = np.unwrap(reference[..., 3:6], axis=1)
    cand_rpy = np.unwrap(candidate[..., 3:6], axis=1)
    center_xyz = ref_xyz.mean(axis=0)
    center_rpy = ref_rpy.mean(axis=0)
    post = slice(1, None)
    gap_xyz = np.sqrt(
        np.mean(np.sum((cand_xyz.mean(axis=0)[post] - center_xyz[post]) ** 2, axis=-1))
    )
    within_xyz = np.sqrt(
        np.mean(np.sum((ref_xyz[:, post] - center_xyz[None, post]) ** 2, axis=-1))
    )
    gap_rpy = np.sqrt(
        np.mean(np.sum((cand_rpy.mean(axis=0)[post] - center_rpy[post]) ** 2, axis=-1))
    )
    within_rpy = np.sqrt(
        np.mean(np.sum((ref_rpy[:, post] - center_rpy[None, post]) ** 2, axis=-1))
    )
    # Reference-branch distances give an interpretable empirical percentile:
    # 0.5 means typical, while >0.95 means outside almost every sampled path.
    ref_distance_xyz = np.sqrt(
        np.mean(np.sum((ref_xyz[:, post] - center_xyz[None, post]) ** 2, axis=-1), axis=1)
    )
    ref_distance_rpy = np.sqrt(
        np.mean(np.sum((ref_rpy[:, post] - center_rpy[None, post]) ** 2, axis=-1), axis=1)
    )
    return {
        "position_gap_cm": float(gap_xyz * 100.0),
        "position_within_rms_cm": float(within_xyz * 100.0),
        "position_between_within": float(gap_xyz / max(within_xyz, 1e-12)),
        "position_percentile": float(np.mean(ref_distance_xyz <= gap_xyz)),
        "orientation_gap_deg": float(np.rad2deg(gap_rpy)),
        "orientation_within_rms_deg": float(np.rad2deg(within_rpy)),
        "orientation_between_within": float(gap_rpy / max(within_rpy, 1e-12)),
        "orientation_percentile": float(np.mean(ref_distance_rpy <= gap_rpy)),
    }


def summarize_case(
    reference_pose: np.ndarray,
    mean_pose: np.ndarray,
    medoid_pose: np.ndarray,
    reference_metric: np.ndarray,
    mean_metric: np.ndarray,
    medoid_metric: np.ndarray,
) -> dict[str, Any]:
    return {
        "mean_arm": path_centrality(mean_pose, reference_pose),
        "medoid_arm": path_centrality(medoid_pose, reference_pose),
        "final_task_metric": {
            "reference_mean": float(reference_metric[:, -1].mean()),
            "reference_std": float(reference_metric[:, -1].std()),
            "mean_arm_mean": float(mean_metric[:, -1].mean()),
            "medoid_arm_mean": float(medoid_metric[:, -1].mean()),
        },
    }


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_states": len(cases)}
    for condition in ("mean_arm", "medoid_arm"):
        result[condition] = {}
        for key in (
            "position_gap_cm",
            "position_between_within",
            "position_percentile",
            "orientation_gap_deg",
            "orientation_between_within",
            "orientation_percentile",
        ):
            values = np.asarray(
                [case["summary"][condition][key] for case in cases], dtype=np.float64
            )
            result[condition][key] = {
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
            }
    mean_outside = np.asarray(
        [
            case["summary"]["mean_arm"]["position_percentile"] >= 0.95
            or case["summary"]["mean_arm"]["orientation_percentile"] >= 0.95
            for case in cases
        ]
    )
    result["mean_arm_outside_95pct_reference_paths"] = int(mean_outside.sum())
    return result


def main() -> None:
    np.random.seed(SEED)
    register_simpler_envs()
    client = PolicyClient(host=HOST, port=PORT, timeout_ms=120_000, strict=False)
    modality = client.get_modality_config()
    if len(modality["action"].delta_indices) < EXECUTION_HORIZON:
        raise RuntimeError("Policy action horizon is shorter than requested execution horizon")

    cases: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    replay_max_error = None
    try:
        for task_number, task in enumerate(TASKS):
            env = gym.make(f"simpler_env_widowx/{task}")
            collected = 0
            try:
                for episode in range(MAX_EPISODES):
                    if collected >= CASES_PER_TASK:
                        break
                    observation, _ = env.reset(seed=SEED + 1000 * task_number + episode)
                    ended = False
                    for macro_step in range(MAX_MACRO_STEPS):
                        if ended or collected >= CASES_PER_TASK:
                            break
                        snapshot = save_snapshot(env, observation)
                        if macro_step in TARGET_STEPS:
                            mean_draws = sample_same_state(client, observation, K_MEAN)
                            references = sample_same_state(client, observation, K_REFERENCE)
                            mean_arm = mean_draws[:, :EXECUTION_HORIZON, :6].mean(axis=0)
                            distances = np.sum(
                                (arm_flat(mean_draws) - mean_arm.reshape(1, -1)) ** 2,
                                axis=1,
                            )
                            medoid_arm = mean_draws[
                                int(np.argmin(distances)), :EXECUTION_HORIZON, :6
                            ]
                            reference_pose, mean_pose, medoid_pose = [], [], []
                            reference_metric, mean_metric, medoid_metric = [], [], []
                            success = {"reference": [], "mean_arm": [], "medoid_arm": []}
                            for reference in references:
                                pose, metric, ok = execute_from_snapshot(
                                    env, task, snapshot, reference
                                )
                                reference_pose.append(pose)
                                reference_metric.append(metric)
                                success["reference"].append(ok)
                                pose, metric, ok = execute_from_snapshot(
                                    env, task, snapshot, paired_chunk(mean_arm, reference)
                                )
                                mean_pose.append(pose)
                                mean_metric.append(metric)
                                success["mean_arm"].append(ok)
                                pose, metric, ok = execute_from_snapshot(
                                    env, task, snapshot, paired_chunk(medoid_arm, reference)
                                )
                                medoid_pose.append(pose)
                                medoid_metric.append(metric)
                                success["medoid_arm"].append(ok)
                            reference_pose = np.asarray(reference_pose)
                            mean_pose = np.asarray(mean_pose)
                            medoid_pose = np.asarray(medoid_pose)
                            reference_metric = np.asarray(reference_metric)
                            mean_metric = np.asarray(mean_metric)
                            medoid_metric = np.asarray(medoid_metric)
                            summary = summarize_case(
                                reference_pose,
                                mean_pose,
                                medoid_pose,
                                reference_metric,
                                mean_metric,
                                medoid_metric,
                            )
                            uid = f"{task}:ep{episode}:m{macro_step}"
                            case = {
                                "uid": uid,
                                "task": task,
                                "episode": episode,
                                "macro_step": macro_step,
                                "env_step": macro_step * EXECUTION_HORIZON,
                                "summary": summary,
                                "success_count": {
                                    key: int(np.sum(value)) for key, value in success.items()
                                },
                                "array_index": len(arrays),
                            }
                            cases.append(case)
                            arrays.append(
                                {
                                    "rgb": np.asarray(observation["video.image_0"]),
                                    "mean_draws": mean_draws.astype(np.float32),
                                    "references": references.astype(np.float32),
                                    "mean_arm": mean_arm.astype(np.float32),
                                    "medoid_arm": medoid_arm.astype(np.float32),
                                    "reference_pose": reference_pose.astype(np.float32),
                                    "mean_pose": mean_pose.astype(np.float32),
                                    "medoid_pose": medoid_pose.astype(np.float32),
                                }
                            )
                            collected += 1
                            print("MEAN_CASE " + json.dumps(case), flush=True)

                            if replay_max_error is None:
                                p0, _, _ = execute_from_snapshot(
                                    env, task, snapshot, references[0]
                                )
                                p1, _, _ = execute_from_snapshot(
                                    env, task, snapshot, references[0]
                                )
                                replay_max_error = float(np.max(np.abs(p0 - p1)))

                        else:
                            mean_draws = sample_same_state(client, observation, 1)

                        # Continue the unbranched rollout with the first draw used
                        # to estimate the mean.  Every selected state is therefore
                        # genuinely reached on policy.
                        observation = restore_snapshot(env, snapshot)
                        observation, _, _, _, _, ended = execute_chunk(
                            env, task, mean_draws[0], EXECUTION_HORIZON
                        )
            finally:
                env.close()
    finally:
        client.close()

    result = {
        "checkpoint": "/mnt/pfs/yuchen/groot/ft_wxflow/checkpoint-20000",
        "tasks": TASKS,
        "seed": SEED,
        "k_mean": K_MEAN,
        "k_reference": K_REFERENCE,
        "execution_horizon": EXECUTION_HORIZON,
        "target_macro_steps": TARGET_STEPS,
        "gripper_protocol": (
            "never averaged; each sampled/mean/medoid arm triple uses the same "
            "sampled gripper sequence"
        ),
        "state_protocol": "on-policy states from multiple seeded episodes",
        "snapshot_replay_max_abs_pose_error": replay_max_error,
        "aggregate": aggregate(cases),
        "cases": cases,
    }
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{OUT_STEM}.json", "w") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(
        f"{OUT_STEM}.npz",
        **{
            f"case_{index}_{key}": value
            for index, case_arrays in enumerate(arrays)
            for key, value in case_arrays.items()
        },
    )
    print(
        f"WIDOWX_ARM_MEAN_DONE states={len(cases)} replay_error={replay_max_error}",
        flush=True,
    )
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
