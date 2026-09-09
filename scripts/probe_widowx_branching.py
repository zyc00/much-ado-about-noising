#!/usr/bin/env python3
"""Same-state closed-loop branching test for WidowX Flow action modes.

The earlier multimodality probe establishes that some single-observation action
chunk distributions have two resolved components.  That alone does not say
whether the components are coherent task-level routes or merely local timing
phases.  This script performs the missing intervention:

1. run the Flow policy on-policy in SimplerEnv;
2. independently discover and confirm a translation/rotation action mode;
3. snapshot the complete simulator and controller state;
4. restore the identical state for balanced branches initialized from either
   component; and
5. let every branch replan closed-loop with the same stochastic policy.

If the components encode coherent routes, their label should continue to
predict later end-effector paths or task progress.  If they are local phase
choices, the initial separation should collapse and the geometric paths should
overlap after allowing temporal alignment.

The simulator and model live in different Python environments on the cluster,
so this script is run with SimplerEnv's Python and obtains actions from the
standard GR00T policy server.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.signal import find_peaks

from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
from gr00t.policy.server_client import PolicyClient


ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
GROUP_COLUMNS = {
    "translation": (0, 1, 2),
    "rotation": (3, 4, 5),
}

HOST = os.environ.get("POLICY_HOST", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "5555"))
OUT_STEM = Path(
    os.environ.get("OUT", "/mnt/pfs/yuchen/groot/widowx_branching_v1")
)
TASKS = tuple(
    value.strip()
    for value in os.environ.get(
        "TASKS", "widowx_close_drawer,widowx_put_eggplant_in_basket"
    ).split(",")
    if value.strip()
)
SEED = int(os.environ.get("SEED", "20260904"))
K_SCREEN = int(os.environ.get("K_SCREEN", "8"))
K_DISCOVERY = int(os.environ.get("K_DISCOVERY", "32"))
K_CONFIRM = int(os.environ.get("K_CONFIRM", "64"))
INFER_BATCH = int(os.environ.get("INFER_BATCH", "8"))
EXECUTION_HORIZON = int(os.environ.get("EXECUTION_HORIZON", "4"))
MODE_TEST_HORIZON = int(os.environ.get("MODE_TEST_HORIZON", str(EXECUTION_HORIZON)))
BRANCH_ENV_STEPS = int(os.environ.get("BRANCH_ENV_STEPS", "48"))
BRANCHES_PER_MODE = int(os.environ.get("BRANCHES_PER_MODE", "12"))
MAX_SCREEN_STATES = int(os.environ.get("MAX_SCREEN_STATES", "180"))
MAX_DEEP_TESTS = int(os.environ.get("MAX_DEEP_TESTS", "24"))
CASES_PER_TASK = int(os.environ.get("CASES_PER_TASK", "1"))
MAX_EPISODE_ENV_STEPS = int(os.environ.get("MAX_EPISODE_ENV_STEPS", "300"))
SCREEN_SEP = float(os.environ.get("SCREEN_SEP", "4.5"))
N_NULL = int(os.environ.get("N_NULL", "5000"))
N_PERM = int(os.environ.get("N_PERM", "5000"))
FORCE_FIRST_BRANCH = os.environ.get("FORCE_FIRST_BRANCH", "0") == "1"


def normal_logpdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    return -0.5 * (math.log(2.0 * math.pi * var) + (x - mean) ** 2 / var)


def fit_gmm2(z: np.ndarray) -> dict[str, Any]:
    """Fit deterministic 1-D one- and two-Gaussian models and return delta-BIC."""
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    total_var = max(float(z.var()), 1e-10)
    floor = max(total_var * 1e-3, 1e-10)
    best = None
    for q0, q1 in ((0.2, 0.8), (0.3, 0.7), (0.4, 0.6)):
        means = np.quantile(z, [q0, q1]).astype(np.float64)
        variances = np.array([total_var * 0.5, total_var * 0.5])
        weights = np.array([0.5, 0.5])
        old_ll = -np.inf
        for _ in range(200):
            logp = np.column_stack(
                [
                    math.log(max(weights[j], 1e-12))
                    + normal_logpdf(z, float(means[j]), float(variances[j]))
                    for j in range(2)
                ]
            )
            row_max = logp.max(axis=1, keepdims=True)
            log_norm = row_max + np.log(np.exp(logp - row_max).sum(axis=1, keepdims=True))
            resp = np.exp(logp - log_norm)
            ll = float(log_norm.sum())
            nk = resp.sum(axis=0).clip(min=1e-8)
            weights = nk / n
            means = (resp * z[:, None]).sum(axis=0) / nk
            variances = (resp * (z[:, None] - means) ** 2).sum(axis=0) / nk
            variances = np.maximum(variances, floor)
            if abs(ll - old_ll) < 1e-8:
                break
            old_ll = ll
        if best is None or ll > best[0]:
            best = (ll, weights.copy(), means.copy(), variances.copy(), resp.copy())
    assert best is not None
    ll2, weights, means, variances, resp = best
    order = np.argsort(means)
    weights, means, variances, resp = (
        weights[order],
        means[order],
        variances[order],
        resp[:, order],
    )
    ll1 = float(normal_logpdf(z, float(z.mean()), total_var).sum())
    delta_bic = (2 * math.log(n) - 2 * ll1) - (5 * math.log(n) - 2 * ll2)
    pooled = math.sqrt(float(np.sum(weights * variances)))
    return {
        "delta_bic": float(delta_bic),
        "sep": abs(float(means[1] - means[0])) / max(pooled, 1e-12),
        "min_weight": float(weights.min()),
        "weights": weights,
        "means": means,
        "variances": variances,
        "labels": np.argmax(resp, axis=1).astype(np.int8),
    }


def pca_screen_stat(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if not len(singular) or singular[0] < 1e-12:
        return 0.0
    z = centered @ vh[0]
    c0, c1 = float(z.min()), float(z.max())
    labels = np.zeros(len(z), dtype=bool)
    for _ in range(100):
        new_labels = np.abs(z - c1) < np.abs(z - c0)
        if new_labels.all() or (~new_labels).all():
            return 0.0
        n0, n1 = float(z[~new_labels].mean()), float(z[new_labels].mean())
        if np.array_equal(labels, new_labels) or abs(n0 - c0) + abs(n1 - c1) < 1e-12:
            labels = new_labels
            break
        labels, c0, c1 = new_labels, n0, n1
    if labels.all() or (~labels).all():
        return 0.0
    n0, n1 = int((~labels).sum()), int(labels.sum())
    v0 = float(z[~labels].var(ddof=1)) if n0 > 1 else 0.0
    v1 = float(z[labels].var(ddof=1)) if n1 > 1 else 0.0
    pooled = math.sqrt(
        ((n0 - 1) * v0 + (n1 - 1) * v1)
        / max(len(z) - 2, 1)
    )
    return abs(float(z[labels].mean() - z[~labels].mean())) / max(pooled, 1e-12)


def density_geometry(fit: dict[str, Any], z: np.ndarray) -> tuple[int, float]:
    pad = max(float(z.std()), 1e-6)
    grid = np.linspace(float(z.min() - pad), float(z.max() + pad), 5000)
    density = np.zeros_like(grid)
    for weight, mean, variance in zip(
        fit["weights"], fit["means"], fit["variances"]
    ):
        density += (
            weight
            / math.sqrt(2 * math.pi * variance)
            * np.exp(-0.5 * (grid - mean) ** 2 / variance)
        )
    peaks, _ = find_peaks(density, prominence=density.max() * 1e-5)
    if len(peaks) < 2:
        return len(peaks), 0.0
    chosen = sorted(sorted(peaks, key=lambda i: density[i], reverse=True)[:2])
    valley = density[chosen[0] : chosen[1] + 1].min()
    depth = 1.0 - valley / min(density[chosen[0]], density[chosen[1]])
    return len(peaks), float(depth)


def null_delta_bic(n: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        result[i] = fit_gmm2(rng.standard_normal(n))["delta_bic"]
    return np.sort(result)


def empirical_p(value: float, sorted_null: np.ndarray) -> float:
    tail = len(sorted_null) - int(np.searchsorted(sorted_null, value, side="left"))
    return float((tail + 1) / (len(sorted_null) + 1))


def flatten_group(chunks: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    return chunks[:, :MODE_TEST_HORIZON, list(columns)].reshape(chunks.shape[0], -1)


def model_observation(raw_observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack raw WidowX observations into GR00T's (B,T,...) sim format."""
    result: dict[str, Any] = {}
    keys = raw_observations[0].keys()
    for key in keys:
        if key.startswith("video."):
            result[key] = np.stack(
                [np.asarray(obs[key], dtype=np.uint8)[None] for obs in raw_observations]
            )
        elif key.startswith("state."):
            result[key] = np.stack(
                [
                    np.asarray(obs[key], dtype=np.float32).reshape(1, -1)
                    for obs in raw_observations
                ]
            )
        elif key.startswith("annotation."):
            result[key] = tuple(str(obs[key]) for obs in raw_observations)
    return result


def concatenate_action(actions: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(actions[f"action.{name}"], dtype=np.float32) for name in ACTION_NAMES],
        axis=-1,
    )


def request_actions(
    client: PolicyClient, observations: list[dict[str, Any]], batch_size: int = INFER_BATCH
) -> np.ndarray:
    chunks = []
    for start in range(0, len(observations), batch_size):
        action, _ = client.get_action(model_observation(observations[start : start + batch_size]))
        chunks.append(concatenate_action(action))
    return np.concatenate(chunks, axis=0).astype(np.float64)


def sample_same_state(client: PolicyClient, observation: dict[str, Any], k: int) -> np.ndarray:
    return request_actions(client, [observation] * k)


def core_env(env: gym.Env):
    return env.unwrapped.env.unwrapped


def save_snapshot(env: gym.Env, observation: dict[str, Any]) -> dict[str, Any]:
    base = env.unwrapped
    core = core_env(env)
    return {
        "sim": core.get_state().copy(),
        "agent": copy.deepcopy(core.agent.get_state()),
        "episode_stats": copy.deepcopy(getattr(core, "episode_stats", None)),
        "core_elapsed": copy.deepcopy(getattr(core, "_elapsed_steps", None)),
        "inner_elapsed": copy.deepcopy(getattr(base.env, "_elapsed_steps", None)),
        "observation": copy.deepcopy(observation),
    }


def restore_snapshot(env: gym.Env, snapshot: dict[str, Any]) -> dict[str, Any]:
    base = env.unwrapped
    core = core_env(env)
    core.set_state(snapshot["sim"].copy())
    core.agent.set_state(copy.deepcopy(snapshot["agent"]))
    if snapshot["episode_stats"] is not None:
        core.episode_stats = copy.deepcopy(snapshot["episode_stats"])
    if snapshot["core_elapsed"] is not None:
        core._elapsed_steps = copy.deepcopy(snapshot["core_elapsed"])
    if snapshot["inner_elapsed"] is not None:
        base.env._elapsed_steps = copy.deepcopy(snapshot["inner_elapsed"])
    return copy.deepcopy(snapshot["observation"])


def pose_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(np.asarray(obs[f"state.{name}"]).reshape(-1)[0]) for name in ACTION_NAMES],
        dtype=np.float64,
    )


def task_state(env: gym.Env, task: str) -> tuple[float, np.ndarray]:
    core = core_env(env)
    if task == "widowx_close_drawer":
        qpos = float(core.art_obj.get_qpos()[core.joint_idx])
        return qpos, np.full(3, np.nan, dtype=np.float64)
    source = np.asarray(core.episode_source_obj.pose.p, dtype=np.float64)
    target = np.asarray(core.episode_target_obj.pose.p, dtype=np.float64)
    return float(np.linalg.norm(source - target)), source


def execute_chunk(
    env: gym.Env,
    task: str,
    chunk: np.ndarray,
    n_steps: int,
) -> tuple[dict[str, Any], list[np.ndarray], list[float], list[np.ndarray], bool, bool]:
    poses, metrics, objects = [], [], []
    success = False
    ended = False
    observation: dict[str, Any] | None = None
    for step in range(min(n_steps, len(chunk))):
        action = {
            f"action.{name}": np.asarray([chunk[step, i]], dtype=np.float32)
            for i, name in enumerate(ACTION_NAMES)
        }
        observation, _, terminated, truncated, info = env.step(action)
        poses.append(pose_from_obs(observation))
        metric, obj = task_state(env, task)
        metrics.append(metric)
        objects.append(obj)
        success |= bool(info.get("success", False))
        ended = bool(terminated or truncated)
        if ended:
            break
    assert observation is not None
    return observation, poses, metrics, objects, success, ended


def deep_mode_test(
    client: PolicyClient,
    observation: dict[str, Any],
    group: str,
    null: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    discovery = sample_same_state(client, observation, K_DISCOVERY)
    confirmation = sample_same_state(client, observation, K_CONFIRM)
    gd = flatten_group(discovery, GROUP_COLUMNS[group])
    gc = flatten_group(confirmation, GROUP_COLUMNS[group])
    center = gd.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(gd - center, full_matrices=False)
    if not len(singular) or singular[0] < 1e-12:
        raise ValueError("Degenerate discovery batch")
    axis = vh[0]
    projected = (gc - center) @ axis
    fit = fit_gmm2(projected)
    n_density_modes, valley_depth = density_geometry(fit, projected)
    fit["p"] = empirical_p(fit["delta_bic"], null)
    fit["n_density_modes"] = int(n_density_modes)
    fit["valley_depth"] = float(valley_depth)
    fit["axis"] = axis
    fit["center"] = center[0]
    fit["projected"] = projected
    fit["detected"] = bool(
        fit["p"] <= 0.01
        and fit["delta_bic"] >= 10.0
        and fit["sep"] >= 2.0
        and fit["min_weight"] >= 0.2
        and n_density_modes >= 2
        and valley_depth >= 0.25
    )
    return fit, discovery, confirmation


def choose_balanced(labels: np.ndarray, projected: np.ndarray, means: np.ndarray) -> np.ndarray:
    selected = []
    for mode in (0, 1):
        indices = np.flatnonzero(labels == mode)
        if len(indices) < BRANCHES_PER_MODE:
            raise ValueError(
                f"Mode {mode} has only {len(indices)} confirmation samples; "
                f"need {BRANCHES_PER_MODE}"
            )
        # Use typical, high-responsibility representatives rather than tails.
        order = indices[np.argsort(np.abs(projected[indices] - means[mode]))]
        selected.extend(order[:BRANCHES_PER_MODE].tolist())
    return np.asarray(selected, dtype=np.int64)


def pad_series(values: list[Any], length: int, width: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        array = np.full((1, width), np.nan) if width else np.full(1, np.nan)
    if len(array) >= length:
        return array[:length]
    return np.concatenate([array, np.repeat(array[-1:], length - len(array), axis=0)], axis=0)


def run_branches(
    env: gym.Env,
    client: PolicyClient,
    task: str,
    state_snapshot: dict[str, Any],
    confirmation: np.ndarray,
    fit: dict[str, Any],
) -> dict[str, Any]:
    selected = choose_balanced(fit["labels"], fit["projected"], fit["means"])
    branches = []
    initial_observation = state_snapshot["observation"]
    initial_pose = pose_from_obs(initial_observation)
    restore_snapshot(env, state_snapshot)
    initial_metric, initial_object = task_state(env, task)

    # First, intervene with a representative sample from the assigned component.
    for sample_index in selected:
        observation = restore_snapshot(env, state_snapshot)
        label = int(fit["labels"][sample_index])
        observation, poses, metrics, objects, success, ended = execute_chunk(
            env, task, confirmation[sample_index], EXECUTION_HORIZON
        )
        branches.append(
            {
                "mode": label,
                "sample_index": int(sample_index),
                "initial_action": confirmation[sample_index].copy(),
                "observation": observation,
                "snapshot": save_snapshot(env, observation),
                "poses": [initial_pose.copy(), *poses],
                "task_metric": [initial_metric, *metrics],
                "object_xyz": [initial_object.copy(), *objects],
                "success": success,
                "ended": ended,
            }
        )

    # Then let all active branches replan.  Policy inference is batched, while a
    # single simulator is reused by restoring each branch's private snapshot.
    while max(len(branch["poses"]) for branch in branches) - 1 < BRANCH_ENV_STEPS:
        active = [i for i, branch in enumerate(branches) if not branch["ended"]]
        if not active:
            break
        observations = [branches[i]["observation"] for i in active]
        chunks = request_actions(client, observations)
        for local_i, branch_i in enumerate(active):
            branch = branches[branch_i]
            restore_snapshot(env, branch["snapshot"])
            remaining = BRANCH_ENV_STEPS - (len(branch["poses"]) - 1)
            observation, poses, metrics, objects, success, ended = execute_chunk(
                env,
                task,
                chunks[local_i],
                min(EXECUTION_HORIZON, remaining),
            )
            branch["observation"] = observation
            branch["snapshot"] = save_snapshot(env, observation)
            branch["poses"].extend(poses)
            branch["task_metric"].extend(metrics)
            branch["object_xyz"].extend(objects)
            branch["success"] |= success
            branch["ended"] |= ended

    poses = np.stack(
        [pad_series(branch["poses"], BRANCH_ENV_STEPS + 1, 7) for branch in branches]
    )
    metrics = np.stack(
        [pad_series(branch["task_metric"], BRANCH_ENV_STEPS + 1) for branch in branches]
    )
    objects = np.stack(
        [pad_series(branch["object_xyz"], BRANCH_ENV_STEPS + 1, 3) for branch in branches]
    )
    initial_actions = np.stack([branch["initial_action"] for branch in branches])
    labels = np.asarray([branch["mode"] for branch in branches], dtype=np.int8)
    success = np.asarray([branch["success"] for branch in branches], dtype=bool)
    return {
        "poses": poses,
        "task_metric": metrics,
        "object_xyz": objects,
        "initial_actions": initial_actions,
        "labels": labels,
        "success": success,
        "selected_confirmation_indices": selected,
    }


def symmetric_curve_rms(curve0: np.ndarray, curve1: np.ndarray) -> float:
    distances = np.linalg.norm(curve0[:, None, :] - curve1[None, :, :], axis=-1)
    return float(math.sqrt((np.mean(distances.min(axis=1) ** 2) + np.mean(distances.min(axis=0) ** 2)) / 2))


def permutation_p(
    values: np.ndarray, labels: np.ndarray, observed: float, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(N_PERM):
        shuffled = rng.permutation(labels)
        stat = float(
            np.sqrt(
                np.mean(
                    (values[shuffled == 1].mean(axis=0) - values[shuffled == 0].mean(axis=0))
                    ** 2
                )
            )
        )
        count += stat >= observed
    return float((count + 1) / (N_PERM + 1))


def summarize_branches(branch: dict[str, Any], seed: int) -> dict[str, Any]:
    poses = branch["poses"]
    labels = branch["labels"]
    xyz = poses[..., :3]
    rpy_deg = np.rad2deg(np.unwrap(poses[..., 3:6], axis=1))
    means_xyz = np.stack([xyz[labels == mode].mean(axis=0) for mode in (0, 1)])
    means_rpy = np.stack([rpy_deg[labels == mode].mean(axis=0) for mode in (0, 1)])
    gap_cm = np.linalg.norm(means_xyz[1] - means_xyz[0], axis=-1) * 100.0
    gap_deg = np.linalg.norm(means_rpy[1] - means_rpy[0], axis=-1)
    within_cm = np.zeros(len(gap_cm), dtype=np.float64)
    within_deg = np.zeros(len(gap_deg), dtype=np.float64)
    for t in range(len(gap_cm)):
        pos_residuals, rot_residuals = [], []
        for mode in (0, 1):
            pos_residuals.append(xyz[labels == mode, t] - means_xyz[mode, t])
            rot_residuals.append(rpy_deg[labels == mode, t] - means_rpy[mode, t])
        position_residuals = np.concatenate(pos_residuals)
        rotation_residuals = np.concatenate(rot_residuals)
        within_cm[t] = math.sqrt(np.mean(np.sum(position_residuals**2, axis=-1))) * 100.0
        within_deg[t] = math.sqrt(np.mean(np.sum(rotation_residuals**2, axis=-1)))
    late = slice(max(1, 3 * len(gap_cm) // 4), len(gap_cm))
    post_initial = slice(EXECUTION_HORIZON + 1, len(gap_cm))
    same_time_rms_cm = float(
        math.sqrt(np.mean(np.sum((means_xyz[1, post_initial] - means_xyz[0, post_initial]) ** 2, axis=-1)))
        * 100.0
    )
    curve_rms_cm = symmetric_curve_rms(
        means_xyz[0, post_initial], means_xyz[1, post_initial]
    ) * 100.0
    late_values = xyz[:, late, :]
    observed = float(
        np.sqrt(
            np.mean(
                (late_values[labels == 1].mean(axis=0) - late_values[labels == 0].mean(axis=0))
                ** 2
            )
        )
    )
    metrics = branch["task_metric"]
    return {
        "n_per_mode": [int((labels == mode).sum()) for mode in (0, 1)],
        "position_gap_cm": gap_cm.tolist(),
        "orientation_gap_deg": gap_deg.tolist(),
        "position_within_rms_cm": within_cm.tolist(),
        "orientation_within_rms_deg": within_deg.tolist(),
        "peak_first_chunk_position_gap_cm": float(gap_cm[1 : EXECUTION_HORIZON + 1].max()),
        "late_position_gap_cm": float(gap_cm[late].mean()),
        "late_orientation_gap_deg": float(gap_deg[late].mean()),
        "late_position_between_within": float(
            gap_cm[late].mean() / max(within_cm[late].mean(), 1e-12)
        ),
        "late_orientation_between_within": float(
            gap_deg[late].mean() / max(within_deg[late].mean(), 1e-12)
        ),
        "position_persistence_ratio": float(
            gap_cm[late].mean() / max(gap_cm[1 : EXECUTION_HORIZON + 1].max(), 1e-12)
        ),
        "same_time_route_rms_cm": same_time_rms_cm,
        "time_aligned_curve_rms_cm": curve_rms_cm,
        "curve_to_same_time_ratio": float(curve_rms_cm / max(same_time_rms_cm, 1e-12)),
        "late_path_permutation_p": permutation_p(late_values, labels, observed, seed),
        "final_task_metric_mean": [
            float(metrics[labels == mode, -1].mean()) for mode in (0, 1)
        ],
        "final_task_metric_delta": float(
            metrics[labels == 1, -1].mean() - metrics[labels == 0, -1].mean()
        ),
        "success_count": [
            int(branch["success"][labels == mode].sum()) for mode in (0, 1)
        ],
    }


def json_fit(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in fit.items()
        if key not in {"labels", "axis", "center", "projected"}
    }


def main() -> None:
    np.random.seed(SEED)
    null = null_delta_bic(K_CONFIRM, N_NULL, SEED + 17)
    register_simpler_envs()
    client = PolicyClient(host=HOST, port=PORT, timeout_ms=120_000, strict=False)
    modality = client.get_modality_config()
    for name in ("video", "state", "language"):
        delta = list(modality[name].delta_indices)
        if delta != [0]:
            raise RuntimeError(f"This probe expects {name} delta_indices=[0], got {delta}")
    action_horizon = len(modality["action"].delta_indices)
    if action_horizon < EXECUTION_HORIZON:
        raise RuntimeError(
            f"action horizon {action_horizon} < execution horizon {EXECUTION_HORIZON}"
        )
    if not (1 <= MODE_TEST_HORIZON <= EXECUTION_HORIZON):
        raise RuntimeError(
            f"mode-test horizon must lie in [1, {EXECUTION_HORIZON}], "
            f"got {MODE_TEST_HORIZON}"
        )
    print(
        f"TASKS={TASKS} K={K_SCREEN}/{K_DISCOVERY}/{K_CONFIRM} "
        f"branches={BRANCHES_PER_MODE}x2 horizon={BRANCH_ENV_STEPS}",
        flush=True,
    )

    cases: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    screen_log: list[dict[str, Any]] = []
    deep_attempts = 0
    total_screened = 0

    try:
        for task_no, task in enumerate(TASKS):
            env = gym.make(f"simpler_env_widowx/{task}")
            task_cases = 0
            task_screened = 0
            episode = 0
            try:
                while (
                    task_cases < CASES_PER_TASK
                    and task_screened < MAX_SCREEN_STATES
                    and deep_attempts < MAX_DEEP_TESTS
                ):
                    observation, _ = env.reset(seed=SEED + task_no * 1000 + episode)
                    episode += 1
                    macro_step = 0
                    ended = False
                    while (
                        not ended
                        and task_cases < CASES_PER_TASK
                        and task_screened < MAX_SCREEN_STATES
                        and deep_attempts < MAX_DEEP_TESTS
                        and macro_step * EXECUTION_HORIZON + BRANCH_ENV_STEPS
                        <= MAX_EPISODE_ENV_STEPS
                    ):
                        found_case = False
                        snapshot = save_snapshot(env, observation)
                        screen_chunks = sample_same_state(client, observation, K_SCREEN)
                        scores = {
                            group: pca_screen_stat(flatten_group(screen_chunks, columns))
                            for group, columns in GROUP_COLUMNS.items()
                        }
                        uid = f"{task}:ep{episode - 1}:m{macro_step}"
                        screen_log.append({"uid": uid, **scores})
                        task_screened += 1
                        total_screened += 1
                        ranked = sorted(scores, key=scores.get, reverse=True)
                        if scores[ranked[0]] >= SCREEN_SEP:
                            deep_attempts += 1
                            accepted = None
                            for group in ranked:
                                fit, discovery, confirmation = deep_mode_test(
                                    client, observation, group, null
                                )
                                print(
                                    f"deep {deep_attempts} {uid} {group}: "
                                    f"screen={scores[group]:.2f} dBIC={fit['delta_bic']:.1f} "
                                    f"sep={fit['sep']:.2f} w={fit['min_weight']:.2f} "
                                    f"valley={fit['valley_depth']:.2f} p={fit['p']:.4g}",
                                    flush=True,
                                )
                                forced = FORCE_FIRST_BRANCH and not cases
                                if (fit["detected"] or forced) and min(
                                    np.bincount(fit["labels"], minlength=2)
                                ) >= BRANCHES_PER_MODE:
                                    accepted = (group, fit, discovery, confirmation)
                                    break
                            if accepted is not None:
                                group, fit, discovery, confirmation = accepted
                                branch = run_branches(
                                    env,
                                    client,
                                    task,
                                    snapshot,
                                    confirmation,
                                    fit,
                                )
                                summary = summarize_branches(
                                    branch, SEED + 10000 + len(cases)
                                )
                                case = {
                                    "uid": uid,
                                    "task": task,
                                    "episode": episode - 1,
                                    "macro_step": macro_step,
                                    "env_step": macro_step * EXECUTION_HORIZON,
                                    "group": group,
                                    "screen_scores": scores,
                                    "confirmation": json_fit(fit),
                                    "summary": summary,
                                    "rgb_index": len(cases),
                                }
                                cases.append(case)
                                arrays.append(
                                    {
                                        "rgb": np.asarray(observation["video.image_0"]),
                                        "discovery": discovery.astype(np.float32),
                                        "confirmation": confirmation.astype(np.float32),
                                        **{key: value for key, value in branch.items() if isinstance(value, np.ndarray)},
                                    }
                                )
                                task_cases += 1
                                found_case = True
                                print("BRANCH_CASE " + json.dumps(case), flush=True)
                                observation = restore_snapshot(env, snapshot)

                        # Preserve an on-policy scan trajectory after any branch intervention.
                        observation = restore_snapshot(env, snapshot)
                        observation, _, _, _, _, ended = execute_chunk(
                            env,
                            task,
                            screen_chunks[0],
                            EXECUTION_HORIZON,
                        )
                        macro_step += 1
                        # Each accepted case comes from a different seeded episode,
                        # preventing several nearby phases of one trajectory from
                        # masquerading as independent route-level evidence.
                        if found_case:
                            ended = True
            finally:
                env.close()
    finally:
        client.close()

    # Bonferroni is reported across all adaptively triggered deep confirmations.
    for case in cases:
        raw_p = float(case["confirmation"]["p"])
        case["confirmation"]["bonferroni_p"] = min(1.0, raw_p * deep_attempts * 2)
    result = {
        "checkpoint": "/mnt/pfs/yuchen/groot/ft_wxflow/checkpoint-20000",
        "tasks": TASKS,
        "seed": SEED,
        "k_screen": K_SCREEN,
        "k_discovery": K_DISCOVERY,
        "k_confirmation": K_CONFIRM,
        "execution_horizon": EXECUTION_HORIZON,
        "mode_test_horizon": MODE_TEST_HORIZON,
        "max_episode_env_steps": MAX_EPISODE_ENV_STEPS,
        "branch_env_steps": BRANCH_ENV_STEPS,
        "branches_per_mode": BRANCHES_PER_MODE,
        "n_screened": total_screened,
        "n_deep_attempts": deep_attempts,
        "selection": {
            "screen_sep_min": SCREEN_SEP,
            "confirmation": (
                "fresh discovery PC1; independent confirmation; raw p<=.01; "
                "delta-BIC>=10; separation>=2 pooled SD; min component>=.2; "
                "two fitted density peaks; valley depth>=.25"
            ),
            "multiple_testing": "Bonferroni p reported across 2 groups x all deep attempts",
        },
        "null_delta_bic": {
            "repetitions": N_NULL,
            "q95": float(np.quantile(null, 0.95)),
            "q99": float(np.quantile(null, 0.99)),
        },
        "cases": cases,
        "screen_log": screen_log,
    }
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{OUT_STEM}.json", "w") as handle:
        json.dump(result, handle, indent=2)
    if arrays:
        np.savez_compressed(
            f"{OUT_STEM}.npz",
            **{
                f"case_{i}_{key}": value
                for i, case_arrays in enumerate(arrays)
                for key, value in case_arrays.items()
            },
        )
    print(
        f"WIDOWX_BRANCHING_DONE cases={len(cases)} screened={total_screened} "
        f"deep={deep_attempts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
