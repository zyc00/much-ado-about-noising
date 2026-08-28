"""Train L2/HG/HT/MIP on actual successful noisy demonstration paths.

The environment is the same 2D funnel-to-dock point robot.  Noise is executed,
not added post hoc to stored labels.  At each even replanning cycle a path
samples a residual branch (70% zero, 20% +0.2, 10% +1.6 in normalized lateral
units).  The next action contains residual -delta/2, which makes the two-step
endpoint exactly equal to the clean oracle endpoint.  Thus every demonstration
is dynamically valid, collision-free, and successful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toy2d_mip_nfl import ANCHOR_SIGMA, T_TWO_STEP, train, train_nll


ROOT = Path(__file__).resolve().parents[1]
X_GOAL_MM = 160.0
FORWARD_MM = 4.0
FORWARD_ACTION_MM = 10.0
LATERAL_ACTION_MM = 2.0
K_SERVO = 0.5
DOCK_TOL_MM = 0.5
HORIZON = 8
N_CYCLES = 40
BRANCH_MODES = np.array([0.0] * 7 + [0.2] * 2 + [1.6], dtype=np.float32)


def half_width_mm(x):
    x = np.asarray(x)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def normalize_state(state):
    out = np.empty_like(state, dtype=np.float32)
    out[:, 0] = state[:, 0] / 80.0 - 1.0
    out[:, 1] = state[:, 1] / 14.0
    return out


def make_path_dataset(n_starts, repeats, horizon, seed):
    if repeats != 10:
        raise ValueError("repeats must be 10 for the exact 70/20/10 balance")
    rng = np.random.default_rng(seed)
    y_start = rng.uniform(-10.0, 10.0, size=n_starts).astype(np.float64)
    n_paths = n_starts * repeats
    y0 = np.repeat(y_start, repeats)

    # At each branch point, every identical-state group has exactly 7/2/1
    # clean/mild/gross paths, with assignments permuted across replicates.
    n_blocks = N_CYCLES // 2
    modes = np.empty((n_starts, repeats, n_blocks), dtype=np.float32)
    for i in range(n_starts):
        for b in range(n_blocks):
            modes[i, :, b] = rng.permutation(BRANCH_MODES)
    modes = modes.reshape(n_paths, n_blocks)

    total = N_CYCLES + horizon - 1
    states = np.empty((n_paths, N_CYCLES, 2), dtype=np.float32)
    actions = np.empty((n_paths, total, 2), dtype=np.float32)
    paths_y = np.empty((n_paths, N_CYCLES + 1), dtype=np.float32)
    paths_y[:, 0] = y0
    y = y0.copy()
    collision = np.zeros(n_paths, dtype=bool)

    for t in range(total):
        x = FORWARD_MM * t
        if t < N_CYCLES:
            states[:, t, 0] = x
            states[:, t, 1] = y
        if t < N_CYCLES and t % 2 == 0:
            residual = modes[:, t // 2].astype(np.float64)
        elif t < N_CYCLES:
            residual = -0.5 * modes[:, t // 2].astype(np.float64)
        else:
            residual = np.zeros(n_paths, dtype=np.float64)
        dy = -K_SERVO * y + residual * LATERAL_ACTION_MM
        actions[:, t, 0] = FORWARD_MM / FORWARD_ACTION_MM
        actions[:, t, 1] = dy / LATERAL_ACTION_MM
        y = y + dy
        if t < N_CYCLES:
            paths_y[:, t + 1] = y
            collision |= np.abs(y) > half_width_mm(x + FORWARD_MM)

    success = (~collision) & (np.abs(paths_y[:, -1]) < DOCK_TOL_MM)
    chunks = np.stack(
        [actions[:, t:t + horizon].reshape(n_paths, 2 * horizon)
         for t in range(N_CYCLES)],
        axis=1,
    )
    flat_state = normalize_state(states.reshape(-1, 2))
    flat_action = chunks.reshape(-1, 2 * horizon)
    order = rng.permutation(len(flat_state))

    chosen = np.linspace(0, n_paths - 1, 15, dtype=int)
    paths = [
        {
            "y0": float(y0[i]),
            "x": [float(v) for v in np.arange(N_CYCLES + 1) * FORWARD_MM],
            "y": [float(v) for v in paths_y[i]],
        }
        for i in chosen
    ]
    meta = {
        "n_paths": n_paths,
        "n_state_action_chunks": int(len(flat_state)),
        "demo_success_rate": float(success.mean()),
        "demo_collision_rate": float(collision.mean()),
        "demo_dock_error_max_mm": float(np.abs(paths_y[:, -1]).max()),
        "paths": paths,
    }
    return (torch.from_numpy(flat_state[order]),
            torch.from_numpy(flat_action[order]), meta)


@torch.inference_mode()
def predict(net, state_world, sampler, device):
    state = torch.from_numpy(normalize_state(state_world.astype(np.float32))).to(device)
    action_dim = net.action_dim
    z0 = torch.zeros((len(state), action_dim), device=device)
    t0 = torch.zeros((len(state), 1), device=device)
    raw = net(state, z0, t0)
    if sampler in ("hg", "ht"):
        return raw[:, :action_dim].cpu().numpy()
    first = raw
    if sampler in ("regression", "mip_step1"):
        out = first
    elif sampler == "mip_full":
        t1 = torch.full((len(state), 1), T_TWO_STEP, device=device)
        out = net(state, first, t1)
    else:
        raise ValueError(sampler)
    return out.cpu().numpy()


def rollout_metrics(net, sampler, device, n_eval):
    x = np.zeros(n_eval, dtype=np.float64)
    y = np.linspace(-10.0, 10.0, n_eval)
    y0 = y.copy()
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    traces_x = [[] for _ in range(n_eval)]
    traces_y = [[] for _ in range(n_eval)]

    for _ in range(70):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        state = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        action = predict(net, state, sampler, device)
        dx = np.clip(action[:, 0] * FORWARD_ACTION_MM, -2.0, 8.0)
        dy = np.clip(action[:, 1] * LATERAL_ACTION_MM, -10.0, 10.0)
        x[ids] += dx
        y[ids] += dy
        for local, idx in enumerate(ids):
            traces_x[idx].append(float(x[idx]))
            traces_y[idx].append(float(y[idx]))
        outside = np.abs(y[ids]) > half_width_mm(x[ids])
        collision[ids[outside]] = True
        reached = x[ids] >= X_GOAL_MM
        y_goal[ids[reached]] = y[ids[reached]]
        active[ids[outside | reached]] = False

    reached = np.isfinite(y_goal)
    success = reached & (np.abs(y_goal) < DOCK_TOL_MM) & ~collision
    chosen = np.linspace(0, n_eval - 1, 11, dtype=int)
    traces = [
        {"y0": float(y0[i]), "x": traces_x[i], "y": traces_y[i]}
        for i in chosen
    ]
    return {
        "success_rate": float(success.mean()),
        "reached_rate": float(reached.mean()),
        "collision_rate": float(collision.mean()),
        "y_goal_mean": float(np.nanmean(y_goal)) if reached.any() else None,
        "abs_y_goal_p50": float(np.nanpercentile(np.abs(y_goal), 50)) if reached.any() else None,
        "abs_y_goal_p90": float(np.nanpercentile(np.abs(y_goal), 90)) if reached.any() else None,
        "traces": traces,
    }


def field_metrics(net, sampler, device):
    # Shared branch states occur every two cycles: x=0,8,...,152.
    branch_x = np.arange(0.0, X_GOAL_MM, 8.0)
    branch_state = np.stack([branch_x, np.zeros_like(branch_x)], axis=1).astype(np.float32)
    branch_action = predict(net, branch_state, sampler, device)
    xx = np.repeat(np.linspace(0.0, 150.0, 31), 31)
    yy = np.tile(np.linspace(-6.0, 6.0, 31), 31)
    state = np.stack([xx, yy], axis=1).astype(np.float32)
    action = predict(net, state, sampler, device)
    clean = (-K_SERVO * yy) / LATERAL_ACTION_MM
    residual = action[:, 1] - clean
    return {
        "branch_state_lateral_action_mean": float(branch_action[:, 1].mean()),
        "branch_state_lateral_action_p10_p50_p90": [
            float(v) for v in np.percentile(branch_action[:, 1], [10, 50, 90])
        ],
        "field_lateral_residual_mean": float(residual.mean()),
        "first_forward_mean": float(action[:, 0].mean()),
    }


def run_cell(seed, args, device):
    state, action, data_meta = make_path_dataset(
        args.n_starts, args.repeats, args.horizon, args.data_seed)
    print(
        f"[seed={seed}] paths={data_meta['n_paths']} chunks={len(state)} "
        f"demo_SR={data_meta['demo_success_rate']:.3f}", flush=True)
    reg, reg_train = train("regression", state, action, seed, args.width,
                           args.steps, args.batch_size, args.lr, device)
    print(f"  regression trained {reg_train['seconds']:.1f}s", flush=True)
    mip, mip_train = train("mip", state, action, seed, args.width,
                           args.steps, args.batch_size, args.lr, device)
    print(f"  mip trained {mip_train['seconds']:.1f}s", flush=True)
    hg, hg_train = train_nll("hg", state, action, seed, args.width, args.steps,
                             args.batch_size, args.lr, device)
    print(f"  hg trained {hg_train['seconds']:.1f}s", flush=True)
    ht, ht_train = train_nll("ht", state, action, seed, args.width, args.steps,
                             args.batch_size, args.lr, device)
    print(f"  ht trained {ht_train['seconds']:.1f}s", flush=True)

    methods = {}
    for name, net, sampler in (
        ("regression", reg, "regression"),
        ("mip_step1", mip, "mip_step1"),
        ("mip_full", mip, "mip_full"),
        ("hg", hg, "hg"),
        ("ht", ht, "ht"),
    ):
        methods[name] = {
            **field_metrics(net, sampler, device),
            **rollout_metrics(net, sampler, device, args.n_eval),
        }
        m = methods[name]
        print(
            f"  {name:11s} SR={m['success_rate']:.3f} "
            f"branch_a={m['branch_state_lateral_action_mean']:+.4f} "
            f"dock={m['abs_y_goal_p50']}", flush=True)
    return {
        "seed": seed,
        "data": data_meta,
        "training": {
            "regression": reg_train, "mip": mip_train,
            "hg": hg_train, "ht": ht_train,
        },
        "methods": methods,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-starts", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-eval", type=int, default=401)
    ap.add_argument("--data-seed", type=int, default=20260805)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "analysis/toy2d_noisy_paths_results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    result = {
        "design": "executed-kick-plus-exact-next-cycle-compensation",
        "status": "trained-model-result",
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        "constants": {
            "x_goal_mm": X_GOAL_MM,
            "forward_action_mm": FORWARD_ACTION_MM,
            "lateral_action_mm": LATERAL_ACTION_MM,
            "dock_tol_mm": DOCK_TOL_MM,
            "anchor_sigma": ANCHOR_SIGMA,
            "branch_modes": [float(v) for v in BRANCH_MODES],
        },
        "cells": [],
    }
    for seed in seeds:
        result["cells"].append(run_cell(seed, args, device))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
