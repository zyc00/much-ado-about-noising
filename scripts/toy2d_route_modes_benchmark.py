"""Continuously valid route-mode paths designed for HT > MIP/L2.

All demonstrations go around a central obstacle and reach the same final dock.
At the shared branch state the route coefficient is -1 (70%), +1 (20%), or
+5 (10%), so its conditional mean is exactly zero.  L2 therefore takes the
blocked center.  HT's learned-scale Student-t objective should choose the
dense -1 route.  MIP's population denoiser at the mean retains a partial route
coefficient near (-.7 + .2)/(.7 + .2) = -.556, which remains in the blocked
band when the clearance is placed between the partial and full routes.

The decision obstacle is a thin vertical gate.  Collision is evaluated where
each executed line segment crosses the gate, rather than only at sampled
replanning endpoints, so every stored demonstration is a physically
collectable collision-free polyline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toy2d_mip_nfl import T_TWO_STEP, train, train_nll


ROOT = Path(__file__).resolve().parents[1]
X_GOAL_MM = 160.0
FORWARD_MM = 4.0
FORWARD_ACTION_MM = 10.0
LATERAL_ACTION_MM = 2.0
DOCK_TOL_MM = 0.5
N_CYCLES = 40
ROUTE_AMPLITUDE_MM = 0.8
GATE_X_MM = 3.0
GATE_HALF_MM = 0.20
ROUTE_MODES = np.array([-1.0] * 7 + [1.0] * 2 + [5.0], dtype=np.float32)


def half_width_mm(x):
    x = np.asarray(x)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def normalize_state(state):
    out = np.empty_like(state, dtype=np.float32)
    out[:, 0] = state[:, 0] / 80.0 - 1.0
    out[:, 1] = state[:, 1] / 14.0
    return out


def route_profile(n):
    p = np.zeros(n, dtype=np.float64)
    p[1:4] = [0.50, 0.85, 1.00]
    p[4:11] = 1.0
    p[11:15] = [0.75, 0.50, 0.25, 0.0]
    return p


def segment_collision_mask(x0, y0, x1, y1):
    """Collision for a complete executed segment, including gate crossing."""
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    wall = ((np.abs(y0) > half_width_mm(x0))
            | (np.abs(y1) > half_width_mm(x1)))
    crosses = (x0 < GATE_X_MM) & (x1 >= GATE_X_MM)
    denom = np.maximum(x1 - x0, 1e-12)
    alpha = np.clip((GATE_X_MM - x0) / denom, 0.0, 1.0)
    y_cross = y0 + alpha * (y1 - y0)
    gate = crosses & (np.abs(y_cross) < GATE_HALF_MM)
    return wall | gate


def make_dataset(n_starts, repeats, horizon, branch_dup, seed):
    if repeats != 10:
        raise ValueError("repeats must be 10 for exact 70/20/10 routes")
    rng = np.random.default_rng(seed)
    y0_unique = rng.uniform(-0.10, 0.10, size=n_starts)
    y0 = np.repeat(y0_unique, repeats)
    n_paths = len(y0)
    mode = np.empty((n_starts, repeats), dtype=np.float32)
    for i in range(n_starts):
        mode[i] = rng.permutation(ROUTE_MODES)
    mode = mode.reshape(-1).astype(np.float64)

    total = N_CYCLES + horizon
    p = route_profile(total + 1)
    t = np.arange(total + 1)
    baseline = y0[:, None] * (0.5 ** t[None, :])
    y = baseline + ROUTE_AMPLITUDE_MM * mode[:, None] * p[None, :]
    x = FORWARD_MM * t
    dy = np.diff(y, axis=1)
    actions = np.empty((n_paths, total, 2), dtype=np.float32)
    actions[:, :, 0] = FORWARD_MM / FORWARD_ACTION_MM
    actions[:, :, 1] = dy / LATERAL_ACTION_MM
    states = np.empty((n_paths, N_CYCLES, 2), dtype=np.float32)
    states[:, :, 0] = x[:N_CYCLES][None, :]
    states[:, :, 1] = y[:, :N_CYCLES]
    chunks = np.stack(
        [actions[:, k:k + horizon].reshape(n_paths, 2 * horizon)
         for k in range(N_CYCLES)], axis=1)

    path_collision = np.any(
        segment_collision_mask(
            x[None, :N_CYCLES], y[:, :N_CYCLES],
            x[None, 1:N_CYCLES + 1], y[:, 1:N_CYCLES + 1]),
        axis=1,
    )
    gate_alpha = GATE_X_MM / FORWARD_MM
    y_at_gate = y[:, 0] + gate_alpha * (y[:, 1] - y[:, 0])
    success = (~path_collision) & (np.abs(y[:, N_CYCLES]) < DOCK_TOL_MM)
    flat_state = normalize_state(states.reshape(-1, 2))
    flat_action = chunks.reshape(-1, 2 * horizon)
    if branch_dup > 1:
        # Decision-balanced path sampling: the one shared branch state would
        # otherwise be only 1/40 of the dataset.  Duplicate it so branch-choice
        # risk and route-continuation risk receive comparable training mass.
        branch_state = normalize_state(states[:, 0])
        branch_action = chunks[:, 0]
        flat_state = np.concatenate(
            [flat_state, np.repeat(branch_state, branch_dup - 1, axis=0)], axis=0)
        flat_action = np.concatenate(
            [flat_action, np.repeat(branch_action, branch_dup - 1, axis=0)], axis=0)
    order = rng.permutation(len(flat_state))
    chosen = np.linspace(0, n_paths - 1, 15, dtype=int)
    paths = [{
        "mode": float(mode[i]),
        "x": [float(v) for v in x[:N_CYCLES + 1]],
        "y": [float(v) for v in y[i, :N_CYCLES + 1]],
    } for i in chosen]
    meta = {
        "n_paths": n_paths,
        "n_chunks": int(len(flat_state)),
        "branch_dup": branch_dup,
        "demo_success_rate": float(success.mean()),
        "demo_collision_rate": float(path_collision.mean()),
        "demo_min_gate_clearance_mm": float(np.abs(y_at_gate).min()),
        "paths": paths,
    }
    return (torch.from_numpy(flat_state[order]),
            torch.from_numpy(flat_action[order]), meta)


@torch.inference_mode()
def predict(net, state_world, sampler, device):
    state = torch.from_numpy(normalize_state(state_world.astype(np.float32))).to(device)
    A = net.action_dim
    z = torch.zeros((len(state), A), device=device)
    t0 = torch.zeros((len(state), 1), device=device)
    raw = net(state, z, t0)
    if sampler in ("hg", "ht"):
        return raw[:, :A].cpu().numpy()
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
    y = np.linspace(-0.10, 0.10, n_eval)
    y0 = y.copy()
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    traces_x = [[] for _ in range(n_eval)]
    traces_y = [[] for _ in range(n_eval)]
    for _ in range(70):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        state = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        action = predict(net, state, sampler, device)
        dx = np.clip(action[:, 0] * FORWARD_ACTION_MM, 0.5, 8.0)
        dy_exec = np.clip(action[:, 1] * LATERAL_ACTION_MM, -6.0, 6.0)
        x_prev = x[ids].copy()
        y_prev = y[ids].copy()
        x[ids] += dx
        y[ids] += dy_exec
        for local, idx in enumerate(ids):
            traces_x[idx].append(float(x[idx]))
            traces_y[idx].append(float(y[idx]))
        hit = segment_collision_mask(x_prev, y_prev, x[ids], y[ids])
        collision[ids[hit]] = True
        reached = x[ids] >= X_GOAL_MM
        y_goal[ids[reached]] = y[ids[reached]]
        active[ids[hit | reached]] = False
    reached = np.isfinite(y_goal)
    success = reached & ~collision & (np.abs(y_goal) < DOCK_TOL_MM)
    chosen = np.linspace(0, n_eval - 1, 11, dtype=int)
    traces = [{"y0": float(y0[i]), "x": traces_x[i], "y": traces_y[i]}
              for i in chosen]
    return {
        "success_rate": float(success.mean()),
        "reached_rate": float(reached.mean()),
        "collision_rate": float(collision.mean()),
        "abs_y_goal_p50": (float(np.nanpercentile(np.abs(y_goal), 50))
                            if reached.any() else None),
        "traces": traces,
    }


def branch_metrics(net, sampler, device):
    y = np.linspace(-0.10, 0.10, 101)
    state = np.stack([np.zeros_like(y), y], axis=1).astype(np.float32)
    action = predict(net, state, sampler, device)
    # First profile increment is +0.5, so full lower-route first action is
    # -0.5*A / lateral-scale = -0.2 normalized units.
    return {
        "first_lateral_action_mean": float(action[:, 1].mean()),
        "first_lateral_action_p10_p50_p90": [
            float(v) for v in np.percentile(action[:, 1], [10, 50, 90])
        ],
    }


def run_cell(seed, args, device):
    state, action, data = make_dataset(args.n_starts, args.repeats,
                                       args.horizon, args.branch_dup,
                                       args.data_seed)
    print(f"[seed={seed}] paths={data['n_paths']} demoSR={data['demo_success_rate']:.3f}",
          flush=True)
    reg, reg_info = train("regression", state, action, seed, args.width,
                          args.steps, args.batch_size, args.lr, device)
    mip, mip_info = train("mip", state, action, seed, args.width,
                          args.steps, args.batch_size, args.lr, device)
    hg, hg_info = train_nll("hg", state, action, seed, args.width, args.steps,
                            args.batch_size, args.lr, device)
    ht, ht_info = train_nll("ht", state, action, seed, args.width, args.steps,
                            args.batch_size, args.lr, device)
    methods = {}
    for name, net, sampler in (
        ("regression", reg, "regression"),
        ("hg", hg, "hg"),
        ("mip_step1", mip, "mip_step1"),
        ("mip_full", mip, "mip_full"),
        ("ht", ht, "ht"),
    ):
        methods[name] = {**branch_metrics(net, sampler, device),
                         **rollout_metrics(net, sampler, device, args.n_eval)}
        m = methods[name]
        print(f"  {name:11s} SR={m['success_rate']:.3f} "
              f"collision={m['collision_rate']:.3f} "
              f"first_ay={m['first_lateral_action_mean']:+.3f}", flush=True)
    checkpoint = None
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = args.checkpoint_dir / f"seed_{seed}.pt"

        def cpu_state_dict(net):
            return {k: v.detach().cpu() for k, v in net.state_dict().items()}

        torch.save({
            "format": "toy2d-route-modes-v1",
            "seed": seed,
            "action_dim": int(action.shape[1]),
            "width": args.width,
            "models": {
                "regression": cpu_state_dict(reg),
                "mip": cpu_state_dict(mip),
                "hg": cpu_state_dict(hg),
                "ht": cpu_state_dict(ht),
            },
            "metrics": methods,
        }, checkpoint)
        print(f"  checkpoint {checkpoint}", flush=True)
    return {
        "seed": seed, "data": data,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "training": {"regression": reg_info, "mip": mip_info,
                     "hg": hg_info, "ht": ht_info},
        "methods": methods,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-starts", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--branch-dup", type=int, default=40)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-eval", type=int, default=401)
    ap.add_argument("--data-seed", type=int, default=20260805)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "analysis/toy2d_route_modes_swept_gate_results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    result = {
        "design": "successful-route-modes-through-swept-segment-gate",
        "status": "trained-model-result",
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        "constants": {
            "route_modes": [float(v) for v in ROUTE_MODES],
            "route_amplitude_mm": ROUTE_AMPLITUDE_MM,
            "gate_x_mm": GATE_X_MM,
            "gate_half_mm": GATE_HALF_MM,
            "nominal_blocked_first_action_half": (
                GATE_HALF_MM * FORWARD_MM / GATE_X_MM / LATERAL_ACTION_MM
            ),
            "dock_tol_mm": DOCK_TOL_MM,
            "population_mip_route_coefficient_approx": -5.0 / 9.0,
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
