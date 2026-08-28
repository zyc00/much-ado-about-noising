"""Spurious-mode toy: the third mechanism cell. A minority fraction P_SPUR of
RECORDED chunks are glitch chunks (stuck lateral axis: +KICK mm/step toward +y);
the demonstrator EXECUTED clean servo, so the oracle is 100% (recorded-only
corruption, same framing as the skew cells).

Predictions: HT rejects the minority mode (mass < 1/(nu+1)) and flies clean;
L2/HG absorb a persistent directional bias (equilibrium offset
p*KICK/((1-p)*K_SERVO), off the 0.5mm dock tolerance); diffusion faithfully
samples the glitch mode with probability p per replan and executes a fatal kick.

Usage: OB_FLOW_K=64 python scripts/toy2d_spurious.py --seeds 0 1 2 --width 256 --steps 24000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy2d_mip_nfl import (  # noqa: E402
    ACTION_MM, DOCK_TOL_MM, FORWARD_MM, K_SERVO, X_GOAL_MM,
    make_chunks, normalize_state, predict, train, train_nll,
)
from toy2d_obstacle import predict_flow, train_flow  # noqa: E402

P_SPUR = float(os.environ.get("SP_P", "0.25"))
KICK = float(os.environ.get("SP_KICK", "2.0"))     # mm/step, fixed +y direction
EXEC_STEPS = int(os.environ.get("SP_EXEC", "8"))   # steps executed per replan
HORIZON = 8
MAX_CHUNKS = 70


def half_width(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def make_ds(n_states, repeats, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, X_GOAL_MM - FORWARD_MM, size=n_states)
    hw = half_width(x)
    y = rng.uniform(-0.85 * hw, 0.85 * hw)
    base = np.stack([x, y], axis=1).astype(np.float32)
    state = np.repeat(base, repeats, axis=0)
    action = make_chunks(state, np.zeros(len(state)), HORIZON)  # clean servo chunks
    spur = rng.random(len(state)) < P_SPUR
    for j in range(HORIZON):  # glitch: forward normal, lateral stuck at +KICK
        action[spur, 2 * j + 1] = KICK / ACTION_MM
    order = rng.permutation(len(state))
    return (torch.from_numpy(normalize_state(state)[order]),
            torch.from_numpy(action[order]))


def rollout(net, sampler, device, n_eval=401):
    x = np.zeros(n_eval)
    y = np.linspace(-10.0, 10.0, n_eval)
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    tr_ids = set(np.linspace(0, n_eval - 1, 13, dtype=int))
    traces = {i: ([0.0], [float(y[i])]) for i in tr_ids}
    for _ in range(MAX_CHUNKS):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        st = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        if sampler == "flow":
            a = predict_flow(net, st, device)
        else:
            a = predict(net, st, sampler, device)
        for j in range(EXEC_STEPS):
            live = active[ids]
            x[ids] = np.where(live, x[ids] + np.clip(a[:, 2 * j] * ACTION_MM, -2.0, 8.0), x[ids])
            y[ids] = np.where(live, y[ids] + np.clip(a[:, 2 * j + 1] * ACTION_MM, -12.0, 12.0), y[ids])
            for ii in tr_ids:
                if active[ii]:
                    traces[ii][0].append(float(x[ii]))
                    traces[ii][1].append(float(y[ii]))
            bad = np.abs(y[ids]) > half_width(x[ids])
            collision[ids[bad & live]] = True
            reach = (x[ids] >= X_GOAL_MM) & live
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[(bad & live) | reach]] = False
    ok = np.isfinite(y_goal) & ~collision
    sr = float((ok & (np.abs(y_goal) < DOCK_TOL_MM)).mean())
    return {"sr": sr, "collision": float(collision.mean()),
            "reached": float(np.isfinite(y_goal).mean()),
            "landing_p50": float(np.nanmedian(y_goal)) if np.isfinite(y_goal).any() else float("nan"),
            "traces": [{"x": v[0], "y": v[1]} for v in traces.values()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"config: p_spur={P_SPUR} kick={KICK}mm/step exec={EXEC_STEPS} "
          f"(oracle 1.0 by construction: corruption is recorded-only)", flush=True)
    arms = [("l2", "regression", "regression"), ("mip", "mip", "mip_full"),
            ("hg", "hg", "hg"),
            ("ht2", "ht", "ht"), ("ht1", "ht1", "ht"), ("ht05", "ht05", "ht"),
            ("flow", "flow", "flow")]
    only = os.environ.get("SP_ARMS", "")
    if only:
        keep = set(only.split(","))
        arms = [a for a in arms if a[0] in keep]
    out = {"config": {"p_spur": P_SPUR, "kick": KICK, "exec": EXEC_STEPS}, "arms": {}}
    for name, kind, sampler in arms:
        srs, reach, cols, lands = [], [], [], []
        seed0 = None
        for seed in args.seeds:
            st, ac = make_ds(3000, 10, seed=1000 + seed)
            if kind in ("ht", "ht1", "ht05", "hg"):
                net, _ = train_nll("ht" if kind == "ht" else kind, st, ac, seed,
                                   args.width, args.steps, 512, 1e-3, device)
            elif kind == "flow":
                net, _ = train_flow(st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            else:
                net, _ = train(kind, st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            r = rollout(net, sampler, device)
            srs.append(r["sr"]); reach.append(r["reached"])
            cols.append(r["collision"]); lands.append(r["landing_p50"])
            if seed == args.seeds[0]:
                seed0 = {"traces": r["traces"]}
        out["arms"][name] = {"sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)),
                             "sr_per_seed": srs, "reached_mean": float(np.mean(reach)),
                             "collision_mean": float(np.mean(cols)),
                             "landing_p50s": lands, "seed0": seed0}
        print(f"{name:10s} SR={np.mean(srs):.3f}±{np.std(srs):.2f}  "
              f"reached={np.mean(reach):.3f}  collision={np.mean(cols):.3f}  "
              f"landing_p50={np.nanmedian(lands):+.2f}", flush=True)
    outp = os.environ.get("SP_OUT", "analysis/toy2d_spurious.json")
    Path(outp).write_text(json.dumps(out, indent=1))
    print("WROTE", outp, flush=True)


if __name__ == "__main__":
    main()
