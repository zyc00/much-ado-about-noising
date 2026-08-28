"""Periodic-detour cell: 'go around a block every 3 cm — except the last 3 cm'.

A block sits in the middle of the corridor every PERIOD mm (default 30 = 3 cm).
At each one the demonstrator detours up or down (60/40), then returns toward the
centre. After the LAST block the corridor is clear and every demo runs straight
to the dock, so the final approach carries no detour signal and docking is not
confounded by the pattern.

Why this shape: each block is an independent binary commitment, so a sampler
that redraws its intention at every replan compounds its per-block failure,
while a committing estimator pays the cost once. The clean tail means the
ending cannot be used as a cue.

Everything is EXECUTED (no injected corruption) and the oracle is ~100%.

Usage: OB_FLOW_K=64 python scripts/toy2d_periodic.py --seeds 0 1 2
Env: PD_PERIOD (mm between blocks, default 30), PD_NBLOCK (default 4),
     PD_PMAJ (up-probability, default 0.6), PD_ARMS, PD_OUT
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
    normalize_state, predict_hist, train_hist,
)
from toy2d_obstacle import HistFlowNet, predict_flow_hist  # noqa: E402

PERIOD = float(os.environ.get("PD_PERIOD", "30.0"))
NBLOCK = int(os.environ.get("PD_NBLOCK", "4"))
P_MAJ = float(os.environ.get("PD_PMAJ", "0.6"))
BLK_HALF, BLK_LEN, DETOUR = 3.0, 8.0, 6.0
HORIZON, MAX_CHUNKS = 8, 70

# blocks at 20, 50, 80, 110 ... ; the tail after the last block stays clear
BLOCKS = [(20.0 + i * PERIOD, 20.0 + i * PERIOD + BLK_LEN) for i in range(NBLOCK)]
TAIL_X = BLOCKS[-1][1]          # detours stop here: the last stretch is clean


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / X_GOAL_MM, 0.0, 1.0)


def in_block(x, y):
    out = np.zeros_like(np.atleast_1d(x), dtype=bool)
    xx, yy = np.atleast_1d(x), np.atleast_1d(y)
    for x0, x1 in BLOCKS:
        out |= (xx >= x0) & (xx <= x1) & (np.abs(yy) < BLK_HALF)
    return out


def aim_for(x, branches):
    """Detour target while approaching each block; 0 in the clean tail."""
    if x >= TAIL_X:
        return 0.0
    for i, (x0, x1) in enumerate(BLOCKS):
        if x < x1:                      # heading for block i
            lead = x0 - PERIOD * 0.55   # commit this far ahead
            return branches[i] * DETOUR if x >= lead else 0.0
    return 0.0


def gen_episodes(n_episodes, seed):
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    n_ok = 0
    for _ in range(n_episodes):
        branches = [1.0 if rng.random() < P_MAJ else -1.0 for _ in range(NBLOCK)]
        y = float(rng.uniform(-9.0, 9.0)); x = 0.0
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        bad = False
        while x < X_GOAL_MM:
            st = np.array([x, y], dtype=np.float32)
            ch = np.zeros(2 * HORIZON, dtype=np.float32)
            xx, yy = x, y
            for j in range(HORIZON):
                yt = aim_for(xx, branches)
                dy = -K_SERVO * (yy - yt)
                ch[2 * j] = FORWARD_MM / ACTION_MM
                ch[2 * j + 1] = dy / ACTION_MM
                yy += dy; xx += FORWARD_MM
                if abs(yy) > half_width(xx) or in_block(xx, yy)[0]:
                    bad = True
            S.append(st); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = xx, yy
        if (not bad) and abs(y) < DOCK_TOL_MM:
            n_ok += 1
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)),
            n_ok / n_episodes)


def train_flow_hist_pd(st, pv, ac, seed, width, steps, batch_size, lr, device):
    torch.manual_seed(seed)
    net = HistFlowNet(ac.shape[1], width).to(device)
    st, pv, ac = st.to(device), pv.to(device), ac.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, len(st), (batch_size,), device=device)
        z0 = torch.randn_like(ac[idx])
        t = torch.rand(batch_size, 1, device=device)
        zt = (1 - t) * z0 + t * ac[idx]
        loss = ((net(st[idx], pv[idx], zt, t) - (ac[idx] - z0)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    return net


def rollout(net, sampler, device, n_eval=401):
    x = np.zeros(n_eval)
    y = np.linspace(-9.0, 9.0, n_eval)
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    prev = torch.zeros(n_eval, 2 * HORIZON, device=device)
    tr_ids = set(np.linspace(0, n_eval - 1, 13, dtype=int))
    traces = {i: ([0.0], [float(y[i])]) for i in tr_ids}
    for _ in range(MAX_CHUNKS):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        st = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        zt = (predict_flow_hist(net, st, prev[ids], device) if sampler == "flow_hist"
              else predict_hist(net, st, prev[ids], device))
        prev[ids] = zt
        a = zt.cpu().numpy()
        for j in range(HORIZON):
            live = active[ids]
            x[ids] = np.where(live, x[ids] + np.clip(a[:, 2 * j] * ACTION_MM, -2.0, 8.0), x[ids])
            y[ids] = np.where(live, y[ids] + np.clip(a[:, 2 * j + 1] * ACTION_MM, -12.0, 12.0), y[ids])
            for ii in tr_ids:
                if active[ii]:
                    traces[ii][0].append(float(x[ii])); traces[ii][1].append(float(y[ii]))
            bad = (np.abs(y[ids]) > half_width(x[ids])) | in_block(x[ids], y[ids])
            collision[ids[bad & live]] = True
            reach = (x[ids] >= X_GOAL_MM) & live
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[(bad & live) | reach]] = False
    ok = np.isfinite(y_goal) & ~collision & (np.abs(y_goal) < DOCK_TOL_MM)
    return {"sr": float(ok.mean()), "collision": float(collision.mean()),
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
    print(f"blocks at {[f'{a:.0f}-{b:.0f}' for a, b in BLOCKS]}; clean tail from x={TAIL_X:.0f}", flush=True)

    arms = [("l2", "regression", "reg"), ("ht2", "ht", "reg"), ("flow", "flow", "flow_hist")]
    only = os.environ.get("PD_ARMS", "")
    if only:
        arms = [a for a in arms if a[0] in set(only.split(","))]
    out = {"config": {"period": PERIOD, "n_block": NBLOCK, "p_maj": P_MAJ,
                      "blocks": BLOCKS, "tail_x": TAIL_X}, "arms": {}}
    for name, kind, sampler in arms:
        srs, cols, seed0 = [], [], None
        for seed in args.seeds:
            st, pv, ac, osr = gen_episodes(3000, seed=1000 + seed)
            if seed == args.seeds[0] and name == arms[0][0]:
                print(f"oracle (executed demos): SR={osr:.3f}", flush=True)
                out["oracle_sr"] = osr
            net = (train_flow_hist_pd(st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device)
                   if kind == "flow" else
                   train_hist(kind, st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device))
            r = rollout(net, sampler, device)
            srs.append(r["sr"]); cols.append(r["collision"])
            if seed == args.seeds[0]:
                seed0 = {"traces": r["traces"]}
        out["arms"][name] = {"sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)),
                             "sr_per_seed": srs, "collision_mean": float(np.mean(cols)),
                             "seed0": seed0}
        print(f"{name:6s} SR={np.mean(srs):.3f}±{np.std(srs):.2f}  collision={np.mean(cols):.3f}", flush=True)
    Path(os.environ.get("PD_OUT", "analysis/toy2d_periodic.json")).write_text(json.dumps(out, indent=1))
    print("WROTE", os.environ.get("PD_OUT", "analysis/toy2d_periodic.json"), flush=True)


if __name__ == "__main__":
    main()
