"""Nuisance cell: the mean-family witness.

Each RECORDED chunk carries a zero-mean but SKEWED operator disturbance
(80% at -b, 20% at +4b, so E[d]=0 while the mode is -b). The demonstrator
EXECUTED clean servo, so the oracle is 100% by construction.

The correct action is the conditional MEAN (the disturbance is nuisance, not
structure), which splits the three families:
  MSE       averages the disturbance away          -> docks
  HT        commits to the disturbance's MODE (-b) -> constant lateral bias
  Diffusion reproduces the disturbance at execution -> noisy final approach

All three arms are history-conditioned (previous action chunk in the
conditioning) so the comparison is not confounded by conditioning.

Usage: OB_FLOW_K=64 python scripts/toy2d_nuisance.py --seeds 0 1 2 3
Env: NU_B (disturbance scale, default 0.05), NU_ARMS, NU_OUT
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
from toy2d_obstacle import (  # noqa: E402
    HistFlowNet, predict_flow_hist,
)

B = float(os.environ.get("NU_B", "0.05"))
HORIZON = 8
MAX_CHUNKS = 70


def half_width(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def draw_disturbance(rng, n):
    """Zero-mean, skewed: 80% at -B, 20% at +4B  =>  E[d] = 0."""
    u = rng.random(n)
    return np.where(u < 0.8, -B, 4.0 * B)


def gen_episodes(n_episodes, seed):
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    n_ok = 0
    for _ in range(n_episodes):
        y = float(rng.uniform(-10.0, 10.0)); x = 0.0
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        while x < X_GOAL_MM:
            st = np.array([x, y], dtype=np.float32)
            d = float(draw_disturbance(rng, 1)[0])
            ch = np.zeros(2 * HORIZON, dtype=np.float32)
            xx, yy = x, y
            for j in range(HORIZON):
                dy = -K_SERVO * yy                 # EXECUTED: clean servo
                ch[2 * j] = FORWARD_MM / ACTION_MM
                ch[2 * j + 1] = dy / ACTION_MM + d  # RECORDED: + nuisance
                yy += dy; xx += FORWARD_MM
            S.append(st); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = xx, yy
        if abs(y) < DOCK_TOL_MM:
            n_ok += 1
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)),
            n_ok / n_episodes)


def train_flow_hist_nu(st, pv, ac, seed, width, steps, batch_size, lr, device):
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
    y = np.linspace(-10.0, 10.0, n_eval)
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
        if sampler == "flow_hist":
            zt = predict_flow_hist(net, st, prev[ids], device)
        else:
            zt = predict_hist(net, st, prev[ids], device)
        prev[ids] = zt
        a = zt.cpu().numpy()
        for j in range(HORIZON):
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
    ok = np.isfinite(y_goal) & ~collision & (np.abs(y_goal) < DOCK_TOL_MM)
    return {"sr": float(ok.mean()), "collision": float(collision.mean()),
            "reached": float(np.isfinite(y_goal).mean()),
            "landing_p50": float(np.nanmedian(y_goal)) if np.isfinite(y_goal).any() else float("nan"),
            "traces": [{"x": v[0], "y": v[1]} for v in traces.values()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arms = [("l2", "regression", "reg_hist"), ("ht2", "ht", "reg_hist"),
            ("flow", "flow_hist", "flow_hist")]
    only = os.environ.get("NU_ARMS", "")
    if only:
        arms = [a for a in arms if a[0] in set(only.split(","))]
    out = {"config": {"b": B, "mix": "80% -b / 20% +4b (zero mean)"}, "arms": {}}
    for name, kind, sampler in arms:
        srs, lands, reach = [], [], []
        seed0 = None
        for seed in args.seeds:
            st, pv, ac, osr = gen_episodes(2500, seed=1000 + seed)
            if seed == args.seeds[0] and name == arms[0][0]:
                print(f"oracle (executed clean servo): SR={osr:.3f}", flush=True)
                out["oracle_sr"] = osr
            if kind == "flow_hist":
                net = train_flow_hist_nu(st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device)
            else:
                net = train_hist(kind, st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device)
            r = rollout(net, sampler, device)
            srs.append(r["sr"]); lands.append(r["landing_p50"]); reach.append(r["reached"])
            if seed == args.seeds[0]:
                seed0 = {"traces": r["traces"]}
        out["arms"][name] = {"sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)),
                             "sr_per_seed": srs, "reached_mean": float(np.mean(reach)),
                             "landing_p50s": lands, "seed0": seed0}
        print(f"{name:6s} SR={np.mean(srs):.3f}±{np.std(srs):.2f}  reached={np.mean(reach):.3f}  "
              f"landing_p50={np.nanmedian(lands):+.3f}mm", flush=True)
    Path(os.environ.get("NU_OUT", "analysis/toy2d_nuisance.json")).write_text(json.dumps(out, indent=1))
    print("WROTE", os.environ.get("NU_OUT", "analysis/toy2d_nuisance.json"), flush=True)


if __name__ == "__main__":
    main()
