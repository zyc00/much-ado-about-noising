"""Clean behavioral slots cell: three valid docks, per-episode operator slot
choice (60/28/12), everything EXECUTED, zero injected corruption, wide corridor
so every lane is collision-free (fixes the original slots witness geometry,
whose +-5mm funnel would have killed the -6/-11 demonstrators).

All arms train on chunks extracted from the SAME episode dataset; the history
arm additionally conditions on the previous chunk. Success = reach x>=160 and
land within SLOT_TOL of any dock.

Usage: python scripts/toy2d_slots_dp.py --seeds 0 1 2 [--width 256 --steps 24000]
Env: SL_ARMS, SL_OUT, SL_TOL (default 1.0), SL_EPISODES (default 4000),
     OB_FLOW_K (flow ODE steps; default 16, use 64)
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
    ACTION_MM, FORWARD_MM, K_SERVO, X_GOAL_MM,
    normalize_state, predict, train, train_nll,
)
from toy2d_obstacle import (  # noqa: E402
    HistFlowNet, predict_flow, predict_flow_hist, train_flow,
)

SLOTS = np.array([0.0, -6.0, -11.0])       # lane centers (mm)
PROBS = np.array([0.60, 0.28, 0.12])
JITTER = np.array([0.08, 0.30, 0.50])      # per-episode aim jitter (mm)
TOL = float(os.environ.get("SL_TOL", "1.0"))
HORIZON = 8
MAX_CHUNKS = 70


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / X_GOAL_MM, 0.0, 1.0)


def gen_episodes(n_episodes, seed):
    """Executed demos: pick a slot per episode, servo to its lane, run to the
    goal. Returns chunk triples (state, prev_chunk, chunk) + oracle stats."""
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    n_ok = 0
    for _ in range(n_episodes):
        k = rng.choice(3, p=PROBS)
        lane = SLOTS[k] + float(np.clip(rng.normal(0.0, JITTER[k]),
                                        -0.9 * TOL, 0.9 * TOL))
        y = float(rng.uniform(-10.0, 10.0)); x = 0.0
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        collided = False
        while x < X_GOAL_MM:
            st = np.array([x, y], dtype=np.float32)
            ch = np.zeros(2 * HORIZON, dtype=np.float32)
            xx, yy = x, y
            for j in range(HORIZON):
                dy = -K_SERVO * (yy - lane)
                ch[2 * j] = FORWARD_MM / ACTION_MM
                ch[2 * j + 1] = dy / ACTION_MM
                yy += dy; xx += FORWARD_MM
                if abs(yy) > half_width(xx):
                    collided = True
            S.append(st); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = xx, yy
        if (not collided) and abs(y - SLOTS[np.argmin(np.abs(y - SLOTS))]) < TOL:
            n_ok += 1
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)),
            n_ok / n_episodes)


def train_flow_hist_ds(st, pv, ac, seed, width, steps, batch_size, lr, device):
    torch.manual_seed(seed)
    net = HistFlowNet(ac.shape[1], width).to(device)
    st, pv, ac = st.to(device), pv.to(device), ac.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(st)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        z0 = torch.randn_like(ac[idx])
        t = torch.rand(batch_size, 1, device=device)
        zt = (1 - t) * z0 + t * ac[idx]
        v = net(st[idx], pv[idx], zt, t)
        loss = ((v - (ac[idx] - z0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
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
        if sampler == "flow":
            a = predict_flow(net, st, device)
        elif sampler == "flow_hist":
            zt = predict_flow_hist(net, st, prev[ids], device)
            prev[ids] = zt
            a = zt.cpu().numpy()
        else:
            a = predict(net, st, sampler, device)
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
    reached = np.isfinite(y_goal)
    derr = np.min(np.abs(y_goal[:, None] - SLOTS[None, :]), axis=1)
    ok = reached & ~collision & (derr < TOL)
    with np.errstate(invalid="ignore"):
        slot_of = np.where(reached, np.argmin(np.abs(y_goal[:, None] - SLOTS[None, :]), axis=1), -1)
    counts = [int(((slot_of == k) & ok).sum()) for k in range(3)]
    return {"sr": float(ok.mean()), "collision": float(collision.mean()),
            "reached": float(reached.mean()),
            "landing_p50": float(np.nanmedian(y_goal)) if reached.any() else float("nan"),
            "dock_counts": counts,
            "traces": [{"x": v[0], "y": v[1]} for v in traces.values()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_eps = int(os.environ.get("SL_EPISODES", "4000"))

    arms = [("l2", "regression", "regression"), ("mip", "mip", "mip_full"),
            ("hg", "hg", "hg"), ("ht2", "ht", "ht"), ("ht05", "ht05", "ht"),
            ("flow", "flow", "flow"), ("flow_hist", "flow_hist", "flow_hist")]
    only = os.environ.get("SL_ARMS", "")
    if only:
        keep = set(only.split(","))
        arms = [a for a in arms if a[0] in keep]
    out = {"config": {"slots": SLOTS.tolist(), "probs": PROBS.tolist(),
                      "jitter": JITTER.tolist(), "tol": TOL, "n_episodes": n_eps},
           "arms": {}}
    for name, kind, sampler in arms:
        srs, cols, lands, docks = [], [], [], []
        seed0 = None
        for seed in args.seeds:
            st, pv, ac, osr = gen_episodes(n_eps, seed=1000 + seed)
            if seed == args.seeds[0] and name == arms[0][0]:
                print(f"oracle (executed demos): SR={osr:.3f}  chunks={len(st)}",
                      flush=True)
                out["oracle_sr"] = osr
            if kind == "flow_hist":
                net = train_flow_hist_ds(st, pv, ac, seed, args.width, args.steps,
                                         512, 1e-3, device)
            elif kind == "flow":
                net, _ = train_flow(st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            elif kind in ("hg", "ht", "ht05"):
                net, _ = train_nll("ht" if kind == "ht" else kind, st, ac, seed,
                                   args.width, args.steps, 512, 1e-3, device)
            else:
                net, _ = train(kind, st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            r = rollout(net, sampler, device)
            srs.append(r["sr"]); cols.append(r["collision"]); lands.append(r["landing_p50"])
            docks.append(r["dock_counts"])
            if seed == args.seeds[0]:
                seed0 = {"traces": r["traces"]}
        out["arms"][name] = {"sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)),
                             "sr_per_seed": srs, "collision_mean": float(np.mean(cols)),
                             "landing_p50s": lands, "dock_counts": docks, "seed0": seed0}
        print(f"{name:10s} SR={np.mean(srs):.3f}±{np.std(srs):.2f}  "
              f"collision={np.mean(cols):.3f}  landing_p50={np.nanmedian(lands):+.2f}  "
              f"docks={docks[0]}", flush=True)
    outp = os.environ.get("SL_OUT", "analysis/toy2d_slots_dp.json")
    Path(outp).write_text(json.dumps(out, indent=1))
    print("WROTE", outp, flush=True)


if __name__ == "__main__":
    main()
