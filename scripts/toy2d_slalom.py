"""Two-plate slot slalom: compounding commitment churn. Each of two wall
plates has three gaps (lanes 0/-6/-11mm); the demonstrator picks a lane PER
PLATE (60/28/12, independent), servos through, and docks at center. All
executed, zero injection, oracle 100%. Failures are TERMINAL (plate collision
or off-dock landing) — infinite time does not help.

Per-plate DP churn was measured at ~0.5 on the single-plate slots cell; two
plates should compound it (~0.25) while HT nu=0.5 re-commits to the plurality
lane at each plate.

Usage: OB_FLOW_K=64 python scripts/toy2d_slalom.py --seeds 0 1 2 --width 256 --steps 24000
Env: SLM_ARMS, SLM_OUT
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
    normalize_state, predict, predict_hist, train, train_hist, train_nll,
)
from toy2d_obstacle import (  # noqa: E402
    HistFlowNet, predict_flow, predict_flow_hist, train_flow,
)

SLOTS = np.array([0.0, -6.0, -11.0])
# Majority mass matters: HT at nu=2 commits only when the OPPOSING mass is below
# 1/(nu+1) = 1/3. At the old [0.60, 0.28, 0.12] the opposing mass is 0.40, just
# outside, and nu=2 goes bimodal across seeds. SLM_PROBS moves the task, not the
# loss, so nu=2 stays the fixed default in every panel.
PROBS = np.array([float(v) for v in os.environ.get("SLM_PROBS", "0.70,0.20,0.10").split(",")])
JITTER = np.array([0.08, 0.30, 0.50])
GAP = 1.2                      # gap half-height in the plates (mm)
PLATES = [(50.0, 58.0), (100.0, 108.0)]
SWITCH = [62.0, 112.0]         # servo target switches after clearing each plate
DOCK_TOL = 1.0
# Execution noise on the demonstrator (mm/step), UNPREDICTABLE from the state.
# Without it the demonstrator is deterministic given the state, the residual goes
# to ~0 on every non-deciding state, A*log(sigma) is unbounded below there, and
# sigma collapses to its floor — which stops HT's mu from committing. Real
# demonstrations always carry such noise (OFT heads sit at sigma ~0.52).
EXEC_NOISE = float(os.environ.get("SLM_EXEC_NOISE", "0.0"))
HORIZON = 8
MAX_CHUNKS = 70


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / X_GOAL_MM, 0.0, 1.0)


def hits_plate(x, y):
    out = np.zeros_like(np.atleast_1d(x), dtype=bool)
    xx, yy = np.atleast_1d(x), np.atleast_1d(y)
    for x0, x1 in PLATES:
        inside = (xx >= x0) & (xx <= x1)
        gap_ok = np.min(np.abs(yy[:, None] - SLOTS[None, :]), axis=1) < GAP
        out |= inside & ~gap_ok
    return out


def gen_episodes(n_episodes, seed):
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    n_ok = 0
    for _ in range(n_episodes):
        k1, k2 = rng.choice(3, p=PROBS), rng.choice(3, p=PROBS)
        lane1 = SLOTS[k1] + float(np.clip(rng.normal(0, JITTER[k1]), -0.9, 0.9))
        lane2 = SLOTS[k2] + float(np.clip(rng.normal(0, JITTER[k2]), -0.9, 0.9))
        y = float(rng.uniform(-10.0, 10.0))
        x = float(rng.uniform(0.0, FORWARD_MM * HORIZON))
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        collided = False
        while x < X_GOAL_MM:
            st = np.array([x, y], dtype=np.float32)
            ch = np.zeros(2 * HORIZON, dtype=np.float32)
            xx, yy = x, y
            for j in range(HORIZON):
                lane = lane1 if xx < SWITCH[0] else (lane2 if xx < SWITCH[1] else 0.0)
                dy = -K_SERVO * (yy - lane)
                if EXEC_NOISE > 0:
                    dy += rng.normal(0.0, EXEC_NOISE)
                ch[2 * j] = FORWARD_MM / ACTION_MM
                ch[2 * j + 1] = dy / ACTION_MM
                yy += dy; xx += FORWARD_MM
                if abs(yy) > half_width(xx) or hits_plate(xx, yy)[0]:
                    collided = True
            S.append(st); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = xx, yy
        if (not collided) and abs(y) < DOCK_TOL:
            n_ok += 1
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)),
            n_ok / n_episodes)


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
        elif sampler == "hist":
            zt = predict_hist(net, st, prev[ids], device)
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
            bad = (np.abs(y[ids]) > half_width(x[ids])) | hits_plate(x[ids], y[ids])
            collision[ids[bad & live]] = True
            reach = (x[ids] >= X_GOAL_MM) & live
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[(bad & live) | reach]] = False
    ok = np.isfinite(y_goal) & ~collision & (np.abs(y_goal) < DOCK_TOL)
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

    arms = [("l2", "regression", "regression"), ("mip", "mip", "mip_full"),
            ("ht2", "ht", "ht"), ("ht05", "ht05", "ht"),
            ("flow", "flow", "flow"), ("flow_hist", "flow_hist", "flow_hist"),
            ("l2_hist", "reg_hist", "hist"), ("ht2_hist", "ht_hist", "hist"),
            ("ht05_hist", "ht05_hist", "hist")]
    only = os.environ.get("SLM_ARMS", "")
    if only:
        keep = set(only.split(","))
        arms = [a for a in arms if a[0] in keep]
    out = {"config": {"slots": SLOTS.tolist(), "probs": PROBS.tolist(), "gap": GAP,
                      "plates": PLATES, "dock_tol": DOCK_TOL}, "arms": {}}
    for name, kind, sampler in arms:
        srs, cols, lands = [], [], []
        seed0 = None
        for seed in args.seeds:
            st, pv, ac, osr = gen_episodes(4000, seed=1000 + seed)
            if seed == args.seeds[0] and name == arms[0][0]:
                print(f"oracle (executed demos): SR={osr:.3f}  chunks={len(st)}", flush=True)
                out["oracle_sr"] = osr
            if kind == "flow_hist":
                torch.manual_seed(seed)
                net = HistFlowNet(ac.shape[1], args.width).to(device)
                stt, pvv, acc = st.to(device), pv.to(device), ac.to(device)
                opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
                net.train()
                for _ in range(args.steps):
                    idx = torch.randint(0, len(stt), (512,), device=device)
                    z0 = torch.randn_like(acc[idx])
                    t = torch.rand(512, 1, device=device)
                    zt = (1 - t) * z0 + t * acc[idx]
                    loss = ((net(stt[idx], pvv[idx], zt, t) - (acc[idx] - z0)) ** 2).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                net.eval()
            elif kind in ("reg_hist", "ht_hist", "ht05_hist"):
                # same episode data + previous chunk in the conditioning as flow_hist
                net = train_hist({"reg_hist": "regression", "ht_hist": "ht"}.get(kind, "ht05"),
                                 st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device)
            elif kind == "flow":
                net, _ = train_flow(st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            elif kind in ("ht", "ht05"):
                net, _ = train_nll("ht" if kind == "ht" else kind, st, ac, seed,
                                   args.width, args.steps, 512, 1e-3, device)
            else:
                net, _ = train(kind, st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            r = rollout(net, sampler, device)
            srs.append(r["sr"]); cols.append(r["collision"]); lands.append(r["landing_p50"])
            if seed == args.seeds[0]:
                seed0 = {"traces": r["traces"]}
        out["arms"][name] = {"sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)),
                             "sr_per_seed": srs, "collision_mean": float(np.mean(cols)),
                             "landing_p50s": lands, "seed0": seed0}
        print(f"{name:10s} SR={np.mean(srs):.3f}±{np.std(srs):.2f}  "
              f"collision={np.mean(cols):.3f}  landing_p50={np.nanmedian(lands):+.2f}", flush=True)
    outp = os.environ.get("SLM_OUT", "analysis/toy2d_slalom.json")
    Path(outp).write_text(json.dumps(out, indent=1))
    print("WROTE", outp, flush=True)


if __name__ == "__main__":
    main()
