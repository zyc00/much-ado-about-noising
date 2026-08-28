"""Transient-excursion cell: MSE vs flow matching.

The demonstrator drives a SQUARE WAVE in lateral y: at random points along the
corridor it steps off-centre by +-AMP, holds for a random stretch, then returns
to the centre line. Direction is up/down with probability P_UP (0.5 => the
detour distribution is zero mean). Pulses are EXECUTED, so the data is a clean
~100%-success demonstrator; nothing is injected.

Every pulse ENDS by x = CLEAN_X, so the last stretch is straight and every demo
docks at y = 0. Because the pulses return to centre on their own, the clean tail
can be SHORT without breaking the oracle — unlike the last-area cell, where the
tail had to be long enough for the demonstrator to re-centre.

  MSE   averages the up/down pulses -> drives the centre -> docks
  Flow  reproduces the pulse behaviour -> can start one in the final chunk and
        arrive off-centre, because "emit pulses with proportion p" is what the
        data shows and "you must arrive at the goal" is not

Usage: OB_FLOW_K=64 python scripts/toy2d_rectwave.py --seeds 0 1 2
Env: RW_AMP (mm, default 6), RW_PUP (default 0.5), RW_LEN/RW_LEN_JIT (pulse hold
     mm), RW_GAP/RW_GAP_JIT (mm between pulses), RW_CLEAN (default 140),
     RW_ARMS, RW_OUT
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

AMP = float(os.environ.get("RW_AMP", "6.0"))
P_UP = float(os.environ.get("RW_PUP", "0.5"))
LEN, LEN_JIT = float(os.environ.get("RW_LEN", "18.0")), float(os.environ.get("RW_LEN_JIT", "6.0"))
GAP, GAP_JIT = float(os.environ.get("RW_GAP", "22.0")), float(os.environ.get("RW_GAP_JIT", "10.0"))
CLEAN_X = float(os.environ.get("RW_CLEAN", "140.0"))
# Gain 1.0 reaches the aim in a single 4mm step, so the demonstrated path is an
# actual square wave (vertical edge, flat top) rather than an exponential ramp.
GAIN = float(os.environ.get("RW_GAIN", "1.0"))
START, START_JIT = float(os.environ.get("RW_START", "8.0")), float(os.environ.get("RW_START_JIT", "30.0"))
# Cell-local dock tolerance. The global 0.5mm sits ~1.5 sigma from a regression
# net's own landing error, so success flips on seed noise; 1.5mm keeps both
# families far from the boundary and makes SR report the mechanism instead.
DOCK = float(os.environ.get("RW_DOCK", "1.5"))
# Action parameterisation. Default (0) is per-step DELTAS, matching the VLA stacks
# (OFT/GR00T/Cosmos3 all use frame-wise relative actions); open-loop execution then
# integrates 8 increments, so per-step error compounds. RW_ABS=1 stores ABSOLUTE
# waypoints instead (x,y normalised to [-1,1]), where each step is set rather than
# accumulated and a per-step error cannot compound.
ABS = os.environ.get("RW_ABS", "0") == "1"
# Execution noise on the demonstrator (mm/step), UNPREDICTABLE from the state.
# Without it the demonstrator is deterministic given the state, the residual goes
# to ~0 on every non-deciding state, A*log(sigma) is unbounded below there, and
# sigma collapses to its floor — which stops HT's mu from committing. Real
# demonstrations always carry such noise (OFT heads sit at sigma ~0.52).
EXEC_NOISE = float(os.environ.get("RW_EXEC_NOISE", "0.0"))
# Control: blank the previous-chunk conditioning for every family, to separate
# "the loss commits to the wrong thing" from "the history slot drives a runaway".
NOHIST = os.environ.get("RW_NOHIST", "") == "1"
HORIZON, MAX_CHUNKS = 8, 70


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / X_GOAL_MM, 0.0, 1.0)


def sample_pulses(rng):
    """Square-wave detours at random positions; all of them end by CLEAN_X."""
    pulses, x = [], float(rng.uniform(START, START + START_JIT))
    while True:
        ln = LEN + rng.uniform(-LEN_JIT, LEN_JIT)
        if x + ln > CLEAN_X:
            break
        pulses.append((x, x + ln, 1.0 if rng.random() < P_UP else -1.0))
        x += ln + GAP + rng.uniform(-GAP_JIT, GAP_JIT)
    return pulses


def demo_aim(x, pulses):
    for x0, x1, side in pulses:
        if x0 <= x < x1:
            return side * AMP
    return 0.0


def gen_episodes(n_episodes, seed):
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    n_ok = 0
    for _ in range(n_episodes):
        pulses = sample_pulses(rng)
        y, x = float(rng.uniform(-8.0, 8.0)), 0.0
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        bad = False
        while x < X_GOAL_MM:
            st = np.array([x, y], dtype=np.float32)
            ch = np.zeros(2 * HORIZON, dtype=np.float32)
            xx, yy = x, y
            for j in range(HORIZON):
                dy = -GAIN * (yy - demo_aim(xx, pulses))
                if EXEC_NOISE > 0:
                    dy += rng.normal(0.0, EXEC_NOISE)
                yy += dy; xx += FORWARD_MM
                if ABS:
                    ch[2 * j] = xx / 80.0 - 1.0
                    ch[2 * j + 1] = yy / 14.0
                else:
                    ch[2 * j] = FORWARD_MM / ACTION_MM
                    ch[2 * j + 1] = (dy) / ACTION_MM
                if abs(yy) > half_width(xx):
                    bad = True
            S.append(st); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = xx, yy
        if (not bad) and abs(y) < DOCK:
            n_ok += 1
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)),
            n_ok / n_episodes)


def train_flow_hist_rw(st, pv, ac, seed, width, steps, batch_size, lr, device):
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


def rollout(net, sampler, device, n_eval=401, seed=0):
    x = np.zeros(n_eval)
    y = np.linspace(-8.0, 8.0, n_eval)
    active = np.ones(n_eval, dtype=bool)
    off = np.zeros(n_eval, dtype=bool)          # left the corridor
    y_goal = np.full(n_eval, np.nan)
    prev = torch.zeros(n_eval, 2 * HORIZON, device=device)
    keep_hist = not NOHIST
    tr_ids = set(np.linspace(0, n_eval - 1, 13, dtype=int))
    traces = {i: ([0.0], [float(y[i])]) for i in tr_ids}
    for _ in range(MAX_CHUNKS):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        st = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        zt = (predict_flow_hist(net, st, prev[ids], device) if sampler == "flow_hist"
              else predict_hist(net, st, prev[ids], device))
        if keep_hist:
            prev[ids] = zt
        a = zt.cpu().numpy()
        for j in range(HORIZON):
            live = active[ids]
            if ABS:
                x[ids] = np.where(live, (a[:, 2 * j] + 1.0) * 80.0, x[ids])
                y[ids] = np.where(live, np.clip(a[:, 2 * j + 1] * 14.0, -14.0, 14.0), y[ids])
            else:
                x[ids] = np.where(live, x[ids] + np.clip(a[:, 2 * j] * ACTION_MM, -2.0, 8.0), x[ids])
                y[ids] = np.where(live, y[ids] + np.clip(a[:, 2 * j + 1] * ACTION_MM, -12.0, 12.0), y[ids])
            for ii in tr_ids:
                if active[ii]:
                    traces[ii][0].append(float(x[ii])); traces[ii][1].append(float(y[ii]))
            for i in ids:
                if active[i] and abs(y[i]) > half_width(x[i]):
                    off[i] = True; active[i] = False
            reach = (x[ids] >= X_GOAL_MM) & active[ids]
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[reach]] = False
    ok = np.isfinite(y_goal) & (np.abs(y_goal) < DOCK)
    return {"sr": float(ok.mean()), "off_corridor": float(off.mean()),
            "reached": float(np.isfinite(y_goal).mean()),
            "landing_absmed": float(np.nanmedian(np.abs(y_goal))) if np.isfinite(y_goal).any() else float("nan"),
            "landing_p50": float(np.nanmedian(y_goal)) if np.isfinite(y_goal).any() else float("nan"),
            "traces": [{"x": v[0], "y": v[1]} for v in traces.values()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"amp {AMP}mm, p_up {P_UP}, pulse {LEN}+-{LEN_JIT}mm, gap {GAP}+-{GAP_JIT}mm, "
          f"straight tail from x={CLEAN_X} ({X_GOAL_MM - CLEAN_X:.0f}mm)", flush=True)

    arms = [("l2", "regression", "reg"), ("flow", "flow", "flow_hist"), ("ht2", "ht", "reg")]
    only = os.environ.get("RW_ARMS", "")
    if only:
        arms = [a for a in arms if a[0] in set(only.split(","))]
    out = {"config": {"amp": AMP, "p_up": P_UP, "len": LEN, "len_jit": LEN_JIT,
                      "gap": GAP, "gap_jit": GAP_JIT, "clean_x": CLEAN_X}, "arms": {}}
    for name, kind, sampler in arms:
        srs, lands, seed0 = [], [], None
        for seed in args.seeds:
            st, pv, ac, osr = gen_episodes(2500, seed=1000 + seed)
            if seed == args.seeds[0] and name == arms[0][0]:
                print(f"oracle (executed square-wave demos): SR={osr:.3f}", flush=True)
                out["oracle_sr"] = osr
            if NOHIST:
                pv = torch.zeros_like(pv)
            net = (train_flow_hist_rw(st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device)
                   if kind == "flow" else
                   train_hist(kind, st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device))
            r = rollout(net, sampler, device, seed=seed)
            srs.append(r["sr"]); lands.append(r["landing_absmed"])
            if seed == args.seeds[0]:
                seed0 = {"traces": r["traces"]}
        out["arms"][name] = {"sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)),
                             "sr_per_seed": srs, "landing_absmed": float(np.nanmedian(lands)),
                             "seed0": seed0}
        print(f"{name:6s} SR={np.mean(srs):.3f}±{np.std(srs):.2f}  "
              f"|landing|_med={np.nanmedian(lands):.2f}mm", flush=True)
    Path(os.environ.get("RW_OUT", "analysis/toy2d_rectwave.json")).write_text(json.dumps(out, indent=1))
    print("WROTE", os.environ.get("RW_OUT", "analysis/toy2d_rectwave.json"), flush=True)


if __name__ == "__main__":
    main()
