"""Unobserved-obstacle cell: MSE vs flow matching.

Each episode places rectangles along the corridor, randomly ABOVE or BELOW the
centre line (50/50). A rectangle above blocks y in [+1, +9]; below blocks
y in [-9, -1].  ==> the centre line y ~ 0 is ALWAYS free, whichever side it is on.

The demonstrator SEES the rectangles and gives each a wide berth (aims to the
free side, +-BERTH). The policy does NOT: its observation is only (x, y), so the
layout is unobserved context. That makes the two families diverge:

  MSE   averages the two detours -> drives the always-free centre line -> safe
  Flow  reproduces the detour DISTRIBUTION -> picks a side at random -> about
        half the time that is the blocked side -> collision, compounding per
        rectangle

The last stretch (x >= CLEAN_X) has no rectangles, so demos re-centre and dock.

Everything is executed; no injected corruption; oracle ~100% by construction.

Usage: OB_FLOW_K=64 python scripts/toy2d_rects.py --seeds 0 1 2
Env: RC_N (rects per episode, default 4), RC_BERTH (default 5mm),
     RC_PUP (prob. the rect is above, default 0.5), RC_CLEAN (default 130),
     RC_ARMS, RC_OUT
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

N_RECT = int(os.environ.get("RC_N", "4"))
BERTH = float(os.environ.get("RC_BERTH", "5.0"))
P_UP = float(os.environ.get("RC_PUP", "0.5"))
CLEAN_X = float(os.environ.get("RC_CLEAN", "130.0"))
RECT_LEN = 10.0
INNER = float(os.environ.get("RC_INNER", "1.0"))   # blocks y in [INNER, OUTER] on its side
OUTER = float(os.environ.get("RC_OUTER", "9.0"))   # ==> |y| < INNER is free either way
LEAD = 14.0                                  # how far ahead the demo starts its berth
HORIZON, MAX_CHUNKS = 8, 70


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / X_GOAL_MM, 0.0, 1.0)


def sample_layout(rng):
    """Rect x-positions spread over the pre-clean stretch, each above or below."""
    xs = np.linspace(25.0, CLEAN_X - 20.0, N_RECT)
    sides = np.where(rng.random(N_RECT) < P_UP, 1.0, -1.0)   # +1 = above
    return list(zip(xs, sides))


def hits(x, y, layout):
    for x0, side in layout:
        if x0 <= x <= x0 + RECT_LEN:
            if side > 0 and INNER <= y <= OUTER:
                return True
            if side < 0 and -OUTER <= y <= -INNER:
                return True
    return False


def demo_aim(x, layout):
    """Demonstrator sees the layout: berth to the FREE side of the next rect."""
    if x >= CLEAN_X:
        return 0.0
    for x0, side in layout:
        if x < x0 + RECT_LEN and x >= x0 - LEAD:
            return -BERTH if side > 0 else BERTH
    return 0.0


def gen_episodes(n_episodes, seed):
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    n_ok = 0
    for _ in range(n_episodes):
        layout = sample_layout(rng)
        y = float(rng.uniform(-8.0, 8.0)); x = 0.0
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        bad = False
        while x < X_GOAL_MM:
            st = np.array([x, y], dtype=np.float32)
            ch = np.zeros(2 * HORIZON, dtype=np.float32)
            xx, yy = x, y
            for j in range(HORIZON):
                yt = demo_aim(xx, layout)
                dy = -K_SERVO * (yy - yt)
                ch[2 * j] = FORWARD_MM / ACTION_MM
                ch[2 * j + 1] = dy / ACTION_MM
                yy += dy; xx += FORWARD_MM
                if abs(yy) > half_width(xx) or hits(xx, yy, layout):
                    bad = True
            S.append(st); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = xx, yy
        if (not bad) and abs(y) < DOCK_TOL_MM:
            n_ok += 1
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)),
            n_ok / n_episodes)


def train_flow_hist_rc(st, pv, ac, seed, width, steps, batch_size, lr, device):
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
    """Each eval rollout gets its own unobserved layout, as at deployment."""
    rng = np.random.default_rng(90000 + seed)
    layouts = [sample_layout(rng) for _ in range(n_eval)]
    x = np.zeros(n_eval)
    y = np.linspace(-8.0, 8.0, n_eval)
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
            for k, idx in enumerate(ids):
                if active[idx] and (abs(y[idx]) > half_width(x[idx])
                                    or hits(x[idx], y[idx], layouts[idx])):
                    collision[idx] = True; active[idx] = False
            reach = (x[ids] >= X_GOAL_MM) & active[ids]
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[reach]] = False
    ok = np.isfinite(y_goal) & ~collision & (np.abs(y_goal) < DOCK_TOL_MM)
    return {"sr": float(ok.mean()), "collision": float(collision.mean()),
            "reached": float(np.isfinite(y_goal).mean()),
            "landing_p50": float(np.nanmedian(y_goal)) if np.isfinite(y_goal).any() else float("nan"),
            "traces": [{"x": v[0], "y": v[1]} for v in traces.values()],
            "layout0": [[float(a_), float(b_)] for a_, b_ in layouts[list(tr_ids)[0]]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{N_RECT} rects/episode, berth {BERTH}mm, p_up {P_UP}, clean tail from x={CLEAN_X}", flush=True)

    arms = [("l2", "regression", "reg"), ("flow", "flow", "flow_hist"), ("ht2", "ht", "reg")]
    only = os.environ.get("RC_ARMS", "")
    if only:
        arms = [a for a in arms if a[0] in set(only.split(","))]
    out = {"config": {"n_rect": N_RECT, "berth": BERTH, "p_up": P_UP, "clean_x": CLEAN_X,
                      "inner": INNER, "outer": OUTER, "rect_len": RECT_LEN}, "arms": {}}
    for name, kind, sampler in arms:
        srs, cols, seed0 = [], [], None
        for seed in args.seeds:
            st, pv, ac, osr = gen_episodes(2500, seed=1000 + seed)
            if seed == args.seeds[0] and name == arms[0][0]:
                print(f"oracle (demonstrator sees the rects): SR={osr:.3f}", flush=True)
                out["oracle_sr"] = osr
            net = (train_flow_hist_rc(st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device)
                   if kind == "flow" else
                   train_hist(kind, st, pv, ac, seed, args.width, args.steps, 512, 1e-3, device))
            r = rollout(net, sampler, device, seed=seed)
            srs.append(r["sr"]); cols.append(r["collision"])
            if seed == args.seeds[0]:
                seed0 = {"traces": r["traces"], "layout0": r["layout0"]}
        out["arms"][name] = {"sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)),
                             "sr_per_seed": srs, "collision_mean": float(np.mean(cols)),
                             "seed0": seed0}
        print(f"{name:6s} SR={np.mean(srs):.3f}±{np.std(srs):.2f}  collision={np.mean(cols):.3f}", flush=True)
    Path(os.environ.get("RC_OUT", "analysis/toy2d_rects.json")).write_text(json.dumps(out, indent=1))
    print("WROTE", os.environ.get("RC_OUT", "analysis/toy2d_rects.json"), flush=True)


if __name__ == "__main__":
    main()
