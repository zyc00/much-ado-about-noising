"""Idle-mode toy: the dual of the obstacle cell. A fraction P_IDLE of RECORDED
chunks are IDLE (zeros — logging dropout / dead-man-release artifacts in the
command stream); the demonstrator EXECUTED clean servo, so the oracle is 100%
at every P_IDLE (recorded-only corruption, same framing as the skew/spurious
cells). States carry no dropout cue. The MAX_CHUNKS=70 budget is ~12x the
executed-demo completion time (~6 chunks).
Predictions: mean family creeps forward and succeeds; mode family commits to
inaction and times out at any budget; diffusion samples go often enough to pass
at moderate P_IDLE and decays as idle mass grows.

Usage: python scripts/toy2d_idle.py --seeds 0 1 2 3 4 5 6 7
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
    make_chunks, normalize_state, predict, predict_hist, train, train_hist,
    train_nll,
)
from toy2d_obstacle import (  # noqa: E402
    HistFlowNet, predict_flow, predict_flow_hist, train_flow,
)

P_IDLE = float(os.environ.get("ID_PIDLE", "0.6"))
SLOW = float(os.environ.get("ID_SLOW", "0.0"))  # >0: majority mode moves at this fraction of servo speed
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
    idle = rng.random(len(state)) < P_IDLE
    action[idle] = action[idle] * SLOW  # SLOW=0 -> pause; SLOW>0 -> slow but moving
    if bool(int(os.environ.get("ID_FILTER", "0"))):
        # no-op filtering: the standard curation step behind the *_no_noops
        # datasets every LIBERO-based VLA trains on. Drops near-zero chunks.
        keep = np.abs(action).max(axis=1) > 1e-6
        state, action = state[keep], action[keep]
    order = rng.permutation(len(state))
    return (torch.from_numpy(normalize_state(state)[order]),
            torch.from_numpy(action[order]))


def make_ds_episodes(n_episodes, seed):
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    for _ in range(n_episodes):
        y = float(rng.uniform(-10.0, 10.0)); x = 0.0
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        while x < X_GOAL_MM:
            st = np.array([[x, y]], dtype=np.float32)
            ch = make_chunks(st, np.zeros(1), HORIZON)[0].copy()
            xx, yy = x, y
            for j in range(HORIZON):
                yy += ch[2 * j + 1] * ACTION_MM
                xx += ch[2 * j] * ACTION_MM
            if rng.random() < P_IDLE:
                ch[:] = ch * SLOW  # SLOW=0 -> pause; SLOW>0 -> slow but moving
            S.append(st[0]); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = xx, yy
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)))


def train_flow_hist_idle(seed, width, steps, batch_size, lr, device, n_eps=3000):
    torch.manual_seed(seed)
    st, pv, ac = make_ds_episodes(n_eps, seed=1000 + seed)
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
    tr_ids = set(np.linspace(0, n_eval - 1, 13, dtype=int))
    traces = {i: ([0.0], [float(y[i])]) for i in tr_ids}
    prev = torch.zeros(n_eval, 2 * HORIZON, device=device)
    for _ in range(MAX_CHUNKS):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        st = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        if sampler == "flow":
            a = predict_flow(net, st, device)
        elif sampler == "reg_hist":
            zt = predict_hist(net, st, prev[ids], device)
            prev[ids] = zt
            a = zt.cpu().numpy()
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
    ok = np.isfinite(y_goal) & ~collision
    sr = float((ok & (np.abs(y_goal) < DOCK_TOL_MM)).mean())
    return {"sr": sr, "collision": float(collision.mean()),
            "reached": float(np.isfinite(y_goal).mean()),
            "timeout": float(active.mean() + (~np.isfinite(y_goal) & ~collision & ~active).mean()),
            "landing_p50": float(np.nanmedian(y_goal)) if np.isfinite(y_goal).any() else float("nan"),
            "traces": [{"x": v[0], "y": v[1]} for v in traces.values()]}


def oracle_sr(n_eval=401):
    """Executed demonstrator = clean servo (idle exists only in the recording).
    Returns (SR, completion chunks worst-case) under the same budget/clipping."""
    y = np.linspace(-10.0, 10.0, n_eval)
    x = np.zeros(n_eval)
    y_goal = np.full(n_eval, np.nan)
    done = np.zeros(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    chunks_used = np.zeros(n_eval)
    for c in range(MAX_CHUNKS):
        for _j in range(HORIZON):
            live = ~done
            y = y + np.where(live, -K_SERVO * y, 0.0)
            x = x + np.where(live, FORWARD_MM, 0.0)
            collision |= live & (np.abs(y) > half_width(x))
            reach = live & (x >= X_GOAL_MM)
            y_goal[reach] = y[reach]
            chunks_used[reach] = c + 1
            done |= collision | reach
        if done.all():
            break
    ok = np.isfinite(y_goal) & (np.abs(y_goal) < DOCK_TOL_MM) & ~collision
    return float(ok.mean()), float(chunks_used.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--width", type=int, default=128)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    o_sr, o_chunks = oracle_sr()
    print(f"oracle SR: {o_sr:.3f} (executed clean servo; worst-case {o_chunks:.0f} "
          f"of {MAX_CHUNKS} chunks; idle is recorded-only)", flush=True)
    arms = [("l2", "regression", "regression"), ("mip", "mip", "mip_full"),
            ("hg", "hg", "hg"),
            ("ht2", "ht", "ht"), ("ht1", "ht1", "ht"), ("ht05", "ht05", "ht"),
            ("flow", "flow", "flow"),
            ("flow_hist", "flow_hist", "flow_hist"),
            ("l2_hist", "l2_hist", "reg_hist"),
            ("ht2_hist", "ht2_hist", "reg_hist")]
    only = os.environ.get("ID_ARMS", "")
    if only:
        keep = set(only.split(","))
        arms = [a for a in arms if a[0] in keep]
    out = {"config": {"p_idle": P_IDLE}, "arms": {}}
    for name, kind, sampler in arms:
        srs, reach, land = [], [], []
        seed0 = None
        for seed in args.seeds:
            st, ac = make_ds(3000, 10, seed=1000 + seed)
            if kind in ("ht", "ht1", "ht05", "hg"):
                net, _ = train_nll("ht" if kind == "ht" else kind, st, ac, seed,
                                   args.width, args.steps, 512, 1e-3, device)
            elif kind == "flow":
                net, _ = train_flow(st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            elif kind == "flow_hist":
                net = train_flow_hist_idle(seed, args.width, args.steps, 512, 1e-3, device)
            elif kind in ("l2_hist", "ht2_hist"):
                hs, hp, ha = make_ds_episodes(3000, seed=1000 + seed)
                net = train_hist("regression" if kind == "l2_hist" else "ht",
                                 hs, hp, ha, seed, args.width, args.steps, 512, 1e-3, device)
            else:
                net, _ = train(kind, st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            r = rollout(net, sampler, device)
            srs.append(r["sr"]); reach.append(r["reached"]); land.append(r["landing_p50"])
            if seed == args.seeds[0]:
                seed0 = {"traces": r["traces"]}
        out["arms"][name] = {"sr_mean": float(np.mean(srs)), "sr_std": float(np.std(srs)),
                             "reached_mean": float(np.mean(reach)),
                             "sr_per_seed": srs, "seed0": seed0}
        print(f"{name:10s} SR={np.mean(srs):.3f}±{np.std(srs):.2f}  reached={np.mean(reach):.3f}",
              flush=True)
    Path(os.environ.get("ID_OUT", "analysis/toy2d_idle.json")).write_text(json.dumps(out, indent=1))
    print("WROTE", os.environ.get("ID_OUT", "analysis/toy2d_idle.json"), flush=True)


if __name__ == "__main__":
    main()
