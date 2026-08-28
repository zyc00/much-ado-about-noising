"""Retry/regrasp toy: behavioral, everything EXECUTED, oracle 100%.

Demonstrator servos to the dock (y=0, tol 0.5mm). With prob F_FLAW the first
dock approach is flawed (hidden cause, not in the state): the demonstrator
backs off to a staging lane (y=+8, dx<0) for 2 chunks, then re-approaches and
docks — at most ONE retry per episode, so demo completion time is bounded
(worst case ~8 chunks). Retreat chunks live at near-dock states with mass
~0.27 (< 1/3).

Rollouts are scored under two budgets, both multiples of the demonstrator's
bounded worst case (oracle 1.0 under both): generous (2x) and tight (1.25x).
Predictions: L2 blends approach+staging at the dock (lands ~+2mm, fails);
HT nu=2 rejects the retreat cluster and docks; DP re-samples retreat i.i.d.
per dock visit (unbounded geometric) — fine at 2x, decaying at 1.25x.

Usage: OB_FLOW_K=64 python scripts/toy2d_retry.py --seeds 0 1 2 --width 256 --steps 24000
Env: RT_ARMS, RT_OUT, RT_FLAW (default 0.6)
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
    normalize_state, predict, train, train_nll,
)
from toy2d_obstacle import (  # noqa: E402
    HistFlowNet, predict_flow, predict_flow_hist, train_flow,
)

F_FLAW = float(os.environ.get("RT_FLAW", "0.4"))
STAGE_Y = 8.0
RETREAT_DX = -2.0            # mm/step during retreat
TRIG_X = float(os.environ.get("RT_TRIG", "120.0"))  # retreat can trigger from here on
HORIZON = 8


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / X_GOAL_MM, 0.0, 1.0)


def servo_chunk(x, y, lane, dx_step):
    ch = np.zeros(2 * HORIZON, dtype=np.float32)
    xx, yy = x, y
    for j in range(HORIZON):
        dy = -K_SERVO * (yy - lane)
        ch[2 * j] = dx_step / ACTION_MM
        ch[2 * j + 1] = dy / ACTION_MM
        yy += dy; xx += dx_step
    return ch, xx, yy


def gen_episodes(n_episodes, seed):
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    lengths = []
    n_ok = 0
    for _ in range(n_episodes):
        flawed = rng.random() < F_FLAW
        y = float(rng.uniform(-10.0, 10.0))
        x = float(rng.uniform(0.0, FORWARD_MM * HORIZON))  # de-grid chunk starts
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        n_ch = 0
        retreat_left = 0
        while x < X_GOAL_MM:
            st = np.array([x, y], dtype=np.float32)
            if retreat_left > 0:
                ch, x2, y2 = servo_chunk(x, y, STAGE_Y, RETREAT_DX)
                retreat_left -= 1
            elif flawed and x >= TRIG_X:
                ch, x2, y2 = servo_chunk(x, y, STAGE_Y, RETREAT_DX)
                retreat_left = 1
                flawed = False          # flaw resolved after this one retry
            else:
                ch, x2, y2 = servo_chunk(x, y, 0.0, FORWARD_MM)
            S.append(st); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = x2, y2; n_ch += 1
        lengths.append(n_ch)
        if abs(y) < DOCK_TOL_MM:
            n_ok += 1
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)),
            n_ok / n_episodes, int(max(lengths)))


def rollout(net, sampler, device, budget, n_eval=401):
    x = np.zeros(n_eval)
    y = np.linspace(-10.0, 10.0, n_eval)
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    prev = torch.zeros(n_eval, 2 * HORIZON, device=device)
    tr_ids = set(np.linspace(0, n_eval - 1, 13, dtype=int))
    traces = {i: ([0.0], [float(y[i])]) for i in tr_ids}
    for _ in range(budget):
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
            x[ids] = np.where(live, x[ids] + np.clip(a[:, 2 * j] * ACTION_MM, -4.0, 8.0), x[ids])
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
            "timeout": float(active.mean()),
            "landing_p50": float(np.nanmedian(y_goal)) if np.isfinite(y_goal).any() else float("nan"),
            "traces": [{"x": v[0], "y": v[1]} for v in traces.values()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arms = [("l2", "regression", "regression"), ("hg", "hg", "hg"),
            ("ht2", "ht", "ht"), ("ht05", "ht05", "ht"), ("mip", "mip", "mip_full"),
            ("flow", "flow", "flow"), ("flow_hist", "flow_hist", "flow_hist")]
    only = os.environ.get("RT_ARMS", "")
    if only:
        keep = set(only.split(","))
        arms = [a for a in arms if a[0] in keep]
    out = {"config": {"f_flaw": F_FLAW, "stage_y": STAGE_Y, "trig_x": TRIG_X},
           "arms": {}}
    budgets = None
    for name, kind, sampler in arms:
        res = {}
        seed0 = None
        for seed in args.seeds:
            st, pv, ac, osr, olen = gen_episodes(4000, seed=1000 + seed)
            if budgets is None:
                budgets = {"generous": 2 * olen, "tight": max(olen + 1, int(round(1.25 * olen))),
                           "exact": olen}
                out["config"].update({"oracle_sr": osr, "oracle_max_chunks": olen,
                                      "budgets": budgets})
                print(f"oracle (executed demos): SR={osr:.3f}  worst-case {olen} chunks; "
                      f"budgets generous={budgets['generous']} tight={budgets['tight']}",
                      flush=True)
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
            elif kind == "flow":
                net, _ = train_flow(st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            elif kind in ("hg", "ht", "ht05"):
                net, _ = train_nll("ht" if kind == "ht" else kind, st, ac, seed,
                                   args.width, args.steps, 512, 1e-3, device)
            else:
                net, _ = train(kind, st, ac, seed, args.width, args.steps, 512, 1e-3, device)
            for bname, b in budgets.items():
                r = rollout(net, sampler, device, b)
                res.setdefault(bname, {"sr": [], "timeout": [], "landing": []})
                res[bname]["sr"].append(r["sr"])
                res[bname]["timeout"].append(r["timeout"])
                res[bname]["landing"].append(r["landing_p50"])
                if seed == args.seeds[0] and bname == "generous":
                    seed0 = {"traces": r["traces"]}
        out["arms"][name] = {
            b: {"sr_mean": float(np.mean(v["sr"])), "sr_std": float(np.std(v["sr"])),
                "sr_per_seed": v["sr"], "timeout_mean": float(np.mean(v["timeout"])),
                "landing_p50s": v["landing"]}
            for b, v in res.items()}
        out["arms"][name]["seed0"] = seed0
        g, t = res["generous"], res["tight"]
        print(f"{name:10s} generous SR={np.mean(g['sr']):.3f}±{np.std(g['sr']):.2f} "
              f"(timeout {np.mean(g['timeout']):.3f}, land {np.nanmedian(g['landing']):+.2f})  |  "
              f"tight SR={np.mean(t['sr']):.3f}±{np.std(t['sr']):.2f} "
              f"(timeout {np.mean(t['timeout']):.3f})", flush=True)
    outp = os.environ.get("RT_OUT", "analysis/toy2d_retry.json")
    Path(outp).write_text(json.dumps(out, indent=1))
    print("WROTE", outp, flush=True)


if __name__ == "__main__":
    main()
