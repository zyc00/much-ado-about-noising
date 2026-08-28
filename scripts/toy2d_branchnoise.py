"""2D verification of the branch+noise endpoint-sweep NFL design.

Labels per state: aim in {-A w.p. 0.6, +A w.p. 0.4} (operator branch choice,
executed) plus a symmetric Gaussian smear (scale SIG). Persistent residual r
shifts the servo equilibrium by r*ACTION_MM/K_SERVO mm, so the two branches
land at -+A*20 mm. Predictions:
  L2        -> mixture mean  (0.6(-A)+0.4(A) = -0.2A  -> -1.2 mm at A=0.3)
  HT nu=2   -> majority branch center (about -6 mm)
  MIP full  -> anchor at mean, denoiser projects onto the near (majority)
               branch's inner edge: between, ON the smeared support
No single dock is assumed: rollouts record the landing y of every episode and
success is computed for a swept dock position d (|y-d| < DOCK_TOL).

Usage: python scripts/toy2d_branchnoise.py --seeds 0 1 2 3 4 5 6 7
Writes analysis/toy2d_branchnoise.json
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
    ViewNet, make_chunks, normalize_state, predict,
    train, train_nll,
)


WALL = os.environ.get("BR_WALL", "")  # orig | wide (default: orig for skewonly)


def half_width_mm(x):
    import numpy as _np
    wall = WALL or ("orig" if os.environ.get("BR_SMEAR") in ("skewonly", "skewwide", "skewflip") else "wide")
    floor = 5.0 if wall == "orig" else 8.0
    return floor + 9.0 * (1.0 - _np.clip(x / 130.0, 0.0, 1.0))

A_BRANCH = float(os.environ.get("BR_A", "0.3"))
SIG = float(os.environ.get("BR_SIG", "0.1"))
P_MAJ = float(os.environ.get("BR_PMAJ", "0.7"))
HORIZON = 8


SMEAR = os.environ.get("BR_SMEAR", "gauss")  # gauss | toyskew | skewonly
B_SKEW = float(os.environ.get("BR_B", "0.05"))


FLIP_PERIOD_MM = float(os.environ.get("BR_FLIP_PERIOD", "64"))


def branch_residuals(n_states: int, repeats: int, rng: np.random.Generator,
                     x_state=None) -> np.ndarray:
    if SMEAR == "skewflip":
        # HIGH-FREQUENCY structure change: the wide-majority skew's mode
        # DIRECTION flips as a square wave along x (period FLIP_PERIOD_MM).
        # Labels are zero-mean at every state; the mode position oscillates.
        assert x_state is not None
        sign = np.where(((x_state // (FLIP_PERIOD_MM / 2)).astype(int) % 2) == 0,
                        -1.0, 1.0)[:, None]
        b = (rng.random((n_states, repeats)) < 0.2).astype(np.float64)
        base = B_SKEW * (5.0 * b - 1.0)
        wid = rng.normal(0.0, SIG, size=base.shape) * (1.0 - b)
        return (sign * (base + wid)).astype(np.float32)
    if SMEAR == "skewonly":
        # EXACTLY the original toy_mip_nfl skew nuisance: no branches,
        # single correct action + zero-mean toyskew jitter {-B:.8, +4B:.2}
        b = (rng.random((n_states, repeats)) < 0.2).astype(np.float64)
        return (B_SKEW * (5.0 * b - 1.0)).astype(np.float32)
    if SMEAR == "skewwide":
        # skew with a WIDE majority component: -B + N(0, SIG) w.p. .8,
        # exact +4B atom w.p. .2 (still zero-mean). Width separates MIP's
        # posterior projection from HT's component-center location.
        b = (rng.random((n_states, repeats)) < 0.2).astype(np.float64)
        base = B_SKEW * (5.0 * b - 1.0)
        wid = rng.normal(0.0, SIG, size=base.shape) * (1.0 - b)
        return (base + wid).astype(np.float32)
    u = rng.random((n_states, repeats))
    aim = np.where(u < P_MAJ, -A_BRANCH, A_BRANCH)
    if SMEAR == "toyskew":
        b = (rng.random(aim.shape) < 0.2).astype(np.float64)
        eps = B_SKEW * (5.0 * b - 1.0)
    else:
        eps = rng.normal(0.0, SIG, size=aim.shape)
    return (aim + eps).astype(np.float32)


def make_ds(n_states: int, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, X_GOAL_MM - FORWARD_MM, size=n_states)
    hw = half_width_mm(x)
    y = rng.uniform(-0.85 * hw, 0.85 * hw)
    base_state = np.stack([x, y], axis=1).astype(np.float32)
    residual = branch_residuals(n_states, repeats, rng, x_state=x)
    state = np.repeat(base_state, repeats, axis=0)
    action = make_chunks(state, residual.reshape(-1), HORIZON)
    state = normalize_state(state)
    order = rng.permutation(len(state))
    return torch.from_numpy(state[order]), torch.from_numpy(action[order])


def rollout_landings(net: ViewNet, sampler: str, device, n_eval: int = 401,
                     chunked: bool = True, want_traces: bool = False) -> dict:
    x = np.zeros(n_eval)
    y = np.linspace(-10.0, 10.0, n_eval)
    tr_ids = set(np.linspace(0, n_eval - 1, 11, dtype=int)) if want_traces else set()
    traces = {i: ([0.0], [float(y[i])]) for i in tr_ids}
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    for _ in range(70):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        state = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        action = predict(net, state, sampler, device)
        n_exec = HORIZON if chunked else 1
        for j in range(n_exec):
            live = active[ids]
            dx = np.clip(action[:, 2 * j] * ACTION_MM, -2.0, 8.0)
            dy = np.clip(action[:, 2 * j + 1] * ACTION_MM, -12.0, 12.0)
            x[ids] = np.where(live, x[ids] + dx, x[ids])
            y[ids] = np.where(live, y[ids] + dy, y[ids])
            for ii in tr_ids:
                if active[ii]:
                    traces[ii][0].append(float(x[ii]))
                    traces[ii][1].append(float(y[ii]))
            outside = np.abs(y[ids]) > half_width_mm(x[ids])
            collision[ids[outside & live]] = True
            reach = (x[ids] >= X_GOAL_MM) & live
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[(outside & live) | reach]] = False
    ok = np.isfinite(y_goal) & ~collision
    return {"y_goal": y_goal, "ok": ok,
            "reached": float(np.isfinite(y_goal).mean()),
            "collision": float(collision.mean()),
            "traces": [{"x": v[0], "y": v[1]} for v in traces.values()]}


def sr_curve(land: dict, docks: np.ndarray) -> np.ndarray:
    yg, ok = land["y_goal"], land["ok"]
    return np.array([
        float((ok & (np.abs(yg - d) < DOCK_TOL_MM)).mean()) for d in docks
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--n_states", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    docks = np.round(np.arange(-8.0, 4.01, 0.25), 2)
    arms = [
        ("l2", "regression", "regression"),
        ("mip_full", "mip", "mip_full"),
        ("mip_step1", "mip", "mip_step1"),
        ("ht2", "ht", "ht"),
        ("ht05", "ht05", "ht"),
        ("hg", "hg", "hg"),
    ]
    out = {"config": {"A": A_BRANCH, "sig": SIG, "p_maj": P_MAJ,
                      "docks": docks.tolist(), "tol": DOCK_TOL_MM,
                      "steps": args.steps, "width": args.width,
                      "chunks": args.n_states * args.repeats},
           "arms": {}}

    # population reference points (mm)
    mm = ACTION_MM / K_SERVO
    out["reference_mm"] = {"mean": (P_MAJ * (-A_BRANCH) + (1 - P_MAJ) * A_BRANCH) * mm,
                          "majority_mode": -A_BRANCH * mm,
                          "minority_mode": A_BRANCH * mm}

    for name, kind, sampler in arms:
        curves, medians, reaches = [], [], []
        for seed in args.seeds:
            st, ac = make_ds(args.n_states, args.repeats, seed=1000 + seed)
            if kind in ("ht", "ht05", "hg"):
                net, _ = train_nll(kind if kind != "ht" else "ht", st, ac,
                                   seed, args.width, args.steps, 512, 1e-3, device)
            else:
                net, _ = train(kind, st, ac, seed, args.width, args.steps,
                               512, 1e-3, device)
            land = rollout_landings(net, sampler, device, want_traces=(seed == args.seeds[0]))
            curves.append(sr_curve(land, docks))
            medians.append(float(np.nanmedian(land["y_goal"])))
            reaches.append(land["reached"])
            if seed == args.seeds[0]:
                seed0 = {"traces": land["traces"],
                         "landings": [float(v) for v in land["y_goal"]]}
        curves = np.array(curves)
        mcurve = curves.mean(axis=0)
        peak = docks[int(mcurve.argmax())]
        out["arms"][name] = {
            "seed0": seed0,
            "sr_curve_mean": mcurve.tolist(),
            "sr_curve_std": curves.std(axis=0).tolist(),
            "landing_median_per_seed": medians,
            "landing_median": float(np.median(medians)),
            "peak_dock_mm": float(peak),
            "peak_sr": float(mcurve.max()),
            "reached_mean": float(np.mean(reaches)),
        }
        print(f"{name:10s} landing_p50={np.median(medians):+.2f} mm  "
              f"peak_dock={peak:+.2f} mm  peak_SR={mcurve.max():.3f}  "
              f"reached={np.mean(reaches):.3f}", flush=True)

    outp = Path(__file__).resolve().parents[1] / "analysis" / "toy2d_branchnoise.json"
    outp.write_text(json.dumps(out, indent=1))
    print("WROTE", outp, flush=True)


if __name__ == "__main__":
    main()
