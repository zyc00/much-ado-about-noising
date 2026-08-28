"""2D obstacle-detour toy (push-T mechanism): the goal is hidden behind a
mid-corridor block; demonstrators detour left (60%) or right (40%) and
re-center. All demos succeed (100% oracle, executed aim choice, no noise).
Prediction: L2 averages the detours into the block; HT plurality-commits and
goes around; MIP is the empirical question.

Usage: python scripts/toy2d_obstacle.py --seeds 0 1 2 3 4 5 6 7
Writes analysis/toy2d_obstacle.json
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
    ViewNet, normalize_state, predict, predict_hist, train, train_hist, train_nll,
)


def train_flow(state, action, seed, width, steps, batch_size, lr, device):
    """Conditional flow matching over action chunks (the diffusion-family arm)."""
    torch.manual_seed(seed)
    A = action.shape[1]
    net = ViewNet(A, width).to(device)
    state = state.to(device)
    action = action.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(state)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        sb, ab = state[idx], action[idx]
        z0 = torch.randn_like(ab)
        t = torch.rand(batch_size, 1, device=device)
        zt = (1 - t) * z0 + t * ab
        v = net(sb, zt, t)
        loss = ((v - (ab - z0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    return net, {}


FLOW_K = int(os.environ.get("OB_FLOW_K", "16"))
LOG_EVERY = int(os.environ.get("OB_LOG_EVERY", "0"))
FLOW_T0 = float(os.environ.get("OB_FLOW_T0", "0"))  # warm-start depth (0 = full re-noise)


@torch.inference_mode()
def predict_flow(net, state_world, device, k_steps=None, z0=None, a_prev=None):
    if k_steps is None:
        k_steps = FLOW_K
    state = torch.from_numpy(normalize_state(state_world.astype("float32"))).to(device)
    noise = z0.clone() if z0 is not None else torch.randn(len(state), net.action_dim, device=device)
    t0 = FLOW_T0 if a_prev is not None else 0.0
    if t0 > 0:
        z = (1 - t0) * noise + t0 * a_prev
    else:
        z = noise
    n_int = max(1, int(round(k_steps * (1 - t0))))
    for k in range(n_int):
        t = torch.full((len(state), 1), t0 + (1 - t0) * k / n_int, device=device)
        z = z + ((1 - t0) / n_int) * net(state, z, t)
    return z.cpu().numpy()

# obstacle geometry (mm)
BLK_X0 = float(os.environ.get("OB_BLK_X0", "60.0"))
BLK_X1 = float(os.environ.get("OB_BLK_X1", "100.0"))
BLK_HALF = 3.0
DETOUR_A = 6.0           # detour aim (mm); passage gap ~5-6mm each side
AIM_X0 = float(os.environ.get("OB_AIM_X0", "20.0"))  # where demos commit
AIM_X1 = 100.0
P_MAJ = float(os.environ.get("OB_PMAJ", "0.6"))
SIG = float(os.environ.get("OB_SIG", "0.05"))  # small aim smear (normalized)
HORIZON = 8


WIDE = bool(int(os.environ.get("OB_WIDE", "0")))


def half_width(x):
    """Default funnel narrows to 5mm; OB_WIDE=1 keeps >=12.5mm so a late block
    (and its +-6mm detour) still fits — the funnel's 5mm tail cannot."""
    if WIDE:
        return 14.0 - 1.5 * np.clip(x / 160.0, 0.0, 1.0)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


# Per-demo commitment point: each demonstrator starts its detour at a different
# x, so the branch fans out gradually instead of forking at one synchronized
# point. OB_AIM_JITTER=0 restores the original single-fork behaviour.
AIM_JIT = float(os.environ.get("OB_AIM_JITTER", "16.0"))

# Execution noise on the demonstrator (mm per step), UNPREDICTABLE from the state.
# Without it the demonstrator is deterministic given (x, y): the network drives the
# residual to ~0 on every non-deciding state, A*log(sigma) is unbounded below there,
# and sigma collapses to its floor. Real demonstrations always carry such noise.
EXEC_NOISE = float(os.environ.get("OB_EXEC_NOISE", "0.0"))


def target_y(x, branch, aim0=None):
    """Demonstrator's lateral target: detour while approaching/passing the
    block, re-center after. aim0 is the per-sample commitment x."""
    a0 = AIM_X0 if aim0 is None else aim0
    return np.where((x >= a0) & (x <= BLK_X1), branch * DETOUR_A, 0.0)


def make_chunks_target(state, branch, rng):
    """Action chunks for a servo tracking target_y(x, branch), with a small
    aim smear. Mirrors make_chunks' dynamics (dy = -K(y - y_t))."""
    n = len(state)
    y = state[:, 1].astype(np.float64).copy()
    x = state[:, 0].astype(np.float64).copy()
    smear = rng.normal(0.0, SIG, size=n) * ACTION_MM
    aim0 = AIM_X0 + rng.uniform(-AIM_JIT, AIM_JIT, size=n) if AIM_JIT > 0 else None
    action = np.empty((n, 2 * HORIZON), dtype=np.float32)
    for j in range(HORIZON):
        yt = target_y(x, branch, aim0) + smear
        dy = -K_SERVO * (y - yt)
        action[:, 2 * j] = FORWARD_MM / ACTION_MM
        action[:, 2 * j + 1] = dy / ACTION_MM
        y += dy
        x += FORWARD_MM
    return action


class HistFlowNet(torch.nn.Module):
    """Flow net conditioned on (state, prev action chunk)."""

    def __init__(self, action_dim, width):
        super().__init__()
        self.action_dim = action_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 + action_dim + action_dim + 1, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, action_dim))

    def forward(self, state, prev, z, t):
        return self.net(torch.cat([state, prev, z, t], dim=1))


def make_ds_episodes(n_episodes, seed):
    """Episode-rollout dataset: (state, prev_chunk, chunk) triples from
    simulated demonstrators (needed for history conditioning)."""
    rng = np.random.default_rng(seed)
    S, P, A = [], [], []
    for _ in range(n_episodes):
        branch = 1.0 if rng.random() < P_MAJ else -1.0
        ep_aim0 = AIM_X0 + (rng.uniform(-AIM_JIT, AIM_JIT) if AIM_JIT > 0 else 0.0)
        y = float(rng.uniform(-10, 10)); x = 0.0
        prev = np.zeros(2 * HORIZON, dtype=np.float32)
        smear = rng.normal(0, SIG)
        while x < X_GOAL_MM:
            st = np.array([[x, y]], dtype=np.float32)
            ch = np.zeros(2 * HORIZON, dtype=np.float32)
            xx, yy = x, y
            for j in range(HORIZON):
                yt = (branch * DETOUR_A + smear * ACTION_MM) if (ep_aim0 <= xx <= BLK_X1) else 0.0
                dy = -K_SERVO * (yy - yt)
                if EXEC_NOISE > 0:
                    dy += rng.normal(0.0, EXEC_NOISE)
                ch[2 * j] = FORWARD_MM / ACTION_MM
                ch[2 * j + 1] = dy / ACTION_MM
                yy += dy; xx += FORWARD_MM
            S.append(st[0]); P.append(prev.copy()); A.append(ch)
            prev = ch; x, y = xx, yy
    S = np.array(S, dtype=np.float32)
    return (torch.from_numpy(normalize_state(S)),
            torch.from_numpy(np.array(P)), torch.from_numpy(np.array(A)))


def train_flow_hist(seed, width, steps, batch_size, lr, device, n_eps=2500, log_every=0):
    torch.manual_seed(seed)
    st, pv, ac = make_ds_episodes(n_eps, seed=1000 + seed)
    net = HistFlowNet(ac.shape[1], width).to(device)
    st, pv, ac = st.to(device), pv.to(device), ac.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(st)
    net.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        z0 = torch.randn_like(ac[idx])
        t = torch.rand(batch_size, 1, device=device)
        zt = (1 - t) * z0 + t * ac[idx]
        v = net(st[idx], pv[idx], zt, t)
        loss = ((v - (ac[idx] - z0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"      [flow_hist] step {step:6d}  loss {float(loss):.5f}", flush=True)
    net.eval()
    return net


@torch.inference_mode()
def predict_flow_hist(net, state_world, prev, device, k_steps=None):
    if k_steps is None:
        k_steps = FLOW_K
    state = torch.from_numpy(normalize_state(state_world.astype("float32"))).to(device)
    z = torch.randn(len(state), net.action_dim, device=device)
    for k in range(k_steps):
        t = torch.full((len(state), 1), k / k_steps, device=device)
        z = z + (1.0 / k_steps) * net(state, prev, z, t)
    return z


def train_reg_hist(kind, seed, width, steps, batch_size, lr, device, n_eps=2500, log_every=0):
    """Regression / HT on the SAME episode-rollout data the history flow arm uses,
    with the previous action chunk in the conditioning — so all three families see
    identical history and the panel compares losses, not conditioning."""
    st, pv, ac = make_ds_episodes(n_eps, seed=1000 + seed)
    return train_hist(kind, st, pv, ac, seed, width, steps, batch_size, lr, device, log_every)


def rollout_flow_hist(net, device, n_eval=401, predict_fn=None):
    predict_fn = predict_fn or predict_flow_hist
    x = np.zeros(n_eval)
    y = np.linspace(-10.0, 10.0, n_eval)
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    prev = torch.zeros(n_eval, net.action_dim, device=device)
    tr_ids = set(np.linspace(0, n_eval - 1, 25, dtype=int))
    traces = {i: ([0.0], [float(y[i])]) for i in tr_ids}
    for _ in range(70):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        st = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        zt = predict_fn(net, st, prev[ids], device)
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
            bad = (np.abs(y[ids]) > half_width(x[ids])) | in_block(x[ids], y[ids])
            collision[ids[bad & live]] = True
            reach = (x[ids] >= X_GOAL_MM) & live
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[(bad & live) | reach]] = False
    ok = np.isfinite(y_goal) & ~collision
    succ = ok & (np.abs(y_goal) < DOCK_TOL_MM)
    sr = float(succ.mean())
    ids_sorted = sorted(traces)
    kinds = ["ok" if succ[i] else ("collision" if collision[i] else "off-dock") for i in ids_sorted]
    return {"sr": sr, "collision": float(collision.mean()),
            "reached": float(np.isfinite(y_goal).mean()),
            "landing_p50": float(np.nanmedian(y_goal)) if np.isfinite(y_goal).any() else float("nan"),
            "traces": [{"x": traces[i][0], "y": traces[i][1], "kind": k}
                       for i, k in zip(ids_sorted, kinds)]}


def make_ds(n_states, repeats, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, X_GOAL_MM - FORWARD_MM, size=n_states)
    hw = half_width(x)
    y = rng.uniform(-0.85 * hw, 0.85 * hw)
    base = np.stack([x, y], axis=1).astype(np.float32)
    state = np.repeat(base, repeats, axis=0)
    branch = np.where(rng.random(len(state)) < P_MAJ, +1.0, -1.0)
    action = make_chunks_target(state, branch, rng)
    order = rng.permutation(len(state))
    return (torch.from_numpy(normalize_state(state)[order]),
            torch.from_numpy(action[order]))


def in_block(x, y):
    return (x >= BLK_X0) & (x <= BLK_X1) & (np.abs(y) <= BLK_HALF)


def rollout(net, sampler, device, n_eval=401):
    frozen = os.environ.get("OB_FLOW_FROZEN", "0") == "1"
    z_fix = None
    a_prev = None
    if sampler == "flow" and frozen:
        g = torch.Generator(device=device).manual_seed(1234)
        z_fix = torch.randn(n_eval, net.action_dim, device=device, generator=g)
    x = np.zeros(n_eval)
    y = np.linspace(-10.0, 10.0, n_eval)
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    tr_ids = set(np.linspace(0, n_eval - 1, 25, dtype=int))
    traces = {i: ([0.0], [float(y[i])]) for i in tr_ids}
    for _ in range(70):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        st = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        if sampler == "flow":
            ap_ids = a_prev[ids] if a_prev is not None else None
            a_np = predict_flow(net, st, device,
                                z0=(z_fix[ids] if z_fix is not None else None),
                                a_prev=ap_ids)
            if FLOW_T0 > 0:
                if a_prev is None:
                    a_prev = torch.zeros(n_eval, net.action_dim, device=device)
                a_prev[ids] = torch.from_numpy(a_np).to(device)
            a = a_np
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
            bad = (np.abs(y[ids]) > half_width(x[ids])) | in_block(x[ids], y[ids])
            collision[ids[bad & live]] = True
            reach = (x[ids] >= X_GOAL_MM) & live
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[(bad & live) | reach]] = False
    ok = np.isfinite(y_goal) & ~collision
    succ = ok & (np.abs(y_goal) < DOCK_TOL_MM)
    sr = float(succ.mean())
    ids_sorted = sorted(traces)
    kinds = ["ok" if succ[i] else ("collision" if collision[i] else "off-dock") for i in ids_sorted]
    return {"sr": sr, "collision": float(collision.mean()),
            "reached": float(np.isfinite(y_goal).mean()),
            "landing_p50": float(np.nanmedian(y_goal)) if np.isfinite(y_goal).any() else float("nan"),
            "traces": [{"x": traces[i][0], "y": traces[i][1], "kind": k}
                       for i, k in zip(ids_sorted, kinds)]}


def oracle_sr(rng, n_eval=401):
    y = np.linspace(-10.0, 10.0, n_eval)
    x = np.zeros(n_eval)
    branch = np.where(rng.random(n_eval) < P_MAJ, +1.0, -1.0)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    done = np.zeros(n_eval, dtype=bool)
    for _ in range(70):
        yt = target_y(x, branch)
        y = y + (-K_SERVO * (y - yt))
        x = x + FORWARD_MM
        live = ~done
        collision |= live & ((np.abs(y) > half_width(x)) | in_block(x, y))
        reach = live & (x >= X_GOAL_MM)
        y_goal[reach] = y[reach]
        done |= collision | reach
        if done.all():
            break
    ok = np.isfinite(y_goal) & (np.abs(y_goal) < DOCK_TOL_MM) & ~collision
    return float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--width", type=int, default=128)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"oracle SR: {oracle_sr(np.random.default_rng(0)):.3f}", flush=True)
    arms = [("l2", "regression", "regression"), ("mip_full", "mip", "mip_full"),
            ("mip_step1", "mip", "mip_step1"), ("ht2", "ht", "ht"),
            ("ht1", "ht1", "ht"), ("ht05", "ht05", "ht"), ("hg", "hg", "hg"),
            ("flow", "flow", "flow"), ("flow_both", "flow_both", "flow"),
            ("flow_hist", "flow_hist", "flow"),
            ("l2_hist", "reg_hist", "hist"), ("ht2_hist", "ht_hist", "hist")]
    only = os.environ.get("OB_ARMS", "")
    if only:
        keep = set(only.split(","))
        arms = [a for a in arms if a[0] in keep]
    out = {"config": {"p_maj": P_MAJ, "sig": SIG, "detour": DETOUR_A,
                      "block": [BLK_X0, BLK_X1, BLK_HALF]}, "arms": {}}
    for name, kind, sampler in arms:
        srs, cols, lands = [], [], []
        seed0 = None
        for seed in args.seeds:
            st, ac = make_ds(3000, 10, seed=1000 + seed)
            if kind in ("flow_hist", "reg_hist", "ht_hist"):
                if kind == "flow_hist":
                    net = train_flow_hist(seed, args.width, args.steps, 512, 1e-3, device,
                                          log_every=LOG_EVERY)
                    r = rollout_flow_hist(net, device)
                else:
                    net = train_reg_hist("regression" if kind == "reg_hist" else "ht",
                                         seed, args.width, args.steps, 512, 1e-3, device,
                                         log_every=LOG_EVERY)
                    r = rollout_flow_hist(net, device, predict_fn=predict_hist)
                srs.append(r["sr"]); cols.append(r["collision"]); lands.append(r["landing_p50"])
                print(f"  seed {seed}: SR={r['sr']:.3f} col={r['collision']:.3f}", flush=True)
                if seed == args.seeds[0]:
                    seed0 = {"traces": r["traces"]}
                continue
            if kind == "flow_both":
                net, _ = train_flow(st, ac, seed, args.width, args.steps, 512, 1e-3, device)
                os.environ["OB_FLOW_FROZEN"] = "0"
                r_rs = rollout(net, "flow", device)
                os.environ["OB_FLOW_FROZEN"] = "1"
                r_fz = rollout(net, "flow", device)
                print(f"  seed {seed}: resample SR={r_rs['sr']:.3f} col={r_rs['collision']:.3f} | "
                      f"frozen SR={r_fz['sr']:.3f} col={r_fz['collision']:.3f}", flush=True)
                srs.append(r_rs["sr"]); cols.append(r_rs["collision"]); lands.append(r_rs["landing_p50"])
                out.setdefault("flow_frozen_sr", []).append(r_fz["sr"])
                if seed == args.seeds[0]:
                    seed0 = {"traces": r_fz["traces"]}
                continue
            if kind in ("ht", "ht1", "ht05", "hg"):
                net, _ = train_nll("ht" if kind == "ht" else kind, st, ac, seed,
                                   args.width, args.steps, 512, 1e-3, device)
            elif kind == "flow":
                net, _ = train_flow(st, ac, seed, args.width, args.steps, 512, 1e-3, device)
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
              f"collision={np.mean(cols):.3f}  landing_p50={np.nanmedian(lands):+.2f}",
              flush=True)
    Path("analysis").mkdir(exist_ok=True)
    outp = os.environ.get("OB_OUT", "analysis/toy2d_obstacle.json")
    Path(outp).write_text(json.dumps(out, indent=1))
    print(f"WROTE {outp}", flush=True)


if __name__ == "__main__":
    main()
