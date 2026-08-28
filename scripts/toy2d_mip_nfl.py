"""Train the proposed 2D no-free-lunch counterexample for MIP.

The task is a point robot moving through a narrowing 2D funnel into a tight
dock.  At every recovery-covered state, the clean lateral controller is a
linear servo.  Demonstrated action chunks additionally contain a latent
zero-mean nuisance residual.  In the primary cell that residual is skewed:

    r = -b with probability .8, and r = +4b with probability .2.

Thus regression's population target is the clean servo.  MIP step 1 has the
same target, while its second view is an action denoiser trained around the
two demonstrated modes.  At inference the second view receives the step-1
mean, and the population denoiser maps that mean toward the nearer dominant
mode.  Replanning turns the resulting per-cycle bias into a docking miss.

The ``contaminated`` cell reverses the winner: 70% of labels are the clean
servo, 20% have a mild +0.2 residual, and 10% are gross +1.6 outliers. The
label mean is exactly +0.2. HT should robustly select the clean majority mode;
L2 fits the corrupt mean, and MIP denoises that mean onto the mild corrupt mode.

Symmetric and Gaussian-like antithetic residuals are falsifier cells.  The
regression and MIP arms use the exact same network class and parameter count.
HT and HG use the same backbone plus the repository's scalar scale head. Their
losses directly port regression_hetero_t_loss and
regression_hetero_gauss_loss from mip/losses.py.

Example:
    python scripts/toy2d_mip_nfl.py --modes skew --seeds 0 --steps 6000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import os
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]

# World/controller units.
X_GOAL_MM = 160.0
FORWARD_MM = 4.0
ACTION_MM = 10.0             # one normalized action unit
K_SERVO = 0.5
DOCK_TOL_MM = 0.5
ANCHOR_SIGMA = 0.1           # 1 - t_two_step in normalized action units
T_TWO_STEP = 0.9
B = 0.1                      # primary residual: -0.1 / +0.4 normalized


def half_width_mm(x: np.ndarray) -> np.ndarray:
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def normalize_state(state: np.ndarray) -> np.ndarray:
    out = np.empty_like(state, dtype=np.float32)
    out[:, 0] = state[:, 0] / 80.0 - 1.0
    out[:, 1] = state[:, 1] / 14.0
    return out


def make_chunks(state: np.ndarray, residual: np.ndarray, horizon: int) -> np.ndarray:
    """Generate dynamically consistent action chunks for a coherent nuisance.

    The residual is a hidden actuator/operator offset during the chunk.  The
    servo responds to earlier offset actions in later elements of the chunk.
    Linearity guarantees E[chunk | state] == chunk(residual=0) whenever
    E[residual]=0.
    """
    n = len(state)
    y = state[:, 1].astype(np.float64).copy()
    x = state[:, 0].astype(np.float64).copy()
    r_mm = residual.astype(np.float64) * ACTION_MM
    phase_damp = bool(int(__import__("os").environ.get("PHASE_DAMP", "0")))
    action = np.empty((n, 2 * horizon), dtype=np.float32)
    for j in range(horizon):
        rj = r_mm
        if phase_damp:
            rj = np.where(x >= PHASE_X_MM, 0.0, r_mm)
        dy = -K_SERVO * y + rj
        action[:, 2 * j] = FORWARD_MM / ACTION_MM
        action[:, 2 * j + 1] = dy / ACTION_MM
        y += dy
        x += FORWARD_MM
    return action


PHASE_X_MM = 130.0           # transit/steady-hand boundary
PHASE_CLIP = 0.30            # actuator safety clip (normalized units)
PHASE_FINAL_SCALE = 0.006    # steady-hand jitter in the final approach


def residual_replicates(mode: str, n_states: int, repeats: int,
                        rng: np.random.Generator,
                        x_state: np.ndarray | None = None) -> np.ndarray:
    if repeats % 10 != 0:
        raise ValueError("repeats must be divisible by 10")
    if mode == "skew":
        base = np.array([-B] * 8 + [4 * B] * 2, dtype=np.float32)
        return np.tile(base, (n_states, repeats // 10)).reshape(n_states, repeats)
    if mode == "symmetric":
        base = np.array([-B] * 5 + [B] * 5, dtype=np.float32)
        return np.tile(base, (n_states, repeats // 10)).reshape(n_states, repeats)
    if mode == "slots":
        # three-slot fork: label variance = between-operator AIM CHOICE,
        # all executed, all successful. No noise channel anywhere.
        u = rng.random((n_states, repeats))
        out = np.empty((n_states, repeats))
        c0 = u < 0.60
        c1 = (u >= 0.60) & (u < 0.88)
        c2 = ~c0 & ~c1
        out[c0] = rng.normal(0.000, 0.004, c0.sum())
        out[c1] = rng.normal(-0.300, 0.015, c1.sum())
        out[c2] = rng.normal(-0.550, 0.025, c2.sum())
        return out.astype(np.float32)
    if mode == "midlock":
        # dominant tight middle (true action); minority bottom branch at
        # 1.7 anchor-sigma; rare ONE-SIDED far glitches place the label
        # MEAN exactly on the bottom branch. Mean-family and MIP's anchor
        # land on the bottom; HT nu=2's redescending influence discounts
        # both far components (far mass 0.28 < 1/3) and locks the middle.
        u = rng.random((n_states, repeats))
        out = np.empty((n_states, repeats))
        m = u < 0.72
        b = (u >= 0.72) & (u < 0.94)
        t = ~m & ~b
        out[m] = rng.normal(0.000, 0.003, m.sum())
        out[b] = rng.normal(-0.170, 0.015, b.sum())
        mu_top = (-0.170 - 0.22 * -0.170) / 0.06
        out[t] = mu_top + np.clip(rng.standard_t(df=1.05, size=t.sum())
                                  * 0.10, -1.0, 1.0)
        return out.astype(np.float32)
    if mode == "midlock":
        # dominant tight middle; minority bottom branch at 1.7 anchor-sigma;
        # rare one-sided far glitches put the label MEAN on the bottom
        # branch (defeats mean-family + MIP anchor; HT nu=2 locks middle).
        u = rng.random((n_states, repeats))
        out = np.empty((n_states, repeats))
        m = u < 0.72
        b = (u >= 0.72) & (u < 0.94)
        t = ~m & ~b
        out[m] = rng.normal(0.000, 0.003, m.sum())
        out[b] = rng.normal(-0.170, 0.015, b.sum())
        out[t] = -2.2117 + np.clip(rng.standard_t(df=1.05, size=t.sum())
                                   * 0.10, -1.0, 1.0)
        return out.astype(np.float32)
    if mode == "threeway":
        # three operator styles: dense bottom / ultra-tight middle (true) /
        # heavy top counterweight (zero mean overall).
        u = rng.random((n_states, repeats))
        out = np.empty((n_states, repeats))
        b = u < 0.48
        m = (u >= 0.48) & (u < 0.904)
        t = ~b & ~m
        out[b] = rng.normal(-0.060, 0.015, b.sum())
        out[m] = rng.normal(0.000, 0.003, m.sum())
        out[t] = 0.30 + np.clip(rng.standard_t(df=1.05, size=t.sum())
                                * 0.060, -0.6, 0.6)
        return out.astype(np.float32)
    if mode == "phase":
        # EXECUTED phase-dependent noise, mirroring the measured structure
        # of real human demos (align/transit heavy-tailed, insertion
        # smooth): transit states get Student-t df=1.05 jitter (clipped by
        # actuator safety), final-approach states a tiny Gaussian tremor.
        # The demonstrator executes this and still docks (oracle check).
        assert x_state is not None
        r = rng.standard_t(df=1.05, size=(n_states, repeats)) * B
        r = np.clip(r, -PHASE_CLIP, PHASE_CLIP)
        calm = rng.normal(0.0, PHASE_FINAL_SCALE, size=(n_states, repeats))
        final = (x_state >= PHASE_X_MM)[:, None]
        return np.where(final, calm, r).astype(np.float32)
    if mode == "heavy":
        # symmetric zero-mean heavy-tailed RECORDED-ONLY corruption
        # (teleop/logging glitches): the demonstrator executes the clean
        # servo (oracle SR = 1 by construction); the labels carry
        # Student-t df=1.05 noise with rare huge spikes.
        r = rng.standard_t(df=1.05, size=(n_states, repeats)) * B
        return np.clip(r, -40 * B, 40 * B).astype(np.float32)
    if mode == "contaminated":
        # 70% correct labels, 20% mild corruption, 10% gross corruption.
        # Their mean is +0.2: 0.7*0 + 0.2*0.2 + 0.1*1.6 = 0.2.
        base = np.array([0.0] * 7 + [2 * B] * 2 + [16 * B], dtype=np.float32)
        return np.tile(base, (n_states, repeats // 10)).reshape(n_states, repeats)
    if mode == "gaussian":
        # Antithetic draws make every state's empirical distribution exactly
        # symmetric and zero mean, while retaining a Gaussian-like continuum.
        z = rng.normal(0.0, 0.2, size=(n_states, repeats // 2)).astype(np.float32)
        return np.concatenate([z, -z], axis=1)
    raise ValueError(f"unknown mode: {mode}")


def make_dataset(mode: str, n_states: int, repeats: int, horizon: int,
                 seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, X_GOAL_MM - FORWARD_MM, size=n_states)
    hw = half_width_mm(x)
    y = rng.uniform(-0.85 * hw, 0.85 * hw)
    base_state = np.stack([x, y], axis=1).astype(np.float32)
    residual = residual_replicates(mode, n_states, repeats, rng, x_state=x)

    state = np.repeat(base_state, repeats, axis=0)
    action = make_chunks(state, residual.reshape(-1), horizon)
    state = normalize_state(state)

    order = rng.permutation(len(state))
    return torch.from_numpy(state[order]), torch.from_numpy(action[order])


class ViewNet(nn.Module):
    """One network shared by regression and both MIP views."""

    def __init__(self, action_dim: int, width: int, scalar_head: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.scalar_head = scalar_head
        self.net = nn.Sequential(
            nn.Linear(2 + action_dim + 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, action_dim + int(scalar_head)),
        )
        if scalar_head:
            # Match the repository's zero-initialized scalar output head.
            with torch.no_grad():
                self.net[-1].weight[action_dim].zero_()
                self.net[-1].bias[action_dim].zero_()

    def forward(self, state: torch.Tensor, anchor: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, anchor, t], dim=1))


def train_nll(kind: str, state: torch.Tensor, action: torch.Tensor,
              seed: int, width: int, steps: int, batch_size: int, lr: float,
              device: torch.device) -> tuple[ViewNet, dict]:
    """Exact toy ports of repository HG / HT (nu=2) objectives.

    Both losses use one learned scale per state and the squared norm of the
    entire action chunk. A diagonal per-coordinate Student-t would be a
    different estimator from mip/losses.py::regression_hetero_t_loss.
    """
    torch.manual_seed(seed)
    A = action.shape[1]
    net = ViewNet(A, width, scalar_head=True).to(device)
    state = state.to(device)
    action = action.to(device)
    raw_nu = torch.nn.Parameter(torch.tensor(1.5, device=device))
    params = list(net.parameters()) + ([raw_nu] if kind == "htl" else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    n = len(state)
    started = time.time()
    net.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        sb, ab = state[idx], action[idx]
        z0 = torch.zeros((batch_size, A), device=device)
        t0 = torch.zeros((batch_size, 1), device=device)
        out = net(sb, z0, t0)
        mu, s_raw = out[:, :A], out[:, A]
        sigma = torch.nn.functional.softplus(s_raw) + 1e-3
        sum_r2 = ((ab - mu) ** 2).sum(dim=1)
        if kind == "welsch":
            # direct reverse-KL surrogate: correntropy/Welsch with fixed
            # kernel width h (mode-seeking; h->inf recovers MSE)
            h = 0.05
            loss = (1.0 - torch.exp(-sum_r2 / (2 * h * h * A))).mean()
        elif kind == "hg":
            loss = (0.5 * sum_r2 / sigma.square() + A * sigma.log()).mean() / A
        else:
            if kind == "htl":
                nu = torch.nn.functional.softplus(raw_nu) + 0.05
            else:
                nu = torch.tensor({"ht05": 0.5, "ht1": 1.0}.get(kind, 2.0),
                                  device=device)
            per = (
                0.5 * (nu + 1.0)
                * torch.log1p(sum_r2 / (nu * sigma.square() * A))
                * A
                + A * sigma.log()
                + A * (-torch.lgamma((nu + 1.0) / 2)
                       + torch.lgamma(nu / 2) + 0.5 * torch.log(nu))
            )
            if HT_BETA > 0.0:
                per = per * sigma.detach().square() ** HT_BETA
            loss = per.mean() / A
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    info = {"seconds": time.time() - started,
            "parameters": sum(p.numel() for p in net.parameters())}
    if kind == "htl":
        info["nu_hat"] = float(torch.nn.functional.softplus(
            raw_nu).detach().cpu() + 0.05)
    return net, info


def train(kind: str, state: torch.Tensor, action: torch.Tensor, seed: int,
          width: int, steps: int, batch_size: int, lr: float,
          device: torch.device) -> tuple[ViewNet, dict]:
    torch.manual_seed(seed)
    net = ViewNet(action.shape[1], width).to(device)
    state = state.to(device)
    action = action.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(state)
    losses = []
    t0 = torch.zeros((batch_size, 1), device=device)
    t1 = torch.full((batch_size, 1), T_TWO_STEP, device=device)
    started = time.time()

    net.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        sb, ab = state[idx], action[idx]
        z0 = torch.zeros_like(ab)
        pred0 = net(sb, z0, t0)
        if kind == "regression":
            loss = ((pred0 - ab) ** 2).mean()
            loss0 = loss
            loss1 = torch.zeros_like(loss)
        elif kind == "mip":
            at = ab + ANCHOR_SIGMA * torch.randn_like(ab)
            pred1 = net(sb, at, t1)
            loss0 = (((pred0 - ab) / T_TWO_STEP) ** 2).mean()
            loss1 = (((pred1 - ab) / ANCHOR_SIGMA) ** 2).mean()
            loss = loss0 + loss1
        else:
            raise ValueError(kind)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
            losses.append({
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "loss0": float(loss0.detach().cpu()),
                "loss1": float(loss1.detach().cpu()),
            })
    net.eval()
    return net, {
        "losses": losses,
        "seconds": time.time() - started,
        "parameters": sum(p.numel() for p in net.parameters()),
    }


SIGMA_FLOOR = float(os.environ.get("HT_SIGMA_FLOOR", "1e-3"))
# beta-NLL (Seitzer et al. 2022): multiply the per-sample NLL by stop_grad(sigma^{2beta}).
# beta is a pure GAIN on the mu-gradient — it does not move the redescending knee,
# which sits at r = sigma*sqrt(nu) for every beta. What it changes is the weighting
# ACROSS states: at each state's own typical residual the gradient scales as
# sigma^(2*beta-1), so beta=0 -> sigma^-1 (favours predictable states), beta=0.5 ->
# flat, beta=1 -> sigma^+1 (the same across-state weighting as MSE/L1). Deciding
# states are the high-sigma ones, so beta>0.5 up-weights exactly the states where
# HT was measured to under-commit (see ht-redescending-optimization-gap).
HT_BETA = float(os.environ.get("HT_BETA", "0.0"))
"""Lower bound on the predicted scale. With a noiseless demonstrator most states
have ~zero residual and A*log(sigma) is unbounded below, so the optimiser drives
sigma to the floor globally; log1p(r2/(nu*sigma^2*A)) then saturates at the rare
ambiguous states and mu stops moving. Real demonstrations have non-zero residual
everywhere and do not hit this."""


def train_hist(kind: str, state, prev, action, seed: int, width: int,
               steps: int, batch_size: int, lr: float, device, log_every: int = 0):
    """Regression / HT with the PREVIOUS ACTION CHUNK in the conditioning.

    Reuses ViewNet unchanged by putting `prev` in the anchor slot (identical
    width to an action chunk), so every family — regression, HT and flow —
    sees exactly the same history information. Losses are byte-identical to
    train()/train_nll(); only the conditioning differs.
    """
    torch.manual_seed(seed)
    A = action.shape[1]
    scalar = kind in ("ht", "ht05", "ht1", "hg")
    net = ViewNet(A, width, scalar_head=scalar).to(device)
    state, prev, action = state.to(device), prev.to(device), action.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(state)
    t0 = torch.zeros((batch_size, 1), device=device)
    net.train()

    # --- optional sigma pre-training on a FROZEN base ------------------------
    # Idea under test: instead of a scalar sbias init (which assumes the residual
    # scale is the same everywhere), freeze mu and fit the sigma head per-sample
    # to the base model's own residuals, so HT starts from a calibrated
    # heteroscedastic sigma. A perfectly fitted sigma gives sigma_i^2 -> m_i, i.e.
    # u_i = m_i/(nu*sigma_i^2) -> 1/nu for EVERY sample: the gate is neutral at
    # init and only starts discriminating as mu improves unevenly.
    # RISK the run is meant to expose: at init the residual is mostly EPISTEMIC
    # (mu is untrained), so this may teach sigma "this sample is noisy" when the
    # truth is "mu has not learned it yet" -- baking in the undersampling.
    # HT_SIGMA_WARM_BASE>0 first trains mu with MSE for that many steps, giving a
    # competent base whose residual is closer to aleatoric.
    n_pre = int(os.environ.get("HT_SIGMA_PRETRAIN", "0"))
    n_warm = int(os.environ.get("HT_SIGMA_WARM_BASE", "0"))
    if scalar and n_pre > 0:
        if n_warm > 0:  # give mu a head start so residuals are less epistemic
            for _ in range(n_warm):
                idx = torch.randint(0, n, (batch_size,), device=device)
                loss = ((net(state[idx], prev[idx], t0)[:, :A] - action[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        # freeze everything except the sigma row of the output layer
        head_w, head_b = net.net[-1].weight, net.net[-1].bias
        opt_s = torch.optim.AdamW([head_w, head_b], lr=lr, weight_decay=0.0)
        nu_pre = float({"ht05": 0.5, "ht1": 1.0}.get(kind, 2.0))
        for _ in range(n_pre):
            idx = torch.randint(0, n, (batch_size,), device=device)
            out = net(state[idx], prev[idx], t0)
            mu, s_raw = out[:, :A].detach(), out[:, A]     # mu frozen
            sigma = torch.nn.functional.softplus(s_raw) + SIGMA_FLOOR
            sum_r2 = ((action[idx] - mu) ** 2).sum(dim=1)
            per = (0.5 * (nu_pre + 1.0)
                   * torch.log1p(sum_r2 / (nu_pre * sigma.square() * A)) * A
                   + A * sigma.log())
            loss = per.mean() / A
            opt_s.zero_grad(set_to_none=True)
            loss.backward()
            # keep ONLY the sigma output row; zero the mu rows of the head
            with torch.no_grad():
                if head_w.grad is not None:
                    head_w.grad[:A].zero_()
                if head_b.grad is not None:
                    head_b.grad[:A].zero_()
            opt_s.step()
        with torch.no_grad():
            out = net(state[:512], prev[:512], t0[:1].expand(min(512, n), 1))
            sg = torch.nn.functional.softplus(out[:, A]) + SIGMA_FLOOR
            rr = ((action[:512] - out[:, :A]) ** 2).mean(dim=1).sqrt()
            print(f"[sigma-pretrain] steps={n_pre} warm={n_warm} "
                  f"sigma[q10/med/q90]={sg.quantile(0.1):.4f}/{sg.median():.4f}/{sg.quantile(0.9):.4f} "
                  f"rms[med]={rr.median():.4f} corr(sigma,rms)="
                  f"{torch.corrcoef(torch.stack([sg, rr]))[0,1]:.3f}", flush=True)

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        sb, pb, ab = state[idx], prev[idx], action[idx]
        out = net(sb, pb, t0)
        if not scalar:
            loss = ((out[:, :A] - ab) ** 2).mean()
        else:
            mu, s_raw = out[:, :A], out[:, A]
            sigma = torch.nn.functional.softplus(s_raw) + SIGMA_FLOOR
            sum_r2 = ((ab - mu) ** 2).sum(dim=1)
            if kind == "hg":
                loss = (0.5 * sum_r2 / sigma.square() + A * sigma.log()).mean() / A
            else:
                nu = torch.tensor({"ht05": 0.5, "ht1": 1.0}.get(kind, 2.0), device=device)
                per = (0.5 * (nu + 1.0) * torch.log1p(sum_r2 / (nu * sigma.square() * A)) * A
                       + A * sigma.log()
                       + A * (-torch.lgamma((nu + 1.0) / 2) + torch.lgamma(nu / 2)
                              + 0.5 * torch.log(nu)))
                if HT_BETA > 0.0:
                    per = per * sigma.detach().square() ** HT_BETA
                loss = per.mean() / A
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"      [{kind}] step {step:6d}  loss {float(loss):.5f}", flush=True)
    net.eval()
    return net


@torch.inference_mode()
def predict_hist(net: ViewNet, state_world, prev, device):
    """Single forward pass with the previous chunk as conditioning."""
    state = torch.from_numpy(normalize_state(state_world.astype(np.float32))).to(device)
    t0 = torch.zeros((len(state), 1), device=device)
    out = net(state, prev, t0)
    return out[:, :net.action_dim]


@torch.inference_mode()
def predict(net: ViewNet, state_world: np.ndarray, sampler: str,
            device: torch.device) -> np.ndarray:
    state = torch.from_numpy(normalize_state(state_world.astype(np.float32))).to(device)
    action_dim = net.action_dim
    if sampler in ("hg", "ht"):
        t0 = torch.zeros((len(state), 1), device=device)
        z0 = torch.zeros((len(state), action_dim), device=device)
        out = net(state, z0, t0)
        return out[:, :action_dim].cpu().numpy()
    z0 = torch.zeros((len(state), action_dim), device=device)
    t0 = torch.zeros((len(state), 1), device=device)
    first = net(state, z0, t0)
    if sampler in ("regression", "mip_step1"):
        out = first
    elif sampler == "mip_full":
        t1 = torch.full((len(state), 1), T_TWO_STEP, device=device)
        out = net(state, first, t1)
    else:
        raise ValueError(sampler)
    return out.cpu().numpy()


def rollout_metrics(net: ViewNet, sampler: str, device: torch.device,
                    n_eval: int = 401,
                    chunked: bool = False, horizon: int = 8) -> dict:
    x = np.zeros(n_eval, dtype=np.float64)
    y = np.linspace(-10.0, 10.0, n_eval)
    y0 = y.copy()
    active = np.ones(n_eval, dtype=bool)
    collision = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    traces_x = [[] for _ in range(n_eval)]
    traces_y = [[] for _ in range(n_eval)]

    for _ in range(70):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        state = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        action = predict(net, state, sampler, device)
        n_exec = horizon if chunked else 1
        for j in range(n_exec):
            live = active[ids]
            dx = np.clip(action[:, 2 * j] * ACTION_MM, -2.0, 8.0)
            dy = np.clip(action[:, 2 * j + 1] * ACTION_MM, -12.0, 12.0)
            x[ids] = np.where(live, x[ids] + dx, x[ids])
            y[ids] = np.where(live, y[ids] + dy, y[ids])
            for local, idx in enumerate(ids):
                if active[idx]:
                    traces_x[idx].append(float(x[idx]))
                    traces_y[idx].append(float(y[idx]))
            outside = np.abs(y[ids]) > half_width_mm(x[ids])
            collision[ids[outside & live]] = True
            reached = (x[ids] >= X_GOAL_MM) & live
            y_goal[ids[reached]] = y[ids[reached]]
            active[ids[(outside & live) | reached]] = False

    reached = np.isfinite(y_goal)
    import os as _os
    if _os.environ.get("SLOT_DOCKS"):
        cents = np.array([float(v) for v in
                          _os.environ["SLOT_DOCKS"].split(",")])
        derr = np.min(np.abs(y_goal[:, None] - cents[None, :]), axis=1)
        success = reached & (derr < float(
            _os.environ.get("SLOT_TOL", "1.0"))) & ~collision
    else:
        success = reached & (np.abs(y_goal) < DOCK_TOL_MM) & ~collision
    chosen = np.linspace(0, n_eval - 1, 11, dtype=int)
    traces = [{
        "y0": float(y0[i]),
        "x": traces_x[i],
        "y": traces_y[i],
    } for i in chosen]
    return {
        "success_rate": float(success.mean()),
        "reached_rate": float(reached.mean()),
        "collision_rate": float(collision.mean()),
        "y_goal_mean": float(np.nanmean(y_goal)),
        "abs_y_goal_p50": float(np.nanpercentile(np.abs(y_goal), 50)),
        "abs_y_goal_p90": float(np.nanpercentile(np.abs(y_goal), 90)),
        "traces": traces,
    }


def field_metrics(net: ViewNet, sampler: str, device: torch.device) -> dict:
    xx = np.repeat(np.linspace(0.0, 150.0, 31), 31)
    yy = np.tile(np.linspace(-6.0, 6.0, 31), 31)
    state = np.stack([xx, yy], axis=1).astype(np.float32)
    action = predict(net, state, sampler, device)
    clean_ay = (-K_SERVO * yy) / ACTION_MM
    residual = action[:, 1] - clean_ay
    return {
        "first_lateral_residual_mean": float(residual.mean()),
        "first_lateral_residual_p10_p50_p90": [
            float(v) for v in np.percentile(residual, [10, 50, 90])
        ],
        "first_forward_mean": float(action[:, 0].mean()),
    }


def oracle_expert_sr(mode: str, rng: np.random.Generator,
                     n_eval: int = 401) -> float:
    """Closed-loop SR of the noisy demonstrator itself (servo + executed
    per-chunk nuisance drawn from the same distribution)."""
    y = np.linspace(-10.0, 10.0, n_eval)
    x = np.zeros(n_eval)
    collision = np.zeros(n_eval, dtype=bool)
    done = np.zeros(n_eval, dtype=bool)
    y_goal = np.full(n_eval, np.nan)
    for _ in range(70):
        r = residual_replicates(mode, n_eval, 10, rng,
                                x_state=x)[:, 0] * ACTION_MM
        dy = -K_SERVO * y + r
        x = x + FORWARD_MM
        y = y + dy
        live = ~done
        collision |= live & (np.abs(y) > half_width_mm(x))
        reach = live & (x >= X_GOAL_MM)
        y_goal[reach] = y[reach]
        done |= collision | reach
        if done.all():
            break
    ok = np.isfinite(y_goal) & (np.abs(y_goal) < DOCK_TOL_MM) & ~collision
    return float(ok.mean())


def analytic_bias(mode: str, horizon: int) -> float:
    """Population deployed MIP residual after mean -> denoiser composition."""
    state = np.array([[80.0, 0.0]], dtype=np.float32)
    clean = make_chunks(state, np.array([0.0]), horizon)[0]
    if mode in ("gaussian", "heavy", "phase"):
        return 0.0
    if mode == "skew":
        modes, probs = np.array([-B, 4 * B]), np.array([0.8, 0.2])
    elif mode == "symmetric":
        modes, probs = np.array([-B, B]), np.array([0.5, 0.5])
    elif mode == "contaminated":
        modes = np.array([0.0, 2 * B, 16 * B])
        probs = np.array([0.7, 0.2, 0.1])
    elif mode == "threeway":
        modes = np.array([-0.060, 0.0, 0.30])
        probs = np.array([0.48, 0.424, 0.096])
    elif mode == "midlock":
        modes = np.array([0.0, -0.170, -2.2117])
        probs = np.array([0.72, 0.22, 0.06])
    elif mode == "slots":
        modes = np.array([0.0, -0.300, -0.550])
        probs = np.array([0.60, 0.28, 0.12])
    else:
        raise ValueError(mode)
    chunks = np.concatenate([
        make_chunks(state, np.array([r]), horizon) for r in modes
    ], axis=0)
    mean = (probs[:, None] * chunks).sum(axis=0)
    d2 = ((chunks - mean[None, :]) ** 2).sum(axis=1)
    weight = probs * np.exp(-d2 / (2 * ANCHOR_SIGMA ** 2))
    posterior = (weight[:, None] * chunks).sum(axis=0) / weight.sum()
    return float(posterior[1] - clean[1])


def analytic_ht_bias(mode: str) -> float:
    """Population HT location for the primary atomic mixture.

    Learned-scale Student-t has a stable scale-collapse solution at the 80%
    atom. Symmetric distributions retain zero location by symmetry.
    """
    return -B if mode == "skew" else 0.0


def run_cell(mode: str, seed: int, args, device: torch.device) -> dict:
    state, action = make_dataset(mode, args.n_states, args.repeats,
                                 args.horizon, args.data_seed)
    osr = (1.0 if mode == "heavy" else
           oracle_expert_sr(mode, np.random.default_rng(7)))
    # mode "phase": the oracle EXECUTES the noise — osr is the real number
    print(f"[{mode} seed={seed}] dataset={len(state)} H={args.horizon} "
          f"oracle-expert SR={osr:.3f}", flush=True)
    reg, reg_train = train("regression", state, action, seed, args.width,
                           args.steps, args.batch_size, args.lr, device)
    print(f"  regression trained {reg_train['seconds']:.1f}s", flush=True)
    mip, mip_train = train("mip", state, action, seed, args.width,
                           args.steps, args.batch_size, args.lr, device)
    print(f"  mip trained {mip_train['seconds']:.1f}s", flush=True)
    hg, hg_train = train_nll("hg", state, action, seed, args.width, args.steps,
                             args.batch_size, args.lr, device)
    print("  hg trained", flush=True)
    ht, ht_train = train_nll("ht", state, action, seed, args.width, args.steps,
                             args.batch_size, args.lr, device)
    print("  ht trained", flush=True)
    ht05, _ = train_nll("ht05", state, action, seed, args.width, args.steps,
                        args.batch_size, args.lr, device)
    wel, _ = train_nll("welsch", state, action, seed, args.width,
                       args.steps, args.batch_size, args.lr, device)
    htl, htl_info = train_nll("htl", state, action, seed, args.width,
                              args.steps, args.batch_size, args.lr, device)
    print(f"  htl trained nu_hat={htl_info.get('nu_hat', -1):.3f}",
          flush=True)

    methods = {}
    for name, net, sampler in (
        ("regression", reg, "regression"),
        ("mip_step1", mip, "mip_step1"),
        ("mip_full", mip, "mip_full"),
        ("hg", hg, "hg"),
        ("ht", ht, "ht"),
        ("ht05", ht05, "ht"),
        ("welsch", wel, "ht"),
        ("htl", htl, "ht"),
    ):
        methods[name] = {
            **field_metrics(net, sampler, device),
            **rollout_metrics(net, sampler, device, args.n_eval,
                              chunked=bool(getattr(args, "chunked", 0)),
                              horizon=args.horizon),
        }
        m = methods[name]
        print(
            f"  {name:11s} SR={m['success_rate']:.3f} "
            f"bias={m['first_lateral_residual_mean']:+.4f} "
            f"|y_goal|p50={m['abs_y_goal_p50']:.2f}mm",
            flush=True,
        )
    return {
        "mode": mode,
        "seed": seed,
        "nu_hat": htl_info.get("nu_hat"),
        "analytic_mip_first_lateral_bias": analytic_bias(mode, args.horizon),
        "analytic_ht_first_lateral_bias": analytic_ht_bias(mode),
        "regression_train": reg_train,
        "mip_train": mip_train,
        "hg_train": hg_train,
        "ht_train": ht_train,
        "methods": methods,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--modes", default="skew",
        help="comma list: skew,symmetric,gaussian,contaminated",
    )
    ap.add_argument("--seeds", default="0", help="comma-separated training seeds")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-states", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-eval", type=int, default=401)
    ap.add_argument("--data-seed", type=int, default=20260805)
    ap.add_argument("--chunked", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "analysis/toy2d_mip_nfl_results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"device={device} modes={modes} seeds={seeds}", flush=True)
    result = {
        "design": "asymmetric-zero-mean-action-jitter",
        "status": "trained-model-result",
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        "constants": {
            "x_goal_mm": X_GOAL_MM,
            "action_mm": ACTION_MM,
            "k_servo": K_SERVO,
            "dock_tol_mm": DOCK_TOL_MM,
            "anchor_sigma": ANCHOR_SIGMA,
            "t_two_step": T_TWO_STEP,
            "b": B,
        },
        "cells": [],
    }
    for mode in modes:
        for seed in seeds:
            result["cells"].append(run_cell(mode, seed, args, device))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
