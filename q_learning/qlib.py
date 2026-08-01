"""Shared library for the MIP/HT-vs-FFN Q-learning experiments.

Built on geyang/ffn (cloned in ./ffn): reuses their RandMDP toy environment,
tabular value iteration, 4x400 MLP architecture, LFF layer (verbatim from
toy_mdp/value_iteration_lff.py), RMSprop lr=1e-4, and the fitted-VI loop with
per-epoch target sync. Adds critic objective variants ported from
mip/losses.py: hetero-Student-t NLL (regression_hetero_t_loss), hetero-Gauss
NLL (regression_hetero_gauss_loss), and a scalar two-step MIP regression
(mip_loss, t_two_step=0.9).
"""
import sys
import types
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

# --- minimal gym stub so ffn/toy_mdp/rand_mdp.py imports without gym ---
if "gym" not in sys.modules:
    gym = types.ModuleType("gym")
    spaces = types.ModuleType("gym.spaces")

    class _Env:
        pass

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    gym.Env = _Env
    spaces.Discrete = _Discrete
    spaces.Box = _Box
    gym.spaces = spaces
    sys.modules["gym"] = gym
    sys.modules["gym.spaces"] = spaces

_here = __file__.rsplit("/", 1)[0]
sys.path.insert(0, _here + "/ffn/toy_mdp")
try:
    from rand_mdp import RandMDP  # noqa: E402
except ModuleNotFoundError:  # pod side: nested clone not synced
    sys.path.insert(0, _here)
    from rand_mdp_vendored import RandMDP  # noqa: E402


def make_mdp(num_states=100, option="fixed", seed=0):
    env = RandMDP(seed=seed, option=option)
    states, rewards, dyn_mats = env.get_discrete_mdp(num_states=num_states)
    return states, rewards, dyn_mats


def perform_vi(rewards, dyn_mats, gamma=0.9, eps=1e-8):
    # theirs (value_iteration_lff.perform_vi)
    q_values = np.zeros(dyn_mats.shape[:2])
    delta = 1.0
    while delta >= eps:
        old = q_values
        q_max = q_values.max(axis=0)
        q_values = rewards + gamma * dyn_mats @ q_max
        delta = np.abs(old - q_values).max()
    return q_values


class LFF(nn.Module):
    # verbatim from ffn/toy_mdp/value_iteration_lff.py
    def __init__(self, input_dim, mapping_size, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = mapping_size * 2
        self.linear = nn.Linear(input_dim, self.output_dim)
        nn.init.normal_(self.linear.weight, 0, scale / self.input_dim)

    def forward(self, x):
        x = self.linear(2 * np.pi * x)
        return torch.sin(x)


def _trunk(in_dim, feat):
    if feat == "mlp":
        first = [nn.Linear(in_dim, 400), nn.ReLU()]
    else:  # "lff<b>"
        b = float(feat[3:])
        first = [LFF(in_dim, 50, scale=b), nn.Linear(100, 400), nn.ReLU()]
    return nn.Sequential(*first,
                         nn.Linear(400, 400), nn.ReLU(),
                         nn.Linear(400, 400), nn.ReLU(),
                         nn.Linear(400, 400), nn.ReLU())


class QNet(nn.Module):
    """Their 4x400 net + a scale head (used only by ht/hg)."""

    def __init__(self, feat="mlp"):
        super().__init__()
        self.trunk = _trunk(1, feat)
        self.q_head = nn.Linear(400, 2)
        self.s_head = nn.Linear(400, 2)

    def forward(self, s):
        h = self.trunk(s)
        return self.q_head(h), self.s_head(h)


class MIPQNet(nn.Module):
    """Scalar port of the MIP flow map: f(anchor, t, state) -> Q(2)."""

    def __init__(self, feat="mlp"):
        super().__init__()
        self.trunk = _trunk(4, feat)  # state(1) + anchor(2) + t(1)
        self.q_head = nn.Linear(400, 2)

    def forward(self, s, anchor, t):
        h = self.trunk(torch.cat(
            [s, anchor, t.expand(len(s), 1)], dim=1))
        return self.q_head(h)


T2 = 0.9  # t_two_step, H4 optimum


def mip_predict(net, s):
    t0 = torch.zeros(1, device=s.device)
    a1 = net(s, torch.zeros(len(s), 2, device=s.device), t0)
    return net(s, a1, t0 + T2)


def make_net(arch, loss):
    feat = arch  # "mlp" or "lff<b>"
    if loss == "mip":
        return MIPQNet(feat)
    return QNet(feat)


def q_of(net, loss, s):
    """Deployed Q estimate (used for eval and for target-net max)."""
    if loss == "mip":
        return mip_predict(net, s)
    return net(s)[0]


def fit_loss(net, loss, s, y, gen, aux=None):
    """One full-batch objective value toward targets y (n,2).

    Rebalancing arms (user claim: FFN ~ weight rebalancing):
      pw05/pw1/pw2 — per-sample w ∝ (r²)^p, p=0.5/1/2, mean-normalized,
        detached (focal/PER-style hard-example UP-weighting)
      pema — w ∝ EMA of r² across epochs (persistent-error up-weighting,
        the convergence-rate-equalizing analog); needs aux dict
      ipw — w ∝ 1/EMA(r²) (precision weighting control, the WRONG
        direction — expected to reproduce the ht/hg clean-NFQ failure)
    """
    if loss == "mip":
        std = y.std().detach() + 1e-6
        t0 = torch.zeros(1, device=s.device)
        pred0 = net(s, torch.zeros(len(s), 2, device=s.device), t0)
        noise = torch.empty_like(y).normal_(0, 1, generator=gen)
        act_t = y + (1 - T2) * std * noise
        pred1 = net(s, act_t, t0 + T2)
        return (((pred0 - y) / T2) ** 2).mean() + \
            (((pred1 - y) / (1 - T2)) ** 2).mean()
    q, s_raw = net(s)
    r2 = (q - y) ** 2
    if loss == "mse":
        return r2.mean()
    if loss.startswith("pw"):
        p = {"05": 0.5, "1": 1.0, "2": 2.0}[loss[2:]]
        w = (r2.detach() + 1e-12) ** p
        w = w / w.mean()
        return (w * r2).mean()
    if loss in ("pema", "ipw"):
        r2d = r2.detach()
        aux["ema"] = 0.9 * aux["ema"] + 0.1 * r2d if "ema" in aux else r2d
        e = aux["ema"] + 1e-12
        w = e if loss == "pema" else 1.0 / e
        w = w / w.mean()
        return (w * r2).mean()
    if loss == "huber":
        return nn.functional.smooth_l1_loss(q, y)
    sigma = nn.functional.softplus(s_raw) + 1e-3
    if loss == "hg":
        return (0.5 * r2 / sigma ** 2 + torch.log(sigma)).mean()
    if loss == "ht":
        nu = 2.0
        return (0.5 * (nu + 1.0) * torch.log1p(r2 / (nu * sigma ** 2))
                + torch.log(sigma)).mean()
    raise ValueError(loss)


def train_supervised(arch, loss, states, targets, q_star, seed, n_epochs=4000,
                     lr=1e-4, device="cpu", eval_every=100):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    net = make_net(arch, loss).to(device)
    optim = torch.optim.RMSprop(net.parameters(), lr=lr)
    s = torch.FloatTensor(states).unsqueeze(-1).to(device)
    y = torch.FloatTensor(targets.T).to(device)  # (n,2)
    qs = torch.FloatTensor(q_star.T).to(device)
    curve = []
    aux = {}
    for epoch in range(n_epochs + 1):
        if epoch % eval_every == 0:
            with torch.no_grad():
                rmse = ((q_of(net, loss, s) - qs) ** 2).mean().sqrt().item()
            curve.append(rmse)
        l = fit_loss(net, loss, s, y, gen, aux)
        optim.zero_grad()
        l.backward()
        optim.step()
    return curve


def train_nfq(arch, loss, states, rewards, dyn_mats, q_star, seed,
              n_epochs=4000, lr=1e-4, gamma=0.9, device="cpu",
              reward_noise=None, eval_every=100):
    """Their perform_deep_vi loop (target sync every epoch), objective
    swapped. reward_noise: None | ("t", df, scale) | ("g", scale) —
    resampled every epoch (stochastic-return model)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 7)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    net = make_net(arch, loss).to(device)
    target_net = deepcopy(net)
    optim = torch.optim.RMSprop(net.parameters(), lr=lr)
    s = torch.FloatTensor(states).unsqueeze(-1).to(device)
    R = torch.FloatTensor(rewards).to(device)  # (2,n)
    P = torch.FloatTensor(dyn_mats).to(device)  # (2,n,n)
    qs = torch.FloatTensor(q_star.T).to(device)
    curve = []
    aux = {}
    for epoch in range(n_epochs + 1):
        target_net.load_state_dict(net.state_dict())
        with torch.no_grad():
            q_max = q_of(target_net, loss, s).max(dim=-1).values  # (n,)
            Re = R
            if reward_noise is not None:
                if reward_noise[0] == "t":
                    eps = rng.standard_t(reward_noise[1], size=R.shape) \
                        * reward_noise[2]
                else:
                    eps = rng.randn(*R.shape) * reward_noise[1]
                Re = R + torch.FloatTensor(eps).to(device)
            td = (Re + gamma * (P @ q_max)).T  # (n,2)
        if epoch % eval_every == 0:
            with torch.no_grad():
                rmse = ((q_of(net, loss, s) - qs) ** 2).mean().sqrt().item()
            curve.append(rmse)
        l = fit_loss(net, loss, s, td, gen, aux)
        optim.zero_grad()
        l.backward()
        optim.step()
    return curve
