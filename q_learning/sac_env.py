"""SAC with critic-objective variants + actor precision-weighting, on
self-contained envs: pendulum | mcc (MountainCarContinuous, gym constants —
the FFN repo's own spectrally-hard example, sparse reward).

New vs sac_pendulum.py: --env mcc, and loss 'htpw' = HT critic + actor loss
reweighted per-sample by normalized detached precision 1/sigma^2(s,a)
(the user's "q-learning reweighting for the actor training").

Usage: python sac_env.py <arch> <loss> <seed> [--env mcc] [--steps N]
       [--tnoise S] [--out f.jsonl]
"""
import argparse
import json
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from qlib import LFF  # noqa: E402
from sac_pendulum import Pendulum, trunk  # noqa: E402,F401

T2 = 0.9


class MountainCar:
    """gym MountainCarContinuous-v0 dynamics, verbatim constants."""

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.pos = self.rng.uniform(-0.6, -0.4)
        self.vel = 0.0
        self.t = 0
        return self.obs()

    def obs(self):
        return np.array([self.pos, self.vel], dtype=np.float32)

    def step(self, a):
        a = float(np.clip(a, -1, 1))
        self.vel += a * 0.0015 - 0.0025 * np.cos(3 * self.pos)
        self.vel = np.clip(self.vel, -0.07, 0.07)
        self.pos += self.vel
        if self.pos < -1.2:
            self.pos, self.vel = -1.2, 0.0
        self.t += 1
        done_goal = self.pos >= 0.45
        r = -0.1 * a * a + (100.0 if done_goal else 0.0)
        return self.obs(), r, done_goal or self.t >= 999


ENVS = {"pendulum": (Pendulum, 3, 2.0, 200),
        "mcc": (MountainCar, 2, 1.0, 999)}


class Critic(nn.Module):
    def __init__(self, arch, obs_dim):
        super().__init__()
        self.tr = trunk(obs_dim + 1, arch)
        self.q = nn.Linear(256, 1)
        self.s = nn.Linear(256, 1)

    def forward(self, o, a):
        h = self.tr(torch.cat([o, a], 1))
        return self.q(h), self.s(h)


class MIPCritic(nn.Module):
    def __init__(self, arch, obs_dim):
        super().__init__()
        self.tr = trunk(obs_dim + 3, arch)
        self.q = nn.Linear(256, 1)

    def forward(self, o, a, anchor, t):
        h = self.tr(torch.cat([o, a, anchor, t.expand(len(o), 1)], 1))
        return self.q(h)


class Actor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.tr = trunk(obs_dim, "mlp")
        self.mu = nn.Linear(256, 1)
        self.logstd = nn.Linear(256, 1)

    def forward(self, o):
        h = self.tr(o)
        mu, logstd = self.mu(h), torch.clamp(self.logstd(h), -5, 2)
        std = logstd.exp()
        eps = torch.randn_like(mu)
        pre = mu + std * eps
        a = torch.tanh(pre)
        logp = (-0.5 * (eps ** 2 + np.log(2 * np.pi)) - logstd
                - torch.log(1 - a ** 2 + 1e-6)).sum(1, keepdim=True)
        return a, logp


def mip_q(net, o, a):
    t0 = torch.zeros(1, device=o.device)
    q1 = net(o, a, torch.zeros(len(o), 1, device=o.device), t0)
    return net(o, a, q1, t0 + T2)


def critic_td_loss(loss, qnet, o, a, y):
    if loss == "mip":
        std = y.std().detach() + 1e-6
        t0 = torch.zeros(1, device=o.device)
        p0 = qnet(o, a, torch.zeros(len(o), 1, device=o.device), t0)
        yt = y + (1 - T2) * std * torch.randn_like(y)
        p1 = qnet(o, a, yt, t0 + T2)
        return (((p0 - y) / T2) ** 2).mean() + \
            (((p1 - y) / (1 - T2)) ** 2).mean()
    q, s_raw = qnet(o, a)
    r2 = (q - y) ** 2
    if loss == "mse":
        return r2.mean()
    if loss == "huber":
        return F.smooth_l1_loss(q, y)
    sig = F.softplus(s_raw) + 1e-3
    if loss == "hg":
        return (0.5 * r2 / sig ** 2 + torch.log(sig)).mean()
    if loss == "htn":
        # HT in its BC operating regime: residuals in running-std-of-target
        # units + sigma floor (blocks the small-error attractor)
        ystd = y.std().detach() + 1e-3
        prev = getattr(qnet, "_ystd", None)
        ys = 0.99 * prev + 0.01 * ystd if prev is not None else ystd
        qnet._ystd = ys
        r2 = r2 / ys ** 2
        sig = F.softplus(s_raw) + 0.1
        nu = 2.0
        return (0.5 * (nu + 1) * torch.log1p(r2 / (nu * sig ** 2))
                + torch.log(sig)).mean()
    if loss in ("ht", "htpw"):
        nu = 2.0
        return (0.5 * (nu + 1) * torch.log1p(r2 / (nu * sig ** 2))
                + torch.log(sig)).mean()
    raise ValueError(loss)


def q_eval(loss, qnet, o, a):
    return mip_q(qnet, o, a) if loss == "mip" else qnet(o, a)[0]


def evaluate(actor, dev, seed, Env, n_ep=5):
    rets = []
    for k in range(n_ep):
        env = Env(10000 + seed * 10 + k)
        o, ret, done = env.reset(), 0.0, False
        while not done:
            with torch.no_grad():
                mu = actor.mu(actor.tr(torch.tensor(o, device=dev)[None]))
            o, r, done = env.step(np.tanh(mu.item()))
            ret += r
        rets.append(ret)
    return float(np.mean(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch")
    ap.add_argument("loss")
    ap.add_argument("seed", type=int)
    ap.add_argument("--env", default="mcc")
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--tnoise", type=float, default=0.0)
    ap.add_argument("--out", default="results_sac_env.jsonl")
    cf = ap.parse_args()
    Env, OD, ASCALE, _ = ENVS[cf.env]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cf.seed)
    rng = np.random.RandomState(cf.seed)

    Qc = MIPCritic if cf.loss == "mip" else Critic
    q1, q2 = Qc(cf.arch, OD).to(dev), Qc(cf.arch, OD).to(dev)
    q1t, q2t = deepcopy(q1), deepcopy(q2)
    actor = Actor(OD).to(dev)
    log_alpha = torch.zeros(1, device=dev, requires_grad=True)
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 1e-3)
    oa = torch.optim.Adam(actor.parameters(), 1e-3)
    ol = torch.optim.Adam([log_alpha], 1e-3)
    tgt_ent = -1.0

    N = 200000
    buf = {k: np.zeros((N, d), dtype=np.float32)
           for k, d in [("o", OD), ("a", 1), ("r", 1), ("o2", OD), ("d", 1)]}
    ptr, full = 0, False
    warm = 20000 if cf.env == "mcc" else 2000
    env = Env(cf.seed)
    o = env.reset()
    curve = []
    n_goal = 0
    a_hold = np.zeros(1)
    for step in range(cf.steps):
        if step < warm:
            if step % 40 == 0:
                a_hold = rng.uniform(-1, 1, size=1)
            a = a_hold
        else:
            with torch.no_grad():
                a, _ = actor(torch.tensor(o, device=dev)[None])
            a = a.cpu().numpy()[0]
        o2, r, done = env.step(a[0] * ASCALE)
        if cf.tnoise > 0:
            r = r + rng.standard_t(2) * cf.tnoise
        for k, v in [("o", o), ("a", a), ("r", [r]), ("o2", o2),
                     ("d", [float(done)])]:
            buf[k][ptr] = v
        ptr = (ptr + 1) % N
        full = full or ptr == 0
        if done and getattr(env, "t", 999) < 999 and cf.env == "mcc":
            n_goal += 1
        o = env.reset() if done else o2

        if step >= warm:
            idx = rng.randint(0, N if full else ptr, size=256)
            B = {k: torch.tensor(v[idx], device=dev) for k, v in buf.items()}
            with torch.no_grad():
                a2, logp2 = actor(B["o2"])
                qt = torch.min(q_eval(cf.loss, q1t, B["o2"], a2),
                               q_eval(cf.loss, q2t, B["o2"], a2))
                y = B["r"] + 0.99 * (1 - B["d"]) * \
                    (qt - log_alpha.exp() * logp2)
            lq = critic_td_loss(cf.loss, q1, B["o"], B["a"], y) + \
                critic_td_loss(cf.loss, q2, B["o"], B["a"], y)
            oq.zero_grad()
            lq.backward()
            oq.step()

            an, logp = actor(B["o"])
            qpi = torch.min(q_eval(cf.loss, q1, B["o"], an),
                            q_eval(cf.loss, q2, B["o"], an))
            per = log_alpha.exp().detach() * logp - qpi
            if cf.loss == "htpw":
                with torch.no_grad():
                    s1 = F.softplus(q1(B["o"], an)[1]) + 1e-3
                    s2 = F.softplus(q2(B["o"], an)[1]) + 1e-3
                    w = 1.0 / (0.5 * (s1 ** 2 + s2 ** 2))
                    w = w / w.mean()
                la = (w * per).mean()
            else:
                la = per.mean()
            oa.zero_grad()
            la.backward()
            oa.step()
            ll = (-log_alpha.exp() * (logp.detach() + tgt_ent)).mean()
            ol.zero_grad()
            ll.backward()
            ol.step()
            with torch.no_grad():
                for tn, on in [(q1t, q1), (q2t, q2)]:
                    for pt, p in zip(tn.parameters(), on.parameters()):
                        pt.mul_(0.995).add_(0.005 * p)

        if (step + 1) % 5000 == 0:
            ret = evaluate(actor, dev, cf.seed, Env)
            curve.append(ret)
            print(f"QSE {cf.env} {cf.arch} {cf.loss} s{cf.seed} "
                  f"step {step + 1} ret {ret:.1f} goals {n_goal}",
                  flush=True)

    rec = {"env": cf.env, "arch": cf.arch, "loss": cf.loss, "seed": cf.seed,
           "tnoise": cf.tnoise, "final": float(np.mean(curve[-3:])),
           "best": max(curve), "curve": curve}
    with open(cf.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"QSE DONE {cf.env} {cf.arch} {cf.loss} s{cf.seed} "
          f"final {rec['final']:.1f} best {rec['best']:.1f} "
          f"goals {n_goal}", flush=True)


if __name__ == "__main__":
    main()
