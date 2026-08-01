"""E5: SAC policy training with critic-objective variants.

Self-contained pendulum swing-up (gym Pendulum-v1 dynamics reimplemented in
numpy — no gym dependency). SAC follows the sac_dennis lineage used by
geyang/ffn: twin critics, soft target update, tanh-Gaussian actor,
learned temperature. The critic TD loss is swapped between:
  mse | huber | ht (Student-t NLL, nu=2) | hg (Gauss NLL) | mip (two-step)
Arch: mlp | lff<b> (their LFF input layer on [s,a]).
Optional heavy-tailed reward noise (--tnoise scale: Student-t df=2) models
the noisy-return regime.

Usage: python sac_pendulum.py <arch> <loss> <seed> [--steps N]
       [--tnoise SCALE] [--out results_sac.jsonl]
Prints QSAC lines; appends JSON record with eval-return curve.
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
from qlib import LFF

T2 = 0.9


class Pendulum:
    """gym Pendulum-v1 dynamics, verbatim constants."""

    max_speed, max_torque, dt, g, m, l = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.th = self.rng.uniform(-np.pi, np.pi)
        self.thdot = self.rng.uniform(-1, 1)
        self.t = 0
        return self.obs()

    def obs(self):
        return np.array([np.cos(self.th), np.sin(self.th), self.thdot],
                        dtype=np.float32)

    def step(self, u):
        u = float(np.clip(u, -self.max_torque, self.max_torque))
        th_n = ((self.th + np.pi) % (2 * np.pi)) - np.pi
        cost = th_n ** 2 + 0.1 * self.thdot ** 2 + 0.001 * u ** 2
        self.thdot = np.clip(
            self.thdot + (3 * self.g / (2 * self.l) * np.sin(self.th)
                          + 3.0 / (self.m * self.l ** 2) * u) * self.dt,
            -self.max_speed, self.max_speed)
        self.th = self.th + self.thdot * self.dt
        self.t += 1
        return self.obs(), -cost, self.t >= 200


def trunk(in_dim, arch, width=256):
    if arch == "mlp":
        first = [nn.Linear(in_dim, width), nn.ReLU()]
    else:
        b = float(arch[3:])
        first = [LFF(in_dim, width // 2, scale=b),
                 nn.Linear(width, width), nn.ReLU()]
    return nn.Sequential(*first, nn.Linear(width, width), nn.ReLU())


class Critic(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.tr = trunk(4, arch)
        self.q = nn.Linear(256, 1)
        self.s = nn.Linear(256, 1)

    def forward(self, o, a):
        h = self.tr(torch.cat([o, a], 1))
        return self.q(h), self.s(h)


class MIPCritic(nn.Module):
    """f(obs, act, anchor, t) -> Q."""

    def __init__(self, arch):
        super().__init__()
        self.tr = trunk(6, arch)
        self.q = nn.Linear(256, 1)

    def forward(self, o, a, anchor, t):
        h = self.tr(torch.cat([o, a, anchor, t.expand(len(o), 1)], 1))
        return self.q(h)


def mip_q(net, o, a):
    t0 = torch.zeros(1, device=o.device)
    q1 = net(o, a, torch.zeros(len(o), 1, device=o.device), t0)
    return net(o, a, q1, t0 + T2)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tr = trunk(3, "mlp")
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
    if loss == "ht":
        nu = 2.0
        return (0.5 * (nu + 1) * torch.log1p(r2 / (nu * sig ** 2))
                + torch.log(sig)).mean()
    raise ValueError(loss)


def q_eval(loss, qnet, o, a):
    return mip_q(qnet, o, a) if loss == "mip" else qnet(o, a)[0]


def evaluate(actor, dev, seed):
    rets = []
    for k in range(5):
        env = Pendulum(10000 + seed * 10 + k)
        o, ret, done = env.reset(), 0.0, False
        while not done:
            with torch.no_grad():
                mu = actor.mu(actor.tr(torch.tensor(o, device=dev)[None]))
            o, r, done = env.step(np.tanh(mu.item()) * 2.0)
            ret += r
        rets.append(ret)
    return float(np.mean(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch")
    ap.add_argument("loss")
    ap.add_argument("seed", type=int)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--tnoise", type=float, default=0.0)
    ap.add_argument("--out", default="results_sac.jsonl")
    cf = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cf.seed)
    rng = np.random.RandomState(cf.seed)

    Qc = MIPCritic if cf.loss == "mip" else Critic
    q1, q2 = Qc(cf.arch).to(dev), Qc(cf.arch).to(dev)
    q1t, q2t = deepcopy(q1), deepcopy(q2)
    actor = Actor().to(dev)
    log_alpha = torch.zeros(1, device=dev, requires_grad=True)
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 1e-3)
    oa = torch.optim.Adam(actor.parameters(), 1e-3)
    ol = torch.optim.Adam([log_alpha], 1e-3)
    tgt_ent = -1.0

    N = 100000
    buf = {k: np.zeros((N, d), dtype=np.float32)
           for k, d in [("o", 3), ("a", 1), ("r", 1), ("o2", 3), ("d", 1)]}
    ptr, full = 0, False
    env = Pendulum(cf.seed)
    o = env.reset()
    curve = []
    for step in range(cf.steps):
        if step < 1000:
            a = rng.uniform(-1, 1, size=1)
        else:
            with torch.no_grad():
                a, _ = actor(torch.tensor(o, device=dev)[None])
            a = a.cpu().numpy()[0]
        o2, r, done = env.step(a[0] * 2.0)
        if cf.tnoise > 0:
            r = r + rng.standard_t(2) * cf.tnoise
        for k, v in [("o", o), ("a", a), ("r", [r]), ("o2", o2),
                     ("d", [float(done)])]:
            buf[k][ptr] = v
        ptr = (ptr + 1) % N
        full = full or ptr == 0
        o = env.reset() if done else o2

        if step >= 1000:
            idx = rng.randint(0, N if full else ptr, size=256)
            B = {k: torch.tensor(v[idx], device=dev)
                 for k, v in buf.items()}
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
            la = (log_alpha.exp().detach() * logp - qpi).mean()
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

        if (step + 1) % 2000 == 0:
            ret = evaluate(actor, dev, cf.seed)
            curve.append(ret)
            print(f"QSAC {cf.arch} {cf.loss} s{cf.seed} step {step + 1} "
                  f"ret {ret:.1f}", flush=True)

    rec = {"arch": cf.arch, "loss": cf.loss, "seed": cf.seed,
           "tnoise": cf.tnoise, "final": np.mean(curve[-3:]),
           "best": max(curve), "curve": curve}
    with open(cf.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"QSAC DONE {cf.arch} {cf.loss} s{cf.seed} tn{cf.tnoise} "
          f"final {rec['final']:.1f}", flush=True)


if __name__ == "__main__":
    main()
