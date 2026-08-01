"""Diagnose the mlp+ht MCC failure: track Q-scale, sigma, and the
gradient share of goal-bearing TD targets under mse vs ht.
Prints DIAG lines. Local, ~30k steps per arm."""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sac_env import (Actor, Critic, MountainCar, critic_td_loss,  # noqa
                     q_eval)
from copy import deepcopy

dev = "cuda" if torch.cuda.is_available() else "cpu"

for LOSS in ["mse", "ht"]:
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    q1, q2 = Critic("mlp", 2).to(dev), Critic("mlp", 2).to(dev)
    q1t, q2t = deepcopy(q1), deepcopy(q2)
    actor = Actor(2).to(dev)
    log_alpha = torch.zeros(1, device=dev, requires_grad=True)
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 1e-3)
    oa = torch.optim.Adam(actor.parameters(), 1e-3)
    ol = torch.optim.Adam([log_alpha], 1e-3)
    N = 100000
    buf = {k: np.zeros((N, d), dtype=np.float32)
           for k, d in [("o", 2), ("a", 1), ("r", 1), ("o2", 2), ("d", 1)]}
    ptr, full = 0, False
    env = MountainCar(0)
    o = env.reset()
    a_hold = np.zeros(1)
    for step in range(30000):
        if step < 20000:
            if step % 40 == 0:
                a_hold = rng.uniform(-1, 1, size=1)
            a = a_hold
        else:
            with torch.no_grad():
                a, _ = actor(torch.tensor(o, device=dev)[None])
            a = a.cpu().numpy()[0]
        o2, r, done = env.step(a[0])
        for k, v in [("o", o), ("a", a), ("r", [r]), ("o2", o2),
                     ("d", [float(done)])]:
            buf[k][ptr] = v
        ptr = (ptr + 1) % N
        full = full or ptr == 0
        o = env.reset() if done else o2
        if step >= 20000:
            idx = rng.randint(0, N if full else ptr, size=256)
            B = {k: torch.tensor(v[idx], device=dev) for k, v in buf.items()}
            with torch.no_grad():
                a2, logp2 = actor(B["o2"])
                qt = torch.min(q_eval(LOSS, q1t, B["o2"], a2),
                               q_eval(LOSS, q2t, B["o2"], a2))
                y = B["r"] + 0.99 * (1 - B["d"]) * \
                    (qt - log_alpha.exp() * logp2)
            lq = critic_td_loss(LOSS, q1, B["o"], B["a"], y) + \
                critic_td_loss(LOSS, q2, B["o"], B["a"], y)
            oq.zero_grad()
            lq.backward()
            oq.step()
            an, logp = actor(B["o"])
            qpi = torch.min(q_eval(LOSS, q1, B["o"], an),
                            q_eval(LOSS, q2, B["o"], an))
            la = (log_alpha.exp().detach() * logp - qpi).mean()
            oa.zero_grad()
            la.backward()
            oa.step()
            ll = (-log_alpha.exp() * (logp.detach() + (-1.0))).mean()
            ol.zero_grad()
            ll.backward()
            ol.step()
            with torch.no_grad():
                for tn, on in [(q1t, q1), (q2t, q2)]:
                    for pt, p in zip(tn.parameters(), on.parameters()):
                        pt.mul_(0.995).add_(0.005 * p)
        if step >= 20000 and (step + 1) % 2500 == 0:
            n = N if full else ptr
            gi = np.where(buf["r"][:n, 0] > 50)[0]
            ri = rng.choice(n, 2048, replace=False)
            samp = np.concatenate([gi, ri])
            S = {k: torch.tensor(v[samp], device=dev) for k, v in buf.items()}
            with torch.no_grad():
                a2, logp2 = actor(S["o2"])
                qt = torch.min(q_eval(LOSS, q1t, S["o2"], a2),
                               q_eval(LOSS, q2t, S["o2"], a2))
                y = S["r"] + 0.99 * (1 - S["d"]) * \
                    (qt - log_alpha.exp() * logp2)
                q, s_raw = q1(S["o"], S["a"])
                sig = F.softplus(s_raw) + 1e-3
                r_ = (q - y).abs()
                g_mse = 2 * r_
                g_ht = 3 * r_ / (2 * sig ** 2 + r_ ** 2)
                g = g_mse if LOSS == "mse" else g_ht
                ng = len(gi)
                share = (g[:ng].sum() / g.sum()).item() if ng else 0.0
                frac = ng / len(samp)
                print(f"DIAG {LOSS} step {step + 1} n_goal_tr {ng} "
                      f"qmax {q.max().item():.1f} ymax {y.max().item():.1f} "
                      f"sig_p50 {sig.median().item():.2f} "
                      f"sig_goal {sig[:ng].median().item() if ng else 0:.2f} "
                      f"grad_share_goal {share:.3f} (pop_frac {frac:.3f}) "
                      f"resid_goal_p50 {r_[:ng].median().item() if ng else 0:.1f}",
                      flush=True)
print("DIAG done", flush=True)
