"""Single-cell driver for the toy-MDP experiments. Appends one JSON line.

Experiments:
  sup_clean   supervised fit of tabular Q* (pure spectral bias, no noise)
  sup_gnoise  supervised fit of Q* + FIXED Gaussian label noise (scale 0.25)
  sup_tnoise  supervised fit of Q* + FIXED Student-t df=2 noise (scale 0.25)
  nfq         neural fitted VI, bootstrapped targets, target sync each epoch
  nfq_tnoise  nfq + per-epoch Student-t df=2 reward noise (scale 0.25)

Usage: python run_toy.py <exp> <arch> <loss> <seed> [out.jsonl]
  arch: mlp | lff1 | lff5 | ...   loss: mse | huber | ht | hg | mip
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from qlib import make_mdp, perform_vi, train_nfq, train_supervised

DEV = os.environ.get("QDEV", "cuda" if torch.cuda.is_available() else "cpu")

NOISE_SCALE = 0.25

exp, arch, loss, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out = sys.argv[5] if len(sys.argv) > 5 else "results.jsonl"

states, rewards, dyn_mats = make_mdp(num_states=100, option="fixed")
q_star = perform_vi(rewards, dyn_mats)

if exp.startswith("sup"):
    rng = np.random.RandomState(1000 + seed)
    y = q_star.copy()
    if exp == "sup_gnoise":
        y = y + rng.randn(*y.shape) * NOISE_SCALE
    elif exp == "sup_tnoise":
        y = y + rng.standard_t(2, size=y.shape) * NOISE_SCALE
    curve = train_supervised(arch, loss, states, y, q_star, seed, device=DEV)
else:
    noise = ("t", 2, NOISE_SCALE) if exp == "nfq_tnoise" else None
    curve = train_nfq(arch, loss, states, rewards, dyn_mats, q_star, seed,
                      reward_noise=noise, device=DEV)

rec = {"exp": exp, "arch": arch, "loss": loss, "seed": seed,
       "final_rmse": curve[-1], "best_rmse": min(curve),
       "rmse_at": {str(i * 100): c for i, c in enumerate(curve)
                   if i % 5 == 0}}
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"QTOY {exp} {arch} {loss} s{seed} final {curve[-1]:.4f} "
      f"best {min(curve):.4f}", flush=True)
