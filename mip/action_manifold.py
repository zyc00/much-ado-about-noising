"""Frozen unconditional action-chunk manifold (denoiser) used as a differentiable
projection head. Train a policy by passing its raw output through this FROZEN denoiser
and computing MSE to the target action -- gradients flow through the frozen manifold,
so the policy co-adapts to land in the right pre-image. See loss_type=regression_manifold.

The denoiser is the same architecture trained by scripts/manifold_train_denoiser.py.
Path + sigma are read from env vars MANIFOLD_CKPT / MANIFOLD_SIGMA so train and eval
share one frozen manifold without touching configs."""
import os
import numpy as np
import torch
import torch.nn as nn

_CACHE = {}


class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.n = nn.LayerNorm(h)

    def forward(self, x):
        return x + self.net(self.n(x))


class Denoiser(nn.Module):
    def __init__(self, dim, h=1024, nblocks=6, sdim=128):
        super().__init__()
        self.register_buffer("ff", torch.randn(sdim // 2) * 2.0)
        self.smlp = nn.Sequential(nn.Linear(sdim, h), nn.SiLU(), nn.Linear(h, h))
        self.inp = nn.Linear(dim, h)
        self.blocks = nn.ModuleList([ResBlock(h) for _ in range(nblocks)])
        self.out = nn.Linear(h, dim)

    def forward(self, x, sigma):
        ls = torch.log(sigma).view(-1, 1) * self.ff.view(1, -1)
        se = torch.cat([torch.sin(2 * np.pi * ls), torch.cos(2 * np.pi * ls)], -1)
        s = self.smlp(se)
        z = self.inp(x) + s
        for b in self.blocks:
            z = b(z)
        return self.out(z)


def get_frozen_denoiser(path, device):
    """Load (and cache) the frozen denoiser. requires_grad=False on its params, but it
    stays differentiable so gradients pass THROUGH it to the policy output."""
    key = (os.path.abspath(path), str(device))
    if key not in _CACHE:
        ck = torch.load(path, map_location=device, weights_only=False)
        D = Denoiser(ck["dim"]).to(device)
        D.load_state_dict(ck["state_dict"])
        D.eval()
        for p in D.parameters():
            p.requires_grad_(False)
        _CACHE[key] = (D, ck)
    return _CACHE[key]


def project(act_pred, device):
    """act_pred (B,H,A) normalized -> pass through frozen manifold at MANIFOLD_SIGMA."""
    path = os.environ["MANIFOLD_CKPT"]
    sigma = float(os.environ.get("MANIFOLD_SIGMA", "0.2"))
    D, ck = get_frozen_denoiser(path, device)
    B = act_pred.shape[0]
    flat = act_pred.reshape(B, -1)
    out = D(flat, torch.full((B, 1), sigma, device=device))
    return out.reshape(act_pred.shape)
