"""Train an UNCONDITIONAL action-chunk denoiser (the 'manifold' / projection operator).
Trained on 20k demos' actions for diversity, in the 2k-normalizer 10-dim rot6d space
(so it matches MSE-clean-2k's output space). Multi-noise-level x-prediction:
  x_noisy = x + sigma*eps,  sigma ~ logU[smin,smax],  predict x.
This learns 'what valid H-step action chunks look like'; at eval we project the MSE-2k
prediction onto it (annealed denoising). Tests: unconditional-manifold(20k) + MSE(2k)
>=? MIP-2k -- i.e. is MIP's edge a separable manifold-projection operator."""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset


def load_ds(path):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(path)}",
            "network=chiunet", "optimization.loss_type=regression",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    return make_dataset(cfg.task)


def build_chunks(ds, na2, H, stride):
    """Normalized (2k-na) 10-dim action chunks of length H, within episodes."""
    A = ds.replay_buffer["action"][:]            # (T,10) unnormalized converted
    An = na2.normalize(A).astype(np.float32)      # 2k-na space
    ends = np.asarray(ds.replay_buffer.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    out = []
    for s, e in zip(starts, ends):
        seg = An[s:e]
        for i in range(0, len(seg) - H + 1, stride):
            out.append(seg[i:i + H])
    return np.asarray(out, dtype=np.float32)      # (N,H,10)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds20", default="data/tool_hang_full2ins_20000.hdf5")
    ap.add_argument("--ds2", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--H", type=int, default=16); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--smin", type=float, default=0.02); ap.add_argument("--smax", type=float, default=1.2)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--out", default="analysis/manifold/denoiser_uncond20k.pt")
    args = ap.parse_args()
    dev = "cuda"
    ds2 = load_ds(args.ds2); na2 = ds2.normalizer["action"]
    ds20 = load_ds(args.ds20)
    X = build_chunks(ds20, na2, args.H, args.stride)
    print(f"manifold chunks from 20k actions (2k-na space): {X.shape}  range[{X.min():.2f},{X.max():.2f}]")
    Xt = torch.tensor(X.reshape(len(X), -1), device=dev)
    dim = Xt.shape[1]; N = len(Xt)
    D = Denoiser(dim).to(dev)
    opt = torch.optim.AdamW(D.parameters(), lr=2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ls_min, ls_max = np.log(args.smin), np.log(args.smax)
    g = torch.Generator(device=dev).manual_seed(0)
    D.train()
    for it in range(args.steps):
        idx = torch.randint(0, N, (args.bs,), device=dev, generator=g)
        x = Xt[idx]
        sigma = torch.exp(torch.rand(args.bs, 1, device=dev, generator=g) * (ls_max - ls_min) + ls_min)
        eps = torch.randn(x.shape, device=dev, generator=g)
        xn = x + sigma * eps
        xhat = D(xn, sigma)
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 4000 == 0 or it == args.steps - 1:
            # report denoise quality at a few sigmas
            with torch.no_grad():
                msg = []
                for sg in [0.05, 0.2, 0.5, 1.0]:
                    xb = Xt[torch.randint(0, N, (2048,), device=dev, generator=g)]
                    e = torch.randn(xb.shape, device=dev, generator=g)
                    xr = D(xb + sg * e, torch.full((2048, 1), sg, device=dev))
                    base = (sg * e).pow(2).mean().sqrt()
                    res = (xr - xb).pow(2).mean().sqrt()
                    msg.append(f"s{sg}:{res/base:.2f}")
                print(f"  it {it:6d} loss {loss.item():.4f}  denoise-residual/noise {' '.join(msg)}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": D.state_dict(), "dim": dim, "H": args.H,
                "smin": args.smin, "smax": args.smax}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
