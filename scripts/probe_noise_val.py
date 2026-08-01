"""Validation loss under per-feature observation noise, vs held-out GT.

Held-out set: demos >= 2000 of the 20k MP file (unseen by every arm).
For each arm (own normalizer, raw-unit comparison) we report
  err(sigma) = mean |a_pred - a_gt| over the executed 8-step chunk
under three noise models on the 106-dim window:
  uniform : i.i.d. N(0, sigma^2) on EVERY feature   (first-order damage is
            sigma^2||J||_F^2 — blind to PR)
  sparse  : N(0, sigma^2) on K randomly chosen features only  (damage is
            carried by the energy of those features — this is where a
            broad feature reliance should pay)
  worst1  : the single feature (of R sampled) whose corruption hurts most
  (all errors in ENV action units, 7-dim: 3 pos + 3 axis-angle + grip)
            (worst-case single-feature dependence)
Prints NVAL lines. Env: NV_ARMS, NV_N, NV_DRAW, NV_K, NV_R.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

HELD = os.environ.get("NV_HELD", "data/tool_hang_full2ins_mp_20k.hdf5")
NST = int(os.environ.get("NV_N", "192"))
NDRAW = int(os.environ.get("NV_DRAW", "8"))
KSP = int(os.environ.get("NV_K", "5"))
RW = int(os.environ.get("NV_R", "12"))
SIGS = [0.0, 0.05, 0.1, 0.2, 0.4]
SIG_SP = float(os.environ.get("NV_SIGSP", "1.0"))
ARMS = [tuple(a.split(":")) for a in os.environ["NV_ARMS"].split(",")]
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

# ---- held-out states + GT chunks (raw units) -------------------------------
h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
held = names[2000:2400]
rng = np.random.RandomState(0)
Wl, Yl = [], []
for dn in held:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], axis=1).astype(np.float32)
    Aa = np.asarray(h[f"data/{dn}/actions"]).astype(np.float32)
    T = len(S)
    if T < 12:
        continue
    for i in rng.choice(np.arange(1, T - 9), 3, replace=False):
        Wl.append(np.stack([S[i - 1], S[i]]))
        Yl.append(Aa[i:i + 8])
h.close()
Wl, Yl = np.stack(Wl), np.stack(Yl)
sel = rng.choice(len(Wl), min(NST, len(Wl)), replace=False)
Wl, Yl = Wl[sel], Yl[sel]
print(f"held-out states {len(Wl)} from {len(held)} unseen demos", flush=True)


def load(loss, dset, ck):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    H = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    cfg.task.horizon = H
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    return cfg, ds, ag, get_sampler(loss), H


for name, loss, ck, dset in ARMS:
    cfg, ds, ag, sampler, H = load(loss, dset, ck)
    no = ds.normalizer["obs"]["state"]
    na = ds.normalizer["action"]
    dev = cfg.optimization.device
    Xn = torch.tensor(np.stack([no.normalize(w).reshape(-1) for w in Wl]),
                      device=dev, dtype=torch.float32)
    Ygt = torch.tensor(Yl, device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev)

    def err_of(Xb, Yb):
        out = []
        for i in range(0, len(Xb), 128):
            xb = Xb[i:i + 128]
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                             a0, {"state": xb.reshape(-1, 2, 53)})
            pe = ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, 1:9])
            pr = torch.tensor(np.asarray(pe), device=dev,
                              dtype=torch.float32)
            out.append((pr - Yb[i:i + 128]).abs().mean(dim=(1, 2)))
        return torch.cat(out)

    base = float(err_of(Xn, Ygt).mean())
    row = [f"sig{s}:{0.0:.4f}" for s in []]
    errs = {}
    for s in SIGS:
        if s == 0.0:
            errs[s] = base
            continue
        acc = []
        for d in range(NDRAW):
            gen.manual_seed(100 + d)
            Xp = Xn + s * torch.randn(Xn.shape, device=dev, generator=gen)
            acc.append(float(err_of(Xp, Ygt).mean()))
        errs[s] = float(np.mean(acc))
    # sparse: K random features per draw
    sp = []
    for d in range(NDRAW):
        gen.manual_seed(200 + d)
        Xp = Xn.clone()
        idx = torch.randint(0, Xn.shape[1], (KSP,), device=dev, generator=gen)
        Xp[:, idx] += SIG_SP * torch.randn((len(Xn), KSP), device=dev,
                                           generator=gen)
        sp.append(float(err_of(Xp, Ygt).mean()))
    # worst-case single feature among R sampled
    per = []
    for d in range(RW):
        gen.manual_seed(300 + d)
        j = int(torch.randint(0, Xn.shape[1], (1,), generator=gen,
                              device=dev))
        Xp = Xn.clone()
        Xp[:, j] += SIG_SP * torch.randn((len(Xn),), device=dev, generator=gen)
        per.append(float(err_of(Xp, Ygt).mean()))
    su = " ".join(f"s{s}:{errs[s]:.4f}" for s in SIGS)
    print(f"NVAL {name} clean {base:.4f} | uniform[{su}] | "
          f"sparse{KSP}@{SIG_SP} {np.mean(sp):.4f} | "
          f"worst1 {max(per):.4f} (mean {np.mean(per):.4f}) | "
          f"deg(0.2) {errs[0.2]-base:.4f} deg_rel {(errs[0.2]-base)/base:.2f}",
          flush=True)
print("NVAL done", flush=True)
