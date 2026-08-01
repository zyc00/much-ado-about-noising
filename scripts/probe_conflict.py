"""Conflict-style channel-dominance test (FDR-analog at the observation
level, after arXiv:2607.21582's behavioral dominance metric).

Construct hybrid observations whose OBJECT channel comes from state A and
whose PROPRIOCEPTION channel (eef pos/quat + gripper) comes from state B —
two coherent on-support states with substantially different policy outputs.
Feed the hybrid to each policy and ask whose action it produces:
  object-win  : output closer to the policy's own a(A)
  proprio-win : closer to its own a(B)
Dominance rate = fraction object-win over informative pairs. Also a graded
score: relative proximity lambda = |f(H)-a(B)| / (|f(H)-a(A)|+|f(H)-a(B)|)
(1 = fully object-driven). Prints CONF lines. Env: CF_ARMS, CF_N.
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

ARMS = [tuple(a.split(":")) for a in os.environ["CF_ARMS"].split(",")]
NPAIR = int(os.environ.get("CF_N", "400"))
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2300]
rng = np.random.RandomState(0)
Wl = []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    if len(S) < 20:
        continue
    for i in rng.choice(np.arange(1, len(S) - 9), 3, replace=False):
        Wl.append(np.stack([S[i - 1], S[i]]))
h.close()
Wl = np.stack(Wl)
ia = rng.choice(len(Wl), NPAIR)
ib = rng.choice(len(Wl), NPAIR)
A, B = Wl[ia], Wl[ib]
# hybrid: object dims (0:44) from A, proprio dims (44:53) from B, both frames
Hh = A.copy()
Hh[:, :, 44:53] = B[:, :, 44:53]
print(f"pairs {NPAIR}", flush=True)

for name, loss, ck, dset in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false",
            "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    H = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device

    def act(WS):
        out = []
        for i in range(0, len(WS), 256):
            xb = torch.tensor(np.stack([no.normalize(w) for w in WS[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            out.append(np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, 1:9])))
        return np.concatenate(out)

    pA, pB, pH = act(A), act(B), act(Hh)
    dAB = np.abs(pA - pB).mean(axis=(1, 2))
    keep = dAB > np.median(dAB)              # informative pairs only
    dHA = np.abs(pH - pA).mean(axis=(1, 2))[keep]
    dHB = np.abs(pH - pB).mean(axis=(1, 2))[keep]
    lam = dHB / (dHA + dHB + 1e-12)          # 1 = object-driven
    win = float((dHA < dHB).mean())
    print(f"CONF {name} n={int(keep.sum())} object_win {win:.3f} "
          f"lambda_mean {lam.mean():.3f} (p10 {np.percentile(lam,10):.3f} "
          f"p90 {np.percentile(lam,90):.3f})", flush=True)
print("CONF done", flush=True)
