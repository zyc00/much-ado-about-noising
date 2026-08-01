"""Trajectory-level failure forensics on roll_dump2 outputs. For each episode:
per-step tube distance to MP-200 training windows (normalized obs, kNN),
phase progress = nearest neighbor's progress fraction. Reports per arm:
SR, failure taxonomy (timeout-in-tube grind vs escape), divergence onset
phase, max progress reached, dwell stats. Env: RD_DIRS=tag:dir,...
Prints AR lines."""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import glob
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.datasets.robomimic_dataset import make_dataset

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PP = [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    L = len(S)
    for i in range(1, L):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        PP.append(i / L)
h.close()
tP = torch.tensor(np.stack(PW), device="cuda")
PP = np.asarray(PP)
print(f"AR pool {len(PP)}", flush=True)

for spec in os.environ["RD_DIRS"].split(","):
    tag, dr = spec.split(":")
    stats = {"succ": 0, "n": 0, "onset_ph": [], "maxprog": [],
             "grind": 0, "escape": 0, "d_p50s": [], "d_p95s": []}
    for f in sorted(glob.glob(f"{dr}/ep_*.npz")):
        z = np.load(f)
        obs, asm = z["obs"], int(z["asm"])
        stats["n"] += 1
        stats["succ"] += asm
        W = np.stack([np.stack([obs[i - 1], obs[i]]).reshape(2, -1)
                      for i in range(1, len(obs))])
        Wn = torch.tensor(np.stack([no.normalize(w) for w in W]).reshape(
            len(W), -1), device="cuda", dtype=tP.dtype)
        D = torch.cdist(Wn, tP)
        dmin, jmin = D.min(dim=1)
        dmin = dmin.cpu().numpy()
        prog = PP[jmin.cpu().numpy()]
        stats["d_p50s"].append(np.median(dmin))
        stats["d_p95s"].append(np.percentile(dmin, 95))
        if not asm:
            stats["maxprog"].append(prog.max())
            esc = np.where(dmin > 2.0)[0]
            sus = [i for i in esc if dmin[i:i + 20].min() > 2.0]
            if len(sus):
                stats["escape"] += 1
                stats["onset_ph"].append(prog[sus[0]])
            else:
                stats["grind"] += 1
                stats["onset_ph"].append(prog[int(np.argmax(
                    np.arange(len(prog)) * (prog > prog.max() - 0.02)))])
    op = np.asarray(stats["onset_ph"])
    mp = np.asarray(stats["maxprog"])
    print(f"AR {tag} SR {stats['succ']}/{stats['n']} | fails: escape "
          f"{stats['escape']} grind {stats['grind']} | onset/stall phase "
          f"{' '.join(f'{v:.2f}' for v in sorted(op))} | maxprog p50 "
          f"{np.median(mp) if len(mp) else -1:.2f} | tube d p50 "
          f"{np.median(stats['d_p50s']):.2f} p95 "
          f"{np.median(stats['d_p95s']):.2f}", flush=True)
print("AR done", flush=True)
