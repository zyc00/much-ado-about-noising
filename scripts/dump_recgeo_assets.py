"""Dump assets for RECOVERY-GEOMETRY sufficiency loss (regression_recgeo).
For puredart pairs in bands [0.5,2) and [2,4) vs the 40-demo anchor cloud:
  w0, ws : normalized obs windows (anchor, perturbed)
  tgt_gt   : native-normalized (10d) delta of GT recovery action at executed slot
  tgt_mip  : MIP-s1 model's response delta (native normalized, executed slot)
  tgt_s10k : MSE-early-s10k model's response delta
  band     : 0 = near [0.5,2), 1 = recoverable [2,4)
Saves logs/recgeo_assets.npz (train pool) and holds out every 5th pair -> *_held arrays.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import h5py
import torch
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import robosuite.utils.transform_utils as T
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
H = 16

cfgdir = os.path.abspath("examples/configs")
with initialize_config_dir(version_base=None, config_dir=cfgdir):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"),
        "network=chiunet", "optimization.loss_type=regression",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
dev = cfg.optimization.device
START = cfg.task.obs_steps - 1


def to10(a7):
    """raw 7d (pos, axisangle, grip) -> native 10d (pos, rot6d, grip)."""
    R = T.quat2mat(T.axisangle2quat(a7[3:6]))
    return np.concatenate([a7[:3], R[:, 0], R[:, 1], [a7[6]]])


def read(path, nmax=None):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
    n = min(nmax, len(h[g])) if nmax else len(h[g])
    out = []
    for i in range(n):
        o = h[f"{g}/demo_{i}/obs"]
        ov = np.concatenate([np.asarray(o[k]) for k in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/demo_{i}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out


def make_responder(ckpt):
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    def resp(win):   # normalized window (2,53) -> native normalized 10d at executed slot
        x = torch.tensor(win[None], device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                an = ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                                  torch.zeros((1, H, 10), device=dev), emb)
        return an[0, START].cpu().numpy()
    return resp


resp_mip = make_responder("logs/full_mip_2000_s1/models/model_latest.pt")
resp_s10 = make_responder("logs/mse_early_2k/models/model_s10000.pt")

clean = read("data/tool_hang_full2ins_2000.hdf5", 40)
cl = np.concatenate([ov for ov, _ in clean], 0)
owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
mu, sig = cl.mean(0), cl.std(0) + 1e-6
tree = cKDTree((cl - mu) / sig)
test = read("data/tool_hang_puredart_full2ins_2000.hdf5", 400)

W0, WS, TG, TM, TS, BAND = [], [], [], [], [], []
c0_mip, c0_s10, c0_gt = {}, {}, {}
for ov, acts in test:
    for t in range(1, len(acts) - 1, 5):
        z = (ov[t] - mu) / sig
        d, idx = tree.query(z)
        if 0.5 <= d < 2.0:
            b = 0
        elif 2.0 <= d < 4.0:
            b = 1
        else:
            continue
        i0, t0 = map(int, owner[int(idx)])
        a0 = clean[i0][1][min(t0, len(clean[i0][1]) - 1)]
        w0 = no.normalize(np.stack([clean[i0][0][max(t0 - 1, 0)], clean[i0][0][t0]]))
        ws = no.normalize(np.stack([ov[t - 1], ov[t]]))
        key = int(idx)
        if key not in c0_mip:
            c0_mip[key] = resp_mip(w0); c0_s10[key] = resp_s10(w0)
            c0_gt[key] = na.normalize(to10(a0)[None])[0]
        tg = na.normalize(to10(acts[t])[None])[0] - c0_gt[key]
        tm = resp_mip(ws) - c0_mip[key]
        tsr = resp_s10(ws) - c0_s10[key]
        W0.append(w0); WS.append(ws); TG.append(tg); TM.append(tm); TS.append(tsr); BAND.append(b)

W0 = np.stack(W0).astype(np.float32); WS = np.stack(WS).astype(np.float32)
TG = np.stack(TG).astype(np.float32); TM = np.stack(TM).astype(np.float32)
TS = np.stack(TS).astype(np.float32); BAND = np.array(BAND, dtype=np.int32)
held = np.arange(len(W0)) % 5 == 0
np.savez("logs/recgeo_assets.npz",
         w0=W0[~held], ws=WS[~held], tgt_gt=TG[~held], tgt_mip=TM[~held],
         tgt_s10k=TS[~held], band=BAND[~held],
         w0_held=W0[held], ws_held=WS[held], tgt_gt_held=TG[held],
         tgt_mip_held=TM[held], tgt_s10k_held=TS[held], band_held=BAND[held])
print(f"saved logs/recgeo_assets.npz train={int((~held).sum())} held={int(held.sum())} "
      f"nearband={int((BAND==0).sum())} recband={int((BAND==1).sum())}")
