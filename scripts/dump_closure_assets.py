"""Dump assets for CLOSURE B3/B4 losses:
 - PAIRS: normalized obs windows (w0 anchor, ws perturbed) for puredart [2,4) pairs
 - BADDIRS: normalized-obs-space perturbation directions for bad1 (pos) and v_mseamp,
   plus the 3d output direction bad1 (action-pos space)
Saves logs/closure_assets.npz
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import h5py
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)

cfgdir = os.path.abspath("examples/configs")
with initialize_config_dir(version_base=None, config_dir=cfgdir):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"),
        "network=chiunet", "optimization.loss_type=regression",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]


def read(path, nmax=None):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
    n = min(nmax, len(h[g])) if nmax else len(h[g])
    out = []
    for i in range(n):
        o = h[f"{g}/demo_{i}/obs"]
        out.append(np.concatenate([np.asarray(o[k]) for k in OK], axis=1).astype(np.float32))
    h.close(); return out


clean = read("data/tool_hang_full2ins_2000.hdf5", 40)
cl = np.concatenate(clean, 0)
owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, ov in enumerate(clean)], 0)
mu, sig = cl.mean(0), cl.std(0) + 1e-6
tree = cKDTree((cl - mu) / sig)
test = read("data/tool_hang_puredart_full2ins_2000.hdf5", 400)
W0, WS = [], []
for ov in test:
    for t in range(1, len(ov) - 1, 6):
        z = (ov[t] - mu) / sig
        d, idx = tree.query(z)
        if not (2.0 <= d < 4.0): continue
        i0, t0 = map(int, owner[int(idx)])
        W0.append(no.normalize(np.stack([clean[i0][max(t0 - 1, 0)], clean[i0][t0]])))
        WS.append(no.normalize(np.stack([ov[t - 1], ov[t]])))
W0 = np.stack(W0).astype(np.float32); WS = np.stack(WS).astype(np.float32)

ref = np.load("analysis/recovery/badmode_ref.npz")
Bx = ref["G_m"][0:3, POS] * ref["sig"][POS][None, :]
wm, Vm = np.linalg.eigh((Bx + Bx.T) / 2)
bad1 = Vm[:, np.argmax(wm)]
vrot = np.linalg.svd(ref["G_m"][3:6, :] * ref["sig"][None, :])[2][0]
# raw-space unit perturbations -> normalized-space via exact linear map
x0 = cl.mean(0)
d_pos_raw = np.zeros(53); d_pos_raw[POS] = bad1 * sig[POS]
d_rot_raw = vrot * sig
w0 = np.stack([x0, x0])
dpos_dir = (no.normalize(np.stack([x0 + d_pos_raw] * 2)) - no.normalize(w0))[0]
drot_dir = (no.normalize(np.stack([x0 + d_rot_raw] * 2)) - no.normalize(w0))[0]
np.savez("logs/closure_assets.npz", w0=W0, ws=WS,
         dpos_dir=dpos_dir.astype(np.float32), drot_dir=drot_dir.astype(np.float32),
         bad1=bad1.astype(np.float32))
print(f"saved logs/closure_assets.npz pairs={len(W0)} "
      f"|dpos_dir|={np.linalg.norm(dpos_dir):.3f} |drot_dir|={np.linalg.norm(drot_dir):.3f}")
