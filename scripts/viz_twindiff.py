"""Quantify the twin pair demo4@127 vs demo95@127: state gap, action-chunk
difference, integrated divergence, and mm-scale overlay figure."""
import os
import sys

sys.path.insert(0, ".")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
dA, dB, i0 = "demo_4", "demo_95", 127


def get(dn):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    return S, A


SA, AA = get(dA)
SB, AB = get(dB)
h.close()
sa, sb = SA[i0], SB[i0]
print(f"TD sync obs-dist (normalized, 2-frame) "
      f"{np.linalg.norm(no.normalize(np.stack([SA[i0-1], sa])) - no.normalize(np.stack([SB[i0-1], sb]))):.3f}",
      flush=True)
print(f"TD eef gap at sync {1000*np.linalg.norm(sa[44:47]-sb[44:47]):.1f}mm | "
      f"object gap {np.linalg.norm(sa[:44]-sb[:44]):.4f} (raw)", flush=True)
ca, cb = AA[i0:i0+8, 0:3], AB[i0:i0+8, 0:3]
print(f"TD chunk pos-delta diff per-step p50 "
      f"{1000*0.05*np.median(np.linalg.norm(ca-cb,axis=1)):.1f}mm-equiv | "
      f"integrated 8-step displacement diff "
      f"{1000*0.05*np.linalg.norm((ca-cb).sum(0)):.1f}mm", flush=True)
ra, rb = AA[i0:i0+8, 3:6], AB[i0:i0+8, 3:6]
print(f"TD chunk rot diff per-step p50 {np.median(np.linalg.norm(ra-rb,axis=1)):.3f} "
      f"(|rot| a {np.abs(ra).sum():.2f} b {np.abs(rb).sum():.2f})", flush=True)
for k in [10, 20, 45]:
    d = 1000*np.linalg.norm(SA[i0+k, 44:47]-SB[i0+k, 44:47])
    print(f"TD eef divergence after {k} steps: {d:.1f}mm", flush=True)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
ax = fig.add_subplot(1, 2, 1, projection="3d")
axs[0].axis("off")
pa = SA[i0-20:i0+45, 44:47]*1000
pb = SB[i0-20:i0+45, 44:47]*1000
ax.plot(pa[:, 0], pa[:, 1], pa[:, 2], "-", color="C0", lw=2, label=dA)
ax.plot(pb[:, 0], pb[:, 1], pb[:, 2], "-", color="C3", lw=2, label=dB)
ax.scatter(*pa[20], color="k", s=70, marker="*")
ax.set_title("eef paths +-(20,45) steps around sync (mm)")
ax.legend()
ax.view_init(elev=22, azim=-60)
axs[1].plot(range(-20, 45), 1000*np.linalg.norm(
    SA[i0-20:i0+45, 44:47]-SB[i0-20:i0+45, 44:47], axis=1), "-k")
axs[1].axvline(0, color="r", ls="--")
axs[1].set_xlabel("steps from sync")
axs[1].set_ylabel("eef separation (mm)")
axs[1].set_title("pair separation over time")
plt.tight_layout()
plt.savefig("analysis/paper/twin_diff.png", dpi=130)
print("TD saved", flush=True)
