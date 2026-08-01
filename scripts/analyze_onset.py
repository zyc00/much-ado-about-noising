"""WHERE does the tail divergence happen (behavior-area attribution)?
From twofactor DUMPTRAJ dumps: (a) expansive cycles (dq in [1,4), dd>+1),
(b) episode onset = first step with d>=2. Attribution via RAW normalized
STATE-space NN against the demo bank (not the model's folded embedding),
giving r-hat = time-relative-to-closure; plus physical features (gripper
aperture, eef z). Bands: appr r<-12 | settle [-12,2) | shoulder [2,10) |
stroke [10,50) | insert/post >=50. Envs: LOSS (for cfg), QD, TAG, AS."""
import glob
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
import h5py
import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.datasets.robomimic_dataset import make_dataset

AS = int(os.environ.get("AS", "8"))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=data/tool_hang_full2ins_2000.hdf5", "network=chiunet",
        f"optimization.loss_type={os.environ.get('LOSS','regression')}", "optimization.auto_resume=false",
        "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:150]
W, R = [], []
for k in keys:
    o = h[f"data/{k}/obs"]
    ov_ = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1)
    g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t - 1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        W.append(np.stack([ov_[t - 1], ov_[t]])); R.append(t - c1)
h.close()
W = np.stack(W); R = np.array(R)
VB = torch.tensor(no.normalize(W).reshape(len(W), -1))

BANDS = [("appr", -1e9, -12), ("settle", -12, 2), ("shoulder", 2, 10), ("stroke", 10, 50), ("ins/post", 50, 1e9)]
def attrib(states, label, TAG):
    if not states:
        print(f"ONSET {TAG} {label}: none", flush=True); return
    Q = np.stack(states)
    Vq = torch.tensor(no.normalize(Q).reshape(len(Q), -1))
    D = torch.cdist(Vq, VB)
    nn = D.topk(5, largest=False).indices.numpy()
    rhat = np.median(R[nn], axis=1)
    grip = Q[:, -1, -2:].sum(axis=1)      # raw gripper qpos aperture
    z = Q[:, -1, 44 + 2]                   # eef z (object block is 44 dims)
    parts = " ".join(f"{nm}:{100*np.mean([(lo <= r_ < hi) for r_ in rhat]):.0f}%" for nm, lo, hi in BANDS)
    print(f"ONSET {TAG} {label} n={len(Q)} | rhat bands {parts} | rhat p10/50/90 = "
          f"{np.percentile(rhat,10):.0f}/{np.median(rhat):.0f}/{np.percentile(rhat,90):.0f} | "
          f"grip p50={np.median(grip):.3f} eefz p50={np.median(z):.3f}", flush=True)

TAG = os.environ.get("TAG", "ckpt")
expc, onset, onset_fail = [], [], []
for f in glob.glob(os.environ["QD"] + "/ep_*.npz"):
    z = np.load(f)
    dser = np.asarray(z["dser"], dtype=np.float64); obsw = z["obsw"]; asm = int(z["asm"])
    T = min(len(dser), len(obsw))
    for q in range(0, T - AS, AS):
        dq = dser[q]; dd = dser[q + AS] - dq
        if 1 <= dq < 4 and dd > 1:
            expc.append(obsw[q])
    on = np.where(dser[:T] >= 2)[0]
    if len(on):
        onset.append(obsw[on[0]])
        if not asm:
            onset_fail.append(obsw[on[0]])
attrib(expc, "expansive-cycles", TAG)
attrib(onset, "first-crossing d>=2 (all eps)", TAG)
attrib(onset_fail, "first-crossing d>=2 (FAILED eps)", TAG)
print("ONSET-DONE")
