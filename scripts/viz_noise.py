"""Visualize the displaced-mean pseudo-noise on MP-200. 4 panels:
A demo bundle colored by local cross-demo action conflict; B zoom on a
high-conflict twin cluster (neighbor chunks fan out; mean vs truth);
C D(eps) scatter (action disagreement vs obs distance, near-twin pairs);
D plan-phase indices of one twin cluster. Saves pseudonoise_fig.png."""
import os
import sys

sys.path.insert(0, ".")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"
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
PW, PA, PD, PE, PP = [], [], [], [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    for i in range(1, L - 16):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        PA.append(A[i:i + 8, 0:3])
        PD.append(di)
        PE.append(S[i, 44:47])
        PP.append(i / L)
h.close()
PW = np.stack(PW)
PA = np.stack(PA)
PD, PE, PP = np.asarray(PD), np.stack(PE), np.asarray(PP)
tP = torch.tensor(PW)

rng = np.random.default_rng(0)
qidx = rng.choice(np.where(PD < 8)[0], 500, replace=False)
CONF, NBRS = [], {}
for qi in qidx:
    d = torch.norm(tP - torch.tensor(PW[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    nb = np.where(d < 0.5)[0]
    if len(nb) < 4:
        CONF.append(0.0)
        continue
    T = PA[nb].reshape(len(nb), -1)
    CONF.append(float(np.sqrt(((T - T.mean(0)) ** 2).mean())))
    NBRS[qi] = nb
CONF = np.asarray(CONF)

fig = plt.figure(figsize=(20, 11))
axA = fig.add_subplot(2, 2, 1, projection="3d")
m8 = PD < 8
sc = axA.scatter(PE[m8][::3, 0], PE[m8][::3, 1], PE[m8][::3, 2], s=2,
                 c="0.8")
top = qidx[np.argsort(CONF)[-120:]]
sc2 = axA.scatter(PE[top, 0], PE[top, 1], PE[top, 2],
                  c=CONF[np.argsort(CONF)[-120:]], cmap="inferno", s=28)
plt.colorbar(sc2, ax=axA, shrink=0.6, label="cross-demo action conflict")
axA.set_title("A. Where the pseudo-noise lives (top-conflict states)")
axA.view_init(elev=22, azim=-60)

best = top[-1]
nb = NBRS[best]
axB = fig.add_subplot(2, 2, 2, projection="3d")
q0 = PE[best]
for j in nb[:14]:
    ch = PE[j] + np.cumsum(PA[j], 0) * 0.05
    ln = np.vstack([PE[j][None], ch])
    axB.plot(ln[:, 0], ln[:, 1], ln[:, 2], "-", color="C0", alpha=0.6,
             lw=1.2)
    axB.scatter(*PE[j], color="C0", s=18)
mu = PA[nb].mean(0)
chm = q0 + np.cumsum(mu, 0) * 0.05
lnm = np.vstack([q0[None], chm])
axB.plot(lnm[:, 0], lnm[:, 1], lnm[:, 2], "--", color="k", lw=3,
         label="neighbor-MEAN chunk")
cht = q0 + np.cumsum(PA[best], 0) * 0.05
lnt = np.vstack([q0[None], cht])
axB.plot(lnt[:, 0], lnt[:, 1], lnt[:, 2], "-", color="C3", lw=3,
         label="TRUE chunk (displaced)")
axB.scatter(*q0, color="C3", s=60, marker="*")
axB.legend()
axB.set_title("B. One twin cluster: states overlap, chunks fan out")
axB.view_init(elev=22, azim=-60)

axC = fig.add_subplot(2, 2, 3)
xs, ys = [], []
for qi in qidx[:300]:
    d = torch.norm(tP - torch.tensor(PW[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    nn = np.argsort(d)[:20]
    for j in nn:
        if d[j] < 2.0:
            xs.append(d[j])
            ys.append(np.abs(PA[qi] - PA[j]).mean())
axC.scatter(xs, ys, s=3, alpha=0.25)
axC.set_xlabel("obs distance (normalized)")
axC.set_ylabel("action-chunk disagreement (raw)")
axC.set_title("C. D(eps): intercept ~0 (deterministic), broad spread at\n"
              "small eps = resolution-limited conflict")
axD = fig.add_subplot(2, 2, 4)
phs = PP[nb]
dms = PD[nb]
axD.scatter(dms, phs, c="C0", s=40)
axD.scatter([PD[best]], [PP[best]], c="C3", s=120, marker="*")
axD.set_xlabel("demo index")
axD.set_ylabel("progress (plan phase)")
axD.set_title("D. The twins' plan phases: same obs window, different\n"
              "episodes/phases -> different labels")
plt.tight_layout()
plt.savefig("analysis/paper/pseudonoise_fig.png", dpi=130)
print("VN saved", flush=True)
print(f"VN cluster: {len(nb)} twins, conf {CONF[np.where(qidx==best)[0][0]]:.4f}, "
      f"obs-dist range within cluster small", flush=True)
