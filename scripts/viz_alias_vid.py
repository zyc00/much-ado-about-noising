"""Vivid visualization of cross-trajectory aliasing at phase regions:
3D demo bundle + progress cursor + AMPLIFIED conflicting action fans at
the current window's aliasing hotspot; camera orbits. Saves mp4 + a static
annotated figure."""
import os
import sys

sys.path.insert(0, ".")
import h5py
import matplotlib
matplotlib.use("Agg")
import imageio
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
NSHOW = 30
E, P, W, A, D = [], [], [], [], []
for di, dn in enumerate(names[:60]):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    Ad = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    for i in range(1, L - 9, 2):
        E.append(S[i, 44:47])
        P.append(i / L)
        W.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        A.append(Ad[i:i + 8, 0:3].mean(0))
        D.append(di)
h.close()
E, P, A, D = np.stack(E), np.asarray(P), np.stack(A), np.asarray(D)
Wn = np.stack(W)
tW = torch.tensor(Wn)
CONF = np.zeros(len(E))
DIRS = {}
sub = np.arange(0, len(E), 3)
for k in sub:
    d = torch.norm(tW - tW[k], dim=1).numpy()
    d[D == D[k]] = 1e9
    nb = np.where(d < 0.5)[0]
    if len(nb) >= 4:
        T = A[nb]
        CONF[k] = np.sqrt(((T - T.mean(0)) ** 2).mean())
        DIRS[k] = (nb[:10], A[nb[:10]])
print(f"VA states {len(E)} conflicts computed {len(sub)}", flush=True)

PHASES = [(0.0, 0.25, "REACH"), (0.25, 0.35, "PICK"), (0.35, 0.6, "LIFT/TRANSIT"),
          (0.6, 0.75, "WRIST ROTATE"), (0.75, 0.9, "ALIGN/INSERT"),
          (0.9, 1.01, "SETTLE")]
mask30 = D < NSHOW
FR = 220
writer = imageio.get_writer("analysis/paper/alias_vid.mp4", fps=10)
for f in range(FR):
    prog = (f / FR)
    az = -60 + 360 * f / FR
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for di in range(NSHOW):
        m = (D == di) & mask30
        ax.plot(E[m, 0], E[m, 1], E[m, 2], color="0.82", lw=0.5, alpha=0.5)
    lo, hi = prog, prog + 0.06
    wm = (P >= lo) & (P < hi) & (CONF > 0)
    if wm.any():
        sc = ax.scatter(E[wm, 0], E[wm, 1], E[wm, 2], c=CONF[wm],
                        cmap="inferno", s=22, vmin=0, vmax=0.25)
        k = np.where(wm)[0][int(np.argmax(CONF[wm]))]
        if k in DIRS and CONF[k] > 0.03:
            nb, TT = DIRS[k]
            for t in TT:
                v = t / (np.linalg.norm(t) + 1e-9) * 0.11
                ax.plot([E[k, 0], E[k, 0] + v[0]], [E[k, 1], E[k, 1] + v[1]],
                        [E[k, 2], E[k, 2] + v[2]], color="C3", lw=1.6,
                        alpha=0.8)
            ax.scatter(*E[k], color="C3", s=90, marker="*")
    ph = next(n for a, b, n in PHASES if a <= prog < b)
    mc = CONF[(P >= lo) & (P < hi) & (CONF > 0)]
    ax.set_title(f"phase: {ph}   progress {prog:.2f}   "
                 f"local cross-demo conflict "
                 f"{(mc.mean() if len(mc) else 0):.3f}\n"
                 "red fan = 10 other demos' (amplified) commanded directions "
                 "at ONE aliased state")
    ax.view_init(elev=22, azim=az)
    ax.set_xlim(-1.25, 0.45)
    ax.set_ylim(-0.8, 1.0)
    ax.set_zlim(0.75, 1.35)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    writer.append_data(img)
    plt.close(fig)
writer.close()
print("VA video saved", flush=True)

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection="3d")
for di in range(NSHOW):
    m = (D == di) & mask30
    ax.plot(E[m, 0], E[m, 1], E[m, 2], color="0.85", lw=0.5, alpha=0.5)
cm = CONF > 0
ax.scatter(E[cm, 0], E[cm, 1], E[cm, 2], c=CONF[cm], cmap="inferno", s=10,
           vmin=0, vmax=0.25)
tops = np.argsort(CONF)[-3:]
for k in tops:
    if k in DIRS:
        for t in DIRS[k][1]:
            v = t / (np.linalg.norm(t) + 1e-9) * 0.13
            ax.plot([E[k, 0], E[k, 0] + v[0]], [E[k, 1], E[k, 1] + v[1]],
                    [E[k, 2], E[k, 2] + v[2]], color="C3", lw=1.8, alpha=0.85)
        ax.scatter(*E[k], color="C3", s=110, marker="*")
ax.set_title("Cross-trajectory aliasing: bundle colored by cross-demo action "
             "conflict;\nred fans = amplified conflicting commanded directions "
             "at the 3 hottest aliased states")
ax.view_init(elev=22, azim=-60)
plt.tight_layout()
plt.savefig("analysis/paper/alias_fig.png", dpi=140)
print("VA fig saved", flush=True)
