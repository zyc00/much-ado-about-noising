"""Does HT's learned sigma(x) track the DATA-GEOMETRIC uncertainty
D(r_nn(x)) — the kNN target dispersion at the dataset's available
resolution? If yes, the hetero objective's 'noise' estimate is a
measurement of local underdetermination (steepness x sparsity), not of any
actual noise.

Per MP-200 training state (6000 windows, same sampling as
probe_loss_balance): data side disp(x) = mean squared difference between
the state's own normalized target chunk and its k=8 nearest CROSS-
trajectory neighbors' chunks (full-dims and gripper-only); model side
sigma(x) from HT snapshots 20k/60k. Prints SGEO lines with per-decile
means and Pearson-on-logs / Spearman correlations.
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
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
H = 8
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

# ---- pool: every 2nd step of every demo (for the kNN structure) ----------
h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PX, PY, PP, PT = [], [], [], []
for ti, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    if L < H + 3:
        continue
    for i in range(1, L - H, 2):
        PX.append(np.stack([S[i - 1], S[i]]).reshape(-1))
        PY.append(A[i:i + H])
        PP.append(i / L)
        PT.append(ti)
h.close()
PX, PY = np.stack(PX), np.stack(PY)
PP, PT = np.asarray(PP), np.asarray(PT, dtype=np.int64)
xmu, xsd = PX.mean(0), PX.std(0) + 1e-6
amu = PY.reshape(-1, 7).mean(0)
asd = PY.reshape(-1, 7).std(0) + 1e-6
Xn = (PX - xmu) / xsd
Yn = (PY - amu) / asd
tree = cKDTree(Xn)
print(f"SGEO pool {len(PX)}", flush=True)

# ---- queries: 6000 states, same RNG scheme as probe_loss_balance ---------
rng = np.random.RandomState(0)
qi = rng.choice(len(PX), 6000, replace=False)
disp_all, disp_grip, rnn = [], [], []
for q in qi:
    d, nb = tree.query(Xn[q], k=40)
    m = PT[nb] != PT[q]
    nb, d = nb[m][:8], d[m][:8]
    dy2 = (Yn[nb] - Yn[q]) ** 2                # (8, H, 7)
    disp_all.append(dy2.mean())
    disp_grip.append(dy2[..., 6].mean())
    rnn.append(d[0])
disp_all, disp_grip = np.asarray(disp_all), np.asarray(disp_grip)
rnn = np.asarray(rnn)
dec = np.clip((PP[qi] * 10).astype(int), 0, 9)
print("SGEO data disp_all_decile " +
      " ".join(f"{disp_all[dec == d].mean():.3f}" for d in range(10)),
      flush=True)
print("SGEO data disp_grip_decile " +
      " ".join(f"{disp_grip[dec == d].mean():.3f}" for d in range(10)),
      flush=True)

# ---- model side: HT sigma at snapshots -----------------------------------
with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.loss_type=regression_hetero_t",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
Hn = int(cfg.task.horizon)
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]
dev = cfg.optimization.device

W = PX[qi].reshape(-1, 2, 53)
for tag, ck in [("20k", "logs/mp200_ht_s1000/models/snap_20000.pt"),
                ("60k", "logs/mp200_ht_s1000/models/snap_60000.pt")]:
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    SIG = []
    for i in range(0, len(W), 256):
        xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), Hn, 10), device=dev)
            t0 = torch.zeros(len(xb), device=dev)
            emb = ag.encoder_ema({"state": xb}, None)
            _, s_raw = ag.flow_map_ema.net(a0, t0, t0, emb)
            SIG.append((torch.nn.functional.softplus(s_raw)
                        .reshape(len(xb), -1).mean(1) + 1e-3).cpu().numpy())
    SIG = np.concatenate(SIG)
    print(f"SGEO HT-{tag} sigma_decile " +
          " ".join(f"{SIG[dec == d].mean():.4f}" for d in range(10)),
          flush=True)
    for lab, dv in (("all", disp_all), ("grip", disp_grip)):
        m = (dv > 0) & (SIG > 0)
        r = float(np.corrcoef(np.log(dv[m] + 1e-6), np.log(SIG[m]))[0, 1])
        rs = float(spearmanr(dv[m], SIG[m]).statistic)
        print(f"SGEO HT-{tag} corr(sigma, disp_{lab}) pearson_log {r:+.3f} "
              f"spearman {rs:+.3f} (n={int(m.sum())})", flush=True)
print("SGEO done", flush=True)
