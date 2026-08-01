"""Mode selection vs mode averaging on HUMAN tool-hang data.

For query states, gather K nearest CROSS-DEMO neighbor states and their
demo actions (raw first-executed dpos, 3D). 2-means the neighbor actions;
where bimodal (separation > 2, both clusters >= 3), score each model's
prediction: axis position t along c_near->c_far (0 = at nearest mode,
0.5 = cluster midpoint/average), selection index, speed percentile
within neighbor speeds, direction cosines. Stratified by neighbor mean
speed tercile (action range). Arms: hMSE / hHT / hMIP. Prints MODE lines.
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

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
D = os.environ.get("MODE_DATA",
                   "/mnt/pfs/yuchen/data/mip/robomimic/tool_hang/ph/"
                   "low_dim.hdf5")
ARMS = [("hMSE", "regression", "logs/snap_hmse_chi/models/model_latest.pt"),
        ("hHT", "regression_hetero_t", "logs/aht_s1000/models/model_latest.pt"),
        ("hMIP", "mip", "logs/hmip0/models/model_latest.pt")]
NQ, K = 500, 12

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg0 = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + D, "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg0, False)
cfg0.task.obs_dim = 53
Hn = int(2 ** np.ceil(np.log2(cfg0.task.horizon)))
cfg0.task.horizon = Hn
ds = make_dataset(cfg0.task)
no = ds.normalizer["obs"]["state"]
na = ds.normalizer["action"]

h = h5py.File(D, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PD, PA = [], [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"]).astype(np.float32)
    for i in range(1, min(len(S), len(A)) - Hn):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])))
        PD.append(di)
        PA.append(A[i][:3])  # raw dpos executed from S[i]
h.close()
PW, PD, PA = np.stack(PW), np.asarray(PD), np.stack(PA)
PF = PW.reshape(len(PW), -1)
tPF = torch.tensor(PF)
print(f"MODE pool {len(PW)} windows, {PD.max() + 1} demos", flush=True)

rng = np.random.default_rng(3)
qs = rng.choice(len(PW), NQ, replace=False)
NEIGH = []
for qi in qs:
    d = torch.norm(tPF - torch.tensor(PF[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    idx = np.argsort(d)[:K]
    NEIGH.append((qi, idx, d[idx]))


def kmeans2(X, iters=15):
    c = X[[0, len(X) // 2]].copy()
    for _ in range(iters):
        a = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in (0, 1):
            if (a == j).any():
                c[j] = X[a == j].mean(0)
    return c, a


BIMODAL, UNIMODAL = [], []
for qi, idx, dists in NEIGH:
    X = PA[idx]
    c, a = kmeans2(X)
    w = np.concatenate([X[a == j] - c[j] for j in (0, 1)])
    within = np.sqrt((w ** 2).sum(1).mean()) + 1e-9
    sep = np.linalg.norm(c[0] - c[1]) / within
    if sep > 2.0 and min((a == 0).sum(), (a == 1).sum()) >= 3:
        BIMODAL.append((qi, idx, c, a, sep))
    else:
        UNIMODAL.append((qi, idx))
print(f"MODE bimodal {len(BIMODAL)} unimodal {len(UNIMODAL)} "
      f"(sep>2, minclust>=3)", flush=True)

for arm, loss, ck in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    fm, en = ag.flow_map_ema, ag.encoder_ema
    dev = cfg.optimization.device
    two = loss.startswith("mip")
    tts = float(cfg.optimization.t_two_step)

    def pred_dpos(w):
        x = torch.tensor(w, device=dev)[None]
        with torch.no_grad():
            emb = en({"state": x}, None)
            z = torch.zeros((1, Hn, 10), device=dev)
            s = torch.zeros((1,), device=dev)
            a1 = fm.get_velocity(s, z, emb)
            if two:
                t = torch.full((1,), tts, device=dev)
                a1 = fm.get_velocity(t, a1, emb)
        au = na.unnormalize(a1.cpu().numpy())[0]
        return au[1, 0:3]

    # offset calibration on unimodal states (o in {0,1} on the pred slice
    # already fixed; calibrate label index implicitly via unimodal error)
    uni_err = []
    for qi, idx in UNIMODAL[:150]:
        p = pred_dpos(PW[qi])
        uni_err.append(np.linalg.norm(p - PA[idx].mean(0)))
    print(f"MODE {arm} unimodal |pred-labelmean| p50 "
          f"{np.median(uni_err):.4f}", flush=True)

    rows = []
    for qi, idx, c, a, sep in BIMODAL:
        p = pred_dpos(PW[qi])
        d0, d1 = np.linalg.norm(p - c[0]), np.linalg.norm(p - c[1])
        near, far = (0, 1) if d0 <= d1 else (1, 0)
        u = (c[far] - c[near])
        L = np.linalg.norm(u) + 1e-12
        t_axis = float((p - c[near]) @ u / L ** 2)  # 0=near mode .5=mid
        sel = float(np.linalg.norm(p - c[far]) /
                    (np.linalg.norm(p - c[near]) +
                     np.linalg.norm(p - c[far]) + 1e-12))
        spds = np.linalg.norm(PA[idx], axis=1)
        sp = np.linalg.norm(p)
        pctl = float((spds < sp).mean())
        spd_ratio = sp / (spds.mean() + 1e-9)
        pd_ = p / (np.linalg.norm(p) + 1e-12)
        mn = PA[idx].mean(0)
        cos_near = float(pd_ @ (c[near] / (np.linalg.norm(c[near]) + 1e-12)))
        cos_mean = float(pd_ @ (mn / (np.linalg.norm(mn) + 1e-12)))
        rows.append((spds.mean(), t_axis, sel, pctl, spd_ratio,
                     cos_near, cos_mean))
    R = np.array(rows)
    ter = np.quantile(R[:, 0], [1 / 3, 2 / 3])
    for lab, m in [("slow", R[:, 0] <= ter[0]),
                   ("mid", (R[:, 0] > ter[0]) & (R[:, 0] <= ter[1])),
                   ("fast", R[:, 0] > ter[1]),
                   ("ALL", np.ones(len(R), bool))]:
        r = R[m]
        print(f"MODE {arm} {lab} n={m.sum()} t_axis p50 "
              f"{np.median(r[:, 1]):.3f} sel p50 {np.median(r[:, 2]):.3f} "
              f"frac(t<0.25) {np.mean(r[:, 1] < 0.25):.2f} "
              f"frac(0.35<t<0.65) {np.mean((r[:, 1] > 0.35) & (r[:, 1] < 0.65)):.2f} "
              f"spd_pctl p50 {np.median(r[:, 3]):.2f} spd_ratio p50 "
              f"{np.median(r[:, 4]):.2f} cos_near p50 "
              f"{np.median(r[:, 5]):.3f} cos_mean p50 "
              f"{np.median(r[:, 6]):.3f}", flush=True)
print("MODE done", flush=True)
