"""Inter/extrapolation anatomy at conflicted MP-200 states, per arm.
For each query x with nearest cross-demo near-twin x':
  midpoint: loss of f((x+x')/2) vs (y+y')/2, excess over endpoint losses
  tangent jitter x+0.25(x'-x): deviation from linear target interp
  normal rays x+delta*u (u perp to local tangent PCs), delta in multiples of
  r_nn: gain |df|/delta and DC-share |mean_t df| / rms_t|df|
Prints IX lines."""
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
]
DELTAS = [0.25, 0.5, 1.0, 2.0]
KNB = 8

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg0 = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg0, False)
cfg0.task.obs_dim = 53
cfg0.task.horizon = int(2 ** np.ceil(np.log2(cfg0.task.horizon)))
Hn = int(cfg0.task.horizon)
ds0 = make_dataset(cfg0.task)
no, na0 = ds0.normalizer["obs"]["state"], ds0.normalizer["action"]


def fwd(a):
    rot = ds0.rotation_transformer.forward(a[..., 3:6])
    return np.concatenate([a[..., :3], rot, a[..., [6]]], -1)


h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PT, PD = [], [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    for i in range(1, len(S) - Hn):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])))
        PT.append(na0.normalize(fwd(A[i - 1:i - 1 + Hn])))
        PD.append(di)
h.close()
PW, PT, PD = np.stack(PW), np.stack(PT), np.asarray(PD)
PF = PW.reshape(len(PW), -1)
print(f"IX pool {len(PW)}", flush=True)

tPF = torch.tensor(PF)
rng = np.random.default_rng(0)
qidx = rng.choice(np.where(PD < 8)[0], 400, replace=False)
Q = []
for qi in qidx:
    d = torch.norm(tPF - torch.tensor(PF[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    j = int(d.argmin())
    r_nn = float(d[j])
    if r_nn > 2.0:
        continue
    T = PA = PT[torch.topk(torch.tensor(-d), KNB).indices.numpy()]
    conf = float(np.sqrt(((T - T.mean(0)) ** 2).mean()))
    nbi = torch.topk(torch.tensor(-d), KNB).indices.numpy()
    NB = PF[nbi] - PF[qi]
    U, S_, Vt = np.linalg.svd(NB, full_matrices=False)
    Q.append((qi, j, r_nn, conf, Vt[:4]))
conf_med = float(np.median([q[3] for q in Q]))
print(f"IX queries {len(Q)} conf_med {conf_med:.4f}", flush=True)

for arm, loss, ck in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    dev = cfg.optimization.device

    def pred(xf):
        xb = torch.tensor(np.asarray(xf, dtype=np.float32).reshape(
            len(xf), 2, -1), device=dev)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), Hn, 10), device=dev)
            return sampler(cfg.optimization, ag.flow_map_ema,
                           ag.encoder_ema, a0, {"state": xb}).cpu().numpy()

    stats = {k: [] for k in ["Lend", "Lmid_ex", "Ltan_ex"]}
    gain = {d: [] for d in DELTAS}
    dcsh = {d: [] for d in DELTAS}
    gain_t = {d: [] for d in DELTAS}
    B = 64
    for bi in range(0, len(Q), B):
        qb = Q[bi:bi + B]
        xs = np.stack([PF[q[0]] for q in qb])
        xps = np.stack([PF[q[1]] for q in qb])
        ys = np.stack([PT[q[0]] for q in qb])
        yps = np.stack([PT[q[1]] for q in qb])
        base = pred(xs)
        pmid = pred((xs + xps) / 2)
        ptan = pred(xs + 0.25 * (xps - xs))
        e_end = ((base - ys) ** 2).mean((1, 2))
        e_endp = ((pred(xps) - yps) ** 2).mean((1, 2))
        e_mid = ((pmid - (ys + yps) / 2) ** 2).mean((1, 2))
        e_tan = ((ptan - (0.75 * ys + 0.25 * yps)) ** 2).mean((1, 2))
        stats["Lend"].extend(e_end)
        stats["Lmid_ex"].extend(e_mid - (e_end + e_endp) / 2)
        stats["Ltan_ex"].extend(e_tan - e_end)
        for dmul in DELTAS:
            xn = []
            for k, q in enumerate(qb):
                V = q[4]
                g = rng.standard_normal(PF.shape[1])
                g = g - V.T @ (V @ g)
                g = g / (np.linalg.norm(g) + 1e-9)
                xn.append(xs[k] + dmul * q[2] * g)
            dfn = pred(np.stack(xn)) - base
            dft = pred(xs + dmul * 0.5 * (xps - xs)) - base
            for k, q in enumerate(qb):
                nrm = np.sqrt((dfn[k] ** 2).mean())
                gain[dmul].append(nrm / (dmul * q[2]))
                dc = np.linalg.norm(dfn[k][1:9].mean(0)) / (np.sqrt(
                    (dfn[k][1:9] ** 2).sum(1)).mean() + 1e-9)
                dcsh[dmul].append(dc)
                gain_t[dmul].append(np.sqrt((dft[k] ** 2).mean()) /
                                    (dmul * 0.5 * q[2] + 1e-9))
    print(f"IX {arm} Lend p50 {np.median(stats['Lend']):.2e} | mid-excess "
          f"p50 {np.median(stats['Lmid_ex']):.2e} p90 "
          f"{np.percentile(stats['Lmid_ex'], 90):.2e} | tan-excess p50 "
          f"{np.median(stats['Ltan_ex']):.2e}", flush=True)
    for dmul in DELTAS:
        print(f"IX {arm} d={dmul} gainN p50 {np.median(gain[dmul]):.4f} "
              f"gainT p50 {np.median(gain_t[dmul]):.4f} DCshare p50 "
              f"{np.median(dcsh[dmul]):.3f} p90 "
              f"{np.percentile(dcsh[dmul], 90):.3f}", flush=True)
print("IX done", flush=True)
