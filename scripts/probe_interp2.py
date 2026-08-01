"""Interpolation anatomy along near-twin cross-demo paths x(l)=(1-l)xi+l xj.
Arms: L2 HT HG MIP MIP1(step1 of MIP ckpt) HEADMSE.
A. Path metrics (21 pts): jump series, flip location l* and magnitude,
   wiggle = sum|df| / |f(1)-f(0)|, P1: dist(MIP1, L2) vs dist(MIP, L2).
B. Exact Jacobians (24-dim executed-pos out x 106 obs in) at 5 l's:
   PR(sv^2), svmax, dependency energy {obj, eefpos, quat, grip},
   top-3 right-subspace rotation between consecutive l's.
C. Step2-corrector test: cos( f_MIP - f_MIP1 , y_i - mean(nbr targets) ).
Strata: conflicted (top-quartile nbr target spread) / sparse (top-quartile
pair distance) / normal. Prints INTJ lines; saves analysis/manifold/interpjac.npz
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
    ("MIP1", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
    ("HEADMSE", "regression_frozentrunk",
     "logs/mp200_headmse_s1000/models/snap_300000.pt"),
]
NPAIR = 45
LGRID = np.linspace(0, 1, 21)
LJAC = [0.0, 0.25, 0.5, 0.75, 1.0]

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


def fwdrot(a):
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
        PT.append(na0.normalize(fwdrot(A[i - 1:i - 1 + Hn])))
        PD.append(di)
h.close()
PW, PT, PD = np.stack(PW), np.stack(PT), np.asarray(PD)
PF = PW.reshape(len(PW), -1)
tPF = torch.tensor(PF)
print(f"INTJ pool {len(PW)}", flush=True)

rng = np.random.default_rng(1)
cand = rng.choice(np.where(PD < 8)[0], 600, replace=False)
pairs = []
for qi in cand:
    d = torch.norm(tPF - torch.tensor(PF[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    j = int(d.argmin())
    r = float(d[j])
    if r > 2.5:
        continue
    nbi = np.argsort(d)[:8]
    T = PT[nbi][:, 1:9, 0:3].reshape(8, -1)
    conf = float(np.sqrt(((T - T.mean(0)) ** 2).mean()))
    disp = (PT[qi][1:9, 0:3].reshape(-1) - T.mean(0))
    pairs.append((qi, j, r, conf, disp))
    if len(pairs) >= 3 * NPAIR:
        break
confs = np.array([p[3] for p in pairs])
rs = np.array([p[2] for p in pairs])
cq, rq = np.percentile(confs, 75), np.percentile(rs, 75)
strat = {}
for p in pairs:
    s = ("conflicted" if p[3] >= cq else
         ("sparse" if p[2] >= rq else "normal"))
    strat.setdefault(s, []).append(p)
sel = {k: v[:15] for k, v in strat.items()}
print("INTJ pairs " + " ".join(f"{k}:{len(v)}" for k, v in sel.items()),
      flush=True)

OBJ = list(range(0, 44)) + list(range(53, 97))
EEF = [44, 45, 46, 97, 98, 99]
QUA = [47, 48, 49, 50, 100, 101, 102, 103]
GRI = [51, 52, 104, 105]

out_npz = {}
for arm, loss, ck in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    fm, en = ag.flow_map_ema, ag.encoder_ema
    dev = cfg.optimization.device
    tts = float(cfg.optimization.t_two_step)

    def fwd(x):
        emb = en({"state": x}, None)
        b = x.shape[0]
        z = torch.zeros((b, Hn, 10), device=x.device)
        s = torch.zeros((b,), device=x.device)
        a1 = fm.get_velocity(s, z, emb)
        if arm == "MIP":
            t = torch.full((b,), tts, device=x.device)
            a1 = fm.get_velocity(t, a1, emb)
        return a1[:, 1:9, 0:3].reshape(b, -1)

    for sname, plist in sel.items():
        JMET = []
        PMET = []
        CORR = []
        for (qi, j, r, conf, disp) in plist:
            xi = torch.tensor(PW[qi], device=dev)
            xj = torch.tensor(PW[j], device=dev)
            X = torch.stack([(1 - l) * xi + l * xj for l in LGRID])
            with torch.no_grad():
                F = fwd(X).cpu().numpy()
            jumps = np.linalg.norm(np.diff(F, axis=0), axis=1)
            span = np.linalg.norm(F[-1] - F[0]) + 1e-9
            lstar = float(LGRID[1:][int(jumps.argmax())])
            PMET.append((jumps.sum() / span, jumps.max() /
                         (np.median(jumps) + 1e-12), lstar))
            if arm == "MIP":
                with torch.no_grad():
                    emb = en({"state": xi[None]}, None)
                    z = torch.zeros((1, Hn, 10), device=dev)
                    s0 = torch.zeros((1,), device=dev)
                    a1 = fm.get_velocity(s0, z, emb)
                    t = torch.full((1,), tts, device=dev)
                    a2 = fm.get_velocity(t, a1, emb)
                dc = (a2 - a1)[0, 1:9, 0:3].reshape(-1).cpu().numpy()
                cs = float(dc @ disp / (np.linalg.norm(dc) *
                                        np.linalg.norm(disp) + 1e-12))
                CORR.append((cs, float(np.linalg.norm(dc) /
                                       (np.linalg.norm(disp) + 1e-12))))
            Vprev = None
            for l in LJAC:
                xl = ((1 - l) * xi + l * xj).clone().requires_grad_(True)
                J = torch.autograd.functional.jacobian(
                    lambda z_: fwd(z_[None])[0], xl)
                J = J.reshape(24, -1).cpu().numpy()
                U, S_, Vt = np.linalg.svd(J, full_matrices=False)
                pr = float((S_ ** 2).sum() ** 2 / ((S_ ** 4).sum() + 1e-18))
                en_tot = (J ** 2).sum() + 1e-18
                dep = [float((J[:, g] ** 2).sum() / en_tot)
                       for g in (OBJ, EEF, QUA, GRI)]
                rot = np.nan
                if Vprev is not None:
                    M = Vprev @ Vt[:3].T
                    rot = float(1 - np.abs(np.linalg.svd(M)[1]).mean())
                Vprev = Vt[:3]
                JMET.append((l, pr, float(S_[0]), rot, *dep))
        PM = np.array(PMET)
        JM = np.array(JMET)
        mid = np.isin(JM[:, 0], [0.25, 0.5, 0.75])
        end = ~mid
        print(f"INTJ {arm} {sname} wiggle p50 {np.median(PM[:, 0]):.2f} "
              f"flipmag p50 {np.median(PM[:, 1]):.1f} lstar p50 "
              f"{np.median(PM[:, 2]):.2f} frac_mid(.4-.6) "
              f"{np.mean((PM[:, 2] >= .4) & (PM[:, 2] <= .6)):.2f}",
              flush=True)
        print(f"INTJ {arm} {sname} PR end {np.median(JM[end, 1]):.1f} mid "
              f"{np.median(JM[mid, 1]):.1f} | svmax end "
              f"{np.median(JM[end, 2]):.3f} mid {np.median(JM[mid, 2]):.3f} "
              f"| rot p50 {np.nanmedian(JM[:, 3]):.3f} p90 "
              f"{np.nanpercentile(JM[:, 3], 90):.3f} | dep(obj/eef/qua/gri) "
              f"{np.median(JM[:, 4]):.2f}/{np.median(JM[:, 5]):.2f}/"
              f"{np.median(JM[:, 6]):.2f}/{np.median(JM[:, 7]):.2f}",
              flush=True)
        out_npz[f"{arm}_{sname}_P"] = PM
        out_npz[f"{arm}_{sname}_J"] = JM
        if arm == "MIP" and CORR:
            C = np.array(CORR)
            print(f"INTJ MIPcorr {sname} cos(step2corr, disp) p50 "
                  f"{np.median(C[:, 0]):+.2f} frac>0 "
                  f"{np.mean(C[:, 0] > 0):.2f} | |corr|/|disp| p50 "
                  f"{np.median(C[:, 1]):.2f}", flush=True)
os.makedirs("analysis/manifold", exist_ok=True)
np.savez_compressed("analysis/manifold/interpjac.npz", **out_npz)
print("INTJ done", flush=True)
