"""Reconcile the two deployed-Jacobian rank protocols on the same
checkpoints.

PART CCCLVIII (probe_offsupport_jac): FULL 160-dim sampler chunk at
rollout-visited states -> L2-2k far-PR 1.8 ("no collapse").
PART CDXXV (probe_rank): 24-dim executed-position block at dataset
states -> L2-2k PR 1.1 rank90=1 ("collapse").

Crosses functional {POS24, FULL160 (+sub-blocks)} x state sample
{data-on, roll-on(<2), roll-far(>10)} for L2-2K and L2-200, and adds
the execution check: |cos| between the top right-singular vector of the
POS24 Jacobian and the local demo tangent at data-on states.
Prints RR lines.
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
from mip.samplers import get_sampler

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
NQ = int(os.environ.get("RR_N", "20"))
TAG = os.environ.get("RR_TAG", "l2mp200v2")
ARMS = ([tuple(a.split(":")) for a in os.environ["RR_ARMS"].split(",")]
        if os.environ.get("RR_ARMS") else [
    ("L2_2K", "regression", "logs/full_regression_2000/models/model_latest.pt",
     "data/tool_hang_full2ins_2000.hdf5"),
    ("L2_200", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_200.hdf5"),
])

z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
Wc = np.concatenate([z[f"W{sd}"] for sd in seeds])
Dc = np.concatenate([z[f"D{sd}"] for sd in seeds])
rng = np.random.RandomState(0)
ROLL = {}
for bname, lo, hi in [("roll-on", 0, 2), ("roll-far", 10, 1e9)]:
    idx = np.where((Dc >= lo) & (Dc < hi))[0]
    ROLL[bname] = Wc[rng.choice(idx, min(NQ, len(idx)), replace=False)]


def pr_stats(J):
    S_ = np.linalg.svd(J, compute_uv=False)
    e = S_ ** 2
    pr = float(e.sum() ** 2 / ((e ** 2).sum() + 1e-18))
    cum = np.cumsum(e) / e.sum()
    r90 = int(np.searchsorted(cum, 0.9) + 1)
    return pr, r90, float(S_[0])


for arm, loss, ck, dset in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    Hn = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    cfg.task.horizon = Hn
    ds = make_dataset(cfg.task)
    no = ds.normalizer["obs"]["state"]
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    fm, en = ag.flow_map_ema, ag.encoder_ema
    dev = cfg.optimization.device
    sampler = get_sampler(loss)

    # dataset on-support windows + demo tangents (normalized space)
    h = h5py.File(dset, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[:8]
    PW, TAN = [], []
    for dn in names:
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(
            np.float32)
        Wn = [no.normalize(np.stack([S[i - 1], S[i]]))
              for i in range(1, len(S) - Hn)]
        for i in range(len(Wn) - 1):
            PW.append(Wn[i])
            TAN.append((Wn[i + 1] - Wn[i]).reshape(-1))
    h.close()
    PW, TAN = np.stack(PW), np.stack(TAN)
    qs = np.random.RandomState(9).choice(len(PW), NQ, replace=False)
    STATES = {"data-on": [(torch.tensor(PW[q]), TAN[q]) for q in qs]}
    for bname, Wb in ROLL.items():
        STATES[bname] = [(torch.tensor(no.normalize(w),
                                       dtype=torch.float32), None)
                         for w in Wb]

    def pos24(x):
        emb = en({"state": x[None]}, None)
        zz = torch.zeros((1, Hn, 10), device=dev)
        s = torch.zeros((1,), device=dev)
        a1 = fm.get_velocity(s, zz, emb)
        return a1[0, 1:9, 0:3].reshape(-1)

    def full160(x):
        a0 = torch.zeros((1, Hn, 10), device=dev)
        return sampler(cfg.optimization, fm, en, a0,
                       {"state": x.reshape(1, 2, 53)}).reshape(-1)

    for sname, items in STATES.items():
        P24, F160, FPOS, FROT, FGRP, COS = [], [], [], [], [], []
        for x, tan in items:
            xg = x.to(dev)
            J24 = torch.autograd.functional.jacobian(
                pos24, xg).reshape(24, -1).cpu().numpy()
            P24.append(pr_stats(J24))
            Jf = torch.autograd.functional.jacobian(
                full160, xg.reshape(-1), vectorize=True)
            Jf = Jf.reshape(Hn, 10, -1).cpu().numpy()
            F160.append(pr_stats(Jf.reshape(Hn * 10, -1)))
            FPOS.append(pr_stats(Jf[1:9, 0:3].reshape(24, -1)))
            FROT.append(pr_stats(Jf[:, 3:9].reshape(Hn * 6, -1)))
            FGRP.append(pr_stats(Jf[:, 9].reshape(Hn, -1)))
            if tan is not None:
                _, _, Vt = np.linalg.svd(J24, full_matrices=False)
                t = tan / (np.linalg.norm(tan) + 1e-12)
                COS.append(abs(float(Vt[0] @ t)))
        for tag, rows in [("POS24", P24), ("FULL160", F160),
                          ("FULLposblk", FPOS), ("FULLrot", FROT),
                          ("FULLgrip", FGRP)]:
            R = np.array(rows)
            print(f"RR {arm} {sname} {tag} PR p50 {np.median(R[:, 0]):.2f} "
                  f"rank90 p50 {np.median(R[:, 1]):.0f} "
                  f"svmax p50 {np.median(R[:, 2]):.3f}", flush=True)
        if COS:
            print(f"RR {arm} {sname} TANALIGN |cos(v1,tangent)| "
                  f"p50 {np.median(COS):.3f} p25 {np.percentile(COS, 25):.3f} "
                  f"p75 {np.percentile(COS, 75):.3f}", flush=True)
print("RR done", flush=True)
