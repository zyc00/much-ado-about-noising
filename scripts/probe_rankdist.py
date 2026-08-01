"""PR heterogeneity along the trajectory for L2-2K (and L2-200 control).

Settles the probe_rank (PR 1.1) vs probe_rankrec (PR 1.74) discrepancy:
both used the same checkpoint/dataset/functional and differ only in the
20-state query draw. Computes the POS24 Jacobian PR at (a) evenly-spaced
states along demos 0-7 (phase-resolved distribution) and (b) the EXACT
query windows probe_rank drew (default_rng(9) over PD<8 with PW built
over ALL demos). Prints RD lines.
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
NSWEEP = int(os.environ.get("RD_N", "150"))
ARMS = ([tuple(a.split(":")) for a in os.environ["RD_ARMS"].split(",")]
        if os.environ.get("RD_ARMS") else [
    ("L2_2K", "regression", "logs/full_regression_2000/models/model_latest.pt",
     "data/tool_hang_full2ins_2000.hdf5"),
    ("L2_200", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_200.hdf5"),
])

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

    h = h5py.File(dset, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
    # reconstruct probe_rank's PD (window demo-index) from lengths alone
    lens = []
    S8 = {}
    for di, dn in enumerate(names):
        n = h[f"data/{dn}/obs/{OK[0]}"].shape[0]
        lens.append(max(n - Hn - 1, 0))
        if di < 8:
            o = h[f"data/{dn}/obs"]
            S8[di] = np.concatenate([np.asarray(o[k]) for k in OK],
                                    1).astype(np.float32)
    h.close()
    PD = np.concatenate([np.full(m, di) for di, m in enumerate(lens)])
    off = np.concatenate([[0], np.cumsum(lens)])
    rng = np.random.default_rng(9)
    qs = rng.choice(np.where(PD < 8)[0], 20, replace=False)
    QDS = [(int(PD[q]), int(q - off[PD[q]] + 1)) for q in qs]  # (demo, i)

    def pos24(x):
        emb = en({"state": x[None]}, None)
        zz = torch.zeros((1, Hn, 10), device=dev)
        s = torch.zeros((1,), device=dev)
        a1 = fm.get_velocity(s, zz, emb)
        return a1[0, 1:9, 0:3].reshape(-1)

    def pr_at(di, i):
        S = S8[di]
        w = no.normalize(np.stack([S[i - 1], S[i]]))
        x = torch.tensor(w, device=dev)
        J = torch.autograd.functional.jacobian(pos24, x).reshape(
            24, -1).cpu().numpy()
        sv = np.linalg.svd(J, compute_uv=False)
        e = sv ** 2
        return float(e.sum() ** 2 / ((e ** 2).sum() + 1e-18)), float(sv[0])

    # (b) exact probe_rank queries
    prs = [pr_at(di, i) for di, i in QDS]
    R = np.array(prs)
    print(f"RD {arm} probe_rank-queries n=20 PR p50 {np.median(R[:, 0]):.2f} "
          f"mean {R[:, 0].mean():.2f} min {R[:, 0].min():.2f} "
          f"max {R[:, 0].max():.2f}", flush=True)
    print("RD " + arm + " query-detail " + " ".join(
        f"d{di}i{i}:{p:.2f}" for (di, i), (p, _) in zip(QDS, prs)),
        flush=True)

    # (a) even sweep over demos 0-7, phase-resolved
    allp = []
    for di in range(8):
        S = S8[di]
        n = len(S) - Hn - 1
        for i in np.linspace(1, n, NSWEEP // 8, dtype=int):
            p, sv = pr_at(di, int(i))
            allp.append((i / n, p, sv))
    A = np.array(allp)
    print(f"RD {arm} sweep n={len(A)} PR p10 {np.percentile(A[:, 1], 10):.2f} "
          f"p50 {np.median(A[:, 1]):.2f} p90 {np.percentile(A[:, 1], 90):.2f} "
          f"frac(PR<1.3) {np.mean(A[:, 1] < 1.3):.2f}", flush=True)
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        print(f"RD {arm} phase[{lo:.2f}-{hi:.2f}] n={m.sum()} "
              f"PR p50 {np.median(A[m, 1]):.2f} "
              f"svmax p50 {np.median(A[m, 2]):.2f}", flush=True)
print("RD done", flush=True)
