"""Annulus occupancy: for sample states of the first 8 demos, count
cross-demo neighbors with eef distance in [5,50]mm AND normalized-obs
distance < 2.0, at 200-demo vs 2000-demo pool. Prints AO lines."""
import os
import sys

sys.path.insert(0, ".")
import h5py
import numpy as np

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
D = "data/tool_hang_full2ins_2000.hdf5"
h = h5py.File(D, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
S_all, E_all, D_all = [], [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    S_all.append(S)
    E_all.append(S[:, 44:47])
    D_all.append(np.full(len(S), di))
h.close()
mu = np.concatenate(S_all).mean(0)
sd = np.concatenate(S_all).std(0)
sd[sd == 0] = 1
E = np.concatenate(E_all)
DD = np.concatenate(D_all)
SN = (np.concatenate(S_all) - mu) / sd
for pool, tag in [(200, "pool200"), (2000, "pool2k")]:
    m = DD < pool
    Ep, SNp, Dp = E[m], SN[m], DD[m]
    cnts, rnns = [], []
    for di in range(8):
        qm = Dp == di
        Q, QS = Ep[qm][::10], SNp[qm][::10]
        for q, qs in zip(Q, QS):
            de = np.linalg.norm(Ep - q, axis=1)
            ds = np.linalg.norm(SNp - qs, axis=1) / np.sqrt(SNp.shape[1])
            ok = (Dp != di) & (de >= 0.005) & (de <= 0.05) & (ds < 0.5)
            cnts.append(int(ok.sum()))
            oth = de[(Dp != di) & (ds < 0.5)] if ((Dp != di) & (ds < 0.5)).any() else np.array([9.9])
            rnns.append(float(oth.min()))
    cnts, rnns = np.asarray(cnts), np.asarray(rnns)
    print(f"AO {tag} annulus-neighbors p50 {np.median(cnts):.0f} p10 "
          f"{np.percentile(cnts, 10):.0f} frac_zero {np.mean(cnts == 0):.2f} "
          f"| cross-demo eef r_nn p50 {1000 * np.median(rnns):.1f}mm p90 "
          f"{1000 * np.percentile(rnns, 90):.1f}mm", flush=True)
print("AO done", flush=True)
