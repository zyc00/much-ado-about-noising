"""H2 test: canonicalized relabeling. For each step of each demo, if a
cross-demo near-twin with LOWER demo index exists within EPS, replace the
action by that twin's action -> approximately state-feedback (conflict-free)
supervision from the same state distribution. Writes _relabel.hdf5."""
import os
import shutil

import h5py
import numpy as np
import torch

SRC = "data/tool_hang_full2ins_mp_200.hdf5"
DST = "data/tool_hang_full2ins_mp200_relabel.hdf5"
EPS = float(os.environ.get("RL_EPS", "0.5"))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
shutil.copy(SRC, DST)
h = h5py.File(DST, "r+")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
S_all, A_all, D_all, I_all = [], [], [], []
mu_sd = None
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    S_all.append(S)
    A_all.append(A)
    D_all.append(np.full(len(S), di))
    I_all.append(np.arange(len(S)))
Sc = np.concatenate(S_all)
mu, sd = Sc.mean(0), Sc.std(0) + 1e-6
Z = torch.tensor((Sc - mu) / sd, device="cuda")
D = np.concatenate(D_all)
Ii = np.concatenate(I_all)
A = np.concatenate(A_all)
tD = torch.tensor(D, device="cuda")
nrep = 0
ptr = 0
for di, dn in enumerate(names):
    L = len(S_all[di])
    Anew = A_all[di].copy()
    if di > 0:
        zq = Z[ptr:ptr + L]
        mask = tD < di
        Zp = Z[mask]
        Ap = A[mask.cpu().numpy()]
        d = torch.cdist(zq, Zp)
        mn, am = d.min(dim=1)
        mn = mn.cpu().numpy() / np.sqrt(Z.shape[1])
        am = am.cpu().numpy()
        for i in range(L):
            if mn[i] < EPS:
                Anew[i] = Ap[am[i]]
                nrep += 1
    dset = h[f"data/{dn}/actions"]
    dset[...] = Anew
    ptr += L
h.close()
print(f"RELABEL done: replaced {nrep} / {len(A)} steps", flush=True)
