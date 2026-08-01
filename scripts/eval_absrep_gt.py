"""STEP 1 (GT representation conversion): does the servo law live in the DELTA action
representation, or does it survive in ABS (goal-pose) representation?
Analytic conversion (no env): goal_pos = eef_pos + 0.05*a_pos ;
goal_ori = R(0.5*a_rot) @ R(eef_quat)  (robosuite OSC delta, world-frame premultiply).
Fit dGT ~ G dz for both representations in PHYSICAL units (m, rad), same pairs, same ridge.
Readouts: R^2, pos-block sym-eigs, top SVs, norm ratio ||G_abs||/||G_delta||."""
import numpy as np
import h5py
from scipy.spatial import cKDTree
import robosuite.utils.transform_utils as T
np.set_printoptions(precision=4, suppress=True)
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47); QUAT = slice(47, 51)

def read(path, nmax=None):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g])[:nmax] if nmax else list(h[g])
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/{k}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out

def goal_of(obs_row, act):
    p = obs_row[POS] + 0.05 * act[:3]
    Rd = T.quat2mat(T.axisangle2quat(0.5 * act[3:6]))
    Rg = Rd @ T.quat2mat(obs_row[QUAT])
    return p, Rg

def rotdiff(Ra, Rb):  # axis-angle of Ra relative to Rb
    return T.quat2axisangle(T.mat2quat(Ra @ Rb.T))

clean = read("data/tool_hang_full2ins_2000.hdf5", 40)
cl = np.concatenate([ov for ov, _ in clean], 0)
owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
mu, sig = cl.mean(0), cl.std(0) + 1e-6
tree = cKDTree((cl - mu) / sig)
test = read("data/dart_test_huge_full2ins.hdf5")

DZ, D_DEL, D_ABS = [], [], []
for ov, acts in test:
    for t in range(1, len(acts) - 1, 4):
        z = (ov[t] - mu) / sig
        d, idx = tree.query(z)
        if not (2.0 <= d < 4.0): continue
        i0, t0 = map(int, owner[int(idx)])
        a0 = clean[i0][1][min(t0, len(clean[i0][1]) - 1)]
        gt = acts[t]
        # delta representation, physical units
        dpos_del = 0.05 * (gt[:3] - a0[:3])
        Rd_s = T.quat2mat(T.axisangle2quat(0.5 * gt[3:6]))
        Rd_0 = T.quat2mat(T.axisangle2quat(0.5 * a0[3:6]))
        drot_del = rotdiff(Rd_s, Rd_0)
        # abs representation
        gp_s, Rg_s = goal_of(ov[t], gt)
        gp_0, Rg_0 = goal_of(clean[i0][0][t0], a0)
        dpos_abs = gp_s - gp_0
        drot_abs = rotdiff(Rg_s, Rg_0)
        DZ.append(z - (cl[int(idx)] - mu) / sig)
        D_DEL.append(np.concatenate([dpos_del, drot_del]))
        D_ABS.append(np.concatenate([dpos_abs, drot_abs]))
DZ = np.stack(DZ); D_DEL = np.stack(D_DEL); D_ABS = np.stack(D_ABS)
print(f"pairs in [2,4): {len(DZ)}   units: meters / radians")

def ridge(X, Y, lam=1e-2):
    A = X.T @ X + lam * len(X) * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ Y).T
def r2cv(X, Y, k=5):
    rng = np.random.RandomState(0); idx = rng.permutation(len(X)); f = np.array_split(idx, k)
    sr = st_ = 0
    for i in range(k):
        te = f[i]; tr = np.concatenate([f[j] for j in range(k) if j != i])
        G = ridge(X[tr], Y[tr]); P = X[te] @ G.T
        sr += np.sum((Y[te] - P) ** 2); st_ += np.sum((Y[te] - Y[tr].mean(0)) ** 2)
    return 1 - sr / st_

for name, Y in [("DELTA repr", D_DEL), ("ABS repr", D_ABS)]:
    G = ridge(DZ, Y)
    B = G[0:3, POS] * sig[POS][None, :]   # d(goal-pos or delta-pos, m) / d(eef-pos deviation, m-ish)
    w = np.linalg.eigvalsh((B + B.T) / 2)
    s = np.linalg.svd(G, compute_uv=False)
    print(f"\n== {name} ==")
    print(f"  5-fold R2            : {r2cv(DZ, Y):.3f}")
    print(f"  |dGT| mean           : {np.linalg.norm(Y, axis=1).mean():.4f}")
    print(f"  pos-block sym-eigs   : {w}  ({'NEG-def (servo)' if np.all(w<0) else 'not neg-def'})")
    print(f"  pos-block:\n{B}")
    print(f"  top-5 SVs            : {s[:5]}")
    print(f"  ||G||_F              : {np.linalg.norm(G):.4f}")
G_d = ridge(DZ, D_DEL); G_a = ridge(DZ, D_ABS)
print(f"\n||G_abs|| / ||G_delta|| = {np.linalg.norm(G_a)/np.linalg.norm(G_d):.3f}")
np.savez("analysis/recovery/absrep_gt.npz", DZ=DZ, D_DEL=D_DEL, D_ABS=D_ABS, G_d=G_d, G_a=G_a)
print("saved analysis/recovery/absrep_gt.npz")
