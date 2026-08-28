"""EXP5: bad-mode projection analysis.
Define MSE's two bad modes from its fitted operator:
  (1) positive-feedback pose eigendirection (eigvec of the positive eigenvalue of
      sym de-z-scored pos-block);
  (2) rank-1 rotation-amplifier input direction (top right-singular vector of the
      de-z-scored rot-block rows 3:6).
For each available operator G, report: signed gain along bad1, gains along the GT servo
eigendirections, rotation response along MSE's amplifier direction, own rot top-sv.
Usage: python bad_mode_projection.py [extra_npz_with_G ...]
Each extra npz must contain 'G' (6x53) fitted in the standard z-scored pairs protocol;
name taken from filename."""
import sys
import numpy as np
import h5py

OKEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)

# --- sig of the 40-demo anchor cloud (standard z-scoring of the protocol) ---
h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
cl = []
for i in range(40):
    o = h[f"data/demo_{i}/obs"]
    cl.append(np.concatenate([np.asarray(o[k]) for k in OKEYS], axis=1).astype(np.float32))
h.close()
cl = np.concatenate(cl, 0)
sig = cl.std(0) + 1e-6

d = np.load("analysis/recovery/operator_fit_b24.npz")
ops = {"G_GT": d["G_gt"], "MSE": d["G_m"], "MIP": d["G_p"]}
for p in sys.argv[1:]:
    z = np.load(p)
    if "G" in z:
        name = p.split("/")[-1].replace("opprobe_", "").replace(".npz", "")
        ops[name] = z["G"]

def posblk(G):
    B = G[0:3, POS] * sig[POS][None, :]
    return (B + B.T) / 2

def rotblk(G):
    return G[3:6, :] * sig[None, :]

S_mse = posblk(ops["MSE"])
w, V = np.linalg.eigh(S_mse)
bad1 = V[:, np.argmax(w)]           # positive-feedback direction (+eig)
S_gt = posblk(ops["G_GT"])
wg, Vg = np.linalg.eigh(S_gt)       # all negative; columns = servo dirs
R_mse = rotblk(ops["MSE"])
_, _, Vt = np.linalg.svd(R_mse)
vrot = Vt[0]                         # amplifier input direction (53-d, z-scored coords)

print(f"MSE pos-block eigs: {np.round(w,3).tolist()}  (bad1 = eig {w.max():+.3f} dir)")
print(f"GT  pos-block eigs: {np.round(wg,3).tolist()}")
print(f"MSE rot-block top sv: {np.linalg.svd(R_mse, compute_uv=False)[0]:.3f}")
print()
hdr = f"{'model':14s} {'gain@bad1':>10s} " + " ".join(f"{'g@srv'+str(k):>8s}" for k in range(3)) + f" {'rot@mseamp':>10s} {'rot_sv1':>8s}"
print(hdr)
for name, G in ops.items():
    S = posblk(G); R = rotblk(G)
    gb = float(bad1 @ S @ bad1)
    gs = [float(Vg[:, k] @ S @ Vg[:, k]) for k in range(3)]
    ra = float(np.linalg.norm(R @ vrot))
    sv1 = float(np.linalg.svd(R, compute_uv=False)[0])
    print(f"{name:14s} {gb:>+10.4f} " + " ".join(f"{g:>+8.4f}" for g in gs) + f" {ra:>10.3f} {sv1:>8.3f}")
