"""EXP5 on-cluster: bad-mode projections for all variant operators (opprobe_*.npz and
ridgesweep_*.npz on the PVC), using reference G_gt/G_m/G_p + sig from badmode_ref.npz."""
import glob
import numpy as np

POS = slice(44, 47)
ref = np.load("analysis/recovery/badmode_ref.npz")
sig = ref["sig"]

def posblk(G):
    B = G[0:3, POS] * sig[POS][None, :]
    return (B + B.T) / 2

def rotblk(G):
    return G[3:6, :] * sig[None, :]

S_mse = posblk(ref["G_m"])
w, V = np.linalg.eigh(S_mse)
bad1 = V[:, np.argmax(w)]
S_gt = posblk(ref["G_gt"])
wg, Vg = np.linalg.eigh(S_gt)
R_mse = rotblk(ref["G_m"])
_, _, Vt = np.linalg.svd(R_mse)
vrot = Vt[0]

ops = {"G_GT": ref["G_gt"], "MSE": ref["G_m"], "MIP": ref["G_p"]}
for p in sorted(glob.glob("logs/*/models/opprobe_*.npz")):
    tag = p.split("opprobe_")[1].replace(".npz", "")
    ops[tag] = np.load(p)["G"]
for p in sorted(glob.glob("logs/*/models/ridgesweep_*.npz")):
    z = np.load(p)
    base = p.split("ridgesweep_")[1].replace(".npz", "")
    for k in z.files:
        if k.startswith("G_lam"):
            ops[f"{base}-ridge-{k[5:]}"] = z[k]

print(f"{'model':22s} {'gain@bad1':>10s} {'g@srv0':>8s} {'g@srv1':>8s} {'g@srv2':>8s} {'rot@mseamp':>10s} {'rot_sv1':>8s}")
for name, G in ops.items():
    S = posblk(G); R = rotblk(G)
    gb = float(bad1 @ S @ bad1)
    gs = [float(Vg[:, k] @ S @ Vg[:, k]) for k in range(3)]
    ra = float(np.linalg.norm(R @ vrot))
    sv1 = float(np.linalg.svd(R, compute_uv=False)[0])
    print(f"BADMODE {name:22s} {gb:>+10.4f} {gs[0]:>+8.4f} {gs[1]:>+8.4f} {gs[2]:>+8.4f} {ra:>10.3f} {sv1:>8.3f}")
