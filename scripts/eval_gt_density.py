"""Re-measure G_GT with ANCHOR DENSITY sweep (support cloud from 40 -> 20,000 demos) on
the SAME test states / same z-ruler / same [2,4)-band (defined by the original 40-demo
cloud). Hypothesis (waypoint+PD): with accurate anchors (true tube foot points), dz becomes
purely NORMAL to the tube and da* strictly obeys the PD law: linearity R^2 rises, pos-block
cleanly negative-definite. Also tangent/normal decomposition at the densest cloud:
tangential component of the law should match script-advance; normal component = -K (PD).
GT-only, no models, no env."""
import numpy as np
import h5py
from scipy.spatial import cKDTree
np.set_printoptions(precision=3, suppress=True)
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)


def read(path, nmax=None, obs_only=False):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
    ks = sorted(h[g].keys(), key=lambda k: int(k.split("_")[1]))[:nmax] if nmax else list(h[g].keys())
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = None if obs_only else np.clip(np.asarray(h[f"{g}/{k}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out


def ridge(X, Y, lam=1e-2):
    A = X.T @ X + lam * len(X) * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ Y).T


def r2cv(X, Y, k=5):
    rng = np.random.RandomState(0); idx = rng.permutation(len(X)); f = np.array_split(idx, k)
    sr = st = 0.0
    for i in range(k):
        te = f[i]; tr = np.concatenate([f[j] for j in range(k) if j != i])
        G = ridge(X[tr], Y[tr]); P = X[te] @ G.T
        sr += np.sum((Y[te] - P) ** 2); st += np.sum((Y[te] - Y[tr].mean(0)) ** 2)
    return 1 - sr / st


def main():
    print("loading 20k demos (obs+actions)...", flush=True)
    demos = read("data/tool_hang_full2ins_20000.hdf5", 20000)
    cl40 = np.concatenate([ov for ov, _ in demos[:40]], 0)
    mu, sig = cl40.mean(0), cl40.std(0) + 1e-6   # SAME ruler as all prior analyses
    tree40 = cKDTree((cl40 - mu) / sig)

    SIZES = [40, 200, 1000, 5000, 20000]
    print("building nested clouds/trees...", flush=True)
    Z_all, owner_all = {}, {}
    for n in SIZES:
        Z = np.concatenate([(ov - mu) / sig for ov, _ in demos[:n]], 0)
        owner = np.concatenate([[(i, t) for t in range(len(demos[i][0]))] for i in range(n)], 0)
        Z_all[n] = Z; owner_all[n] = owner
    trees = {n: cKDTree(Z_all[n]) for n in SIZES}
    print("trees ready", flush=True)

    test = read("data/dart_test_huge_full2ins.hdf5")
    # test states selected by the ORIGINAL band (40-demo cloud) for comparability
    sel = []
    for ov, acts in test:
        for t in range(1, len(acts) - 1, 4):
            z = (ov[t] - mu) / sig
            d40, _ = tree40.query(z)
            if 2.0 <= d40 < 4.0:
                sel.append((z, acts[t, :6]))
    Zs = np.stack([s[0] for s in sel]); GTs = np.stack([s[1] for s in sel])
    print(f"test states in original [2,4) band: {len(Zs)}\n")

    print(f"{'cloud':>7} {'anchor d':>9} {'R2':>6} {'pos-block sym-eigs':>26} {'top3 SV':>20} {'res ratio':>9}")
    keep = {}
    for n in SIZES:
        dd, ii = trees[n].query(Zs)
        DZ = Zs - Z_all[n][ii]
        A0 = np.stack([demos[owner_all[n][j][0]][1][min(owner_all[n][j][1], len(demos[owner_all[n][j][0]][1]) - 1), :6] for j in ii])
        DA = GTs - A0
        G = ridge(DZ, DA)
        B = G[0:3, POS] * sig[POS][None, :]
        w = np.linalg.eigvalsh((B + B.T) / 2)
        s = np.linalg.svd(G, compute_uv=False)
        P = DZ @ G.T
        res = np.median(np.linalg.norm(DA - P, axis=1) / np.maximum(np.linalg.norm(DA, axis=1), 1e-6))
        print(f"{n:>7} {dd.mean():>9.2f} {r2cv(DZ, DA):>6.3f} {np.round(w,3)!s:>26} {np.round(s[:3],2)!s:>20} {res:>9.2f}")
        keep[n] = (DZ, DA, ii, dd)

    # tangent/normal decomposition at densest cloud
    n = 20000
    DZ, DA, ii, dd = keep[n]
    U = np.zeros_like(DZ)
    for r, j in enumerate(ii):
        i0, t0 = owner_all[n][j]
        ov = demos[i0][0]
        t1, t2 = max(t0 - 1, 0), min(t0 + 1, len(ov) - 1)
        u = ((ov[t2] - mu) / sig - (ov[t1] - mu) / sig)
        U[r] = u / (np.linalg.norm(u) + 1e-9)
    tang = np.sum(DZ * U, axis=1, keepdims=True)
    DZ_t = tang * U; DZ_n = DZ - DZ_t
    print(f"\ntangent/normal split (20k anchors): |dz_t|/|dz| mean = {np.mean(np.abs(tang[:,0])/np.linalg.norm(DZ,axis=1)):.2f}")
    Gn = ridge(DZ_n, DA); Gt = ridge(DZ_t, DA)
    print(f"  R2(normal-only)  = {r2cv(DZ_n, DA):.3f}")
    print(f"  R2(tangent-only) = {r2cv(DZ_t, DA):.3f}")
    Bn = Gn[0:3, POS] * sig[POS][None, :]
    wn = np.linalg.eigvalsh((Bn + Bn.T) / 2)
    print(f"  NORMAL-law pos-block sym-eigs: {np.round(wn,3)}  ({'STRICT neg-def' if np.all(wn<0) else 'not neg-def'})")
    np.savez("analysis/recovery/gt_density.npz", DZ=DZ, DA=DA, U=U)
    print("saved analysis/recovery/gt_density.npz")


if __name__ == "__main__":
    main()
