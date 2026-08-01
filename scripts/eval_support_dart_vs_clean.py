"""Is the region that is OFF the clean manifold actually ON the DART manifold?
Build two reference clouds (clean vs DART-proxy) with a COMMON standardization (clean
mu/sig, so distances share one scale) and a MATCHED number of points (1-NN distance
shrinks with cloud size, so equalize it). For each held-out test state, compute
d_clean and d_dart. Report d_dart binned by d_clean, and the fraction of off-clean
states that are in-DART. Pure obs geometry: no env, no model.

On the noise question: DART noise FATTENS the cloud (fills the off-clean region) ->
that is exactly why it covers more. Using clean mu/sig keeps the ruler identical;
matching point counts removes the density confound."""
import numpy as np
import h5py
from scipy.spatial import cKDTree
np.set_printoptions(precision=3, suppress=True)

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def stack(path, ndemo=None, stride=1):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g])
    if ndemo: ks = ks[:ndemo]
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        v = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        out.append(v[::stride])
    h.close(); return np.concatenate(out, 0)


# common standardization from CLEAN (same ruler as all other analyses), 40 demos
clean_full = stack("data/tool_hang_full2ins_2000.hdf5", ndemo=40)
mu, sig = clean_full.mean(0), clean_full.std(0) + 1e-6

# matched-size clouds: cap both to the same N points
dart_full = stack("data/tool_hang_dart_full2ins_2000.hdf5", ndemo=40)
N = min(len(clean_full), len(dart_full))
rng = np.random.RandomState(0)
ci = rng.choice(len(clean_full), N, replace=False)
di = rng.choice(len(dart_full), N, replace=False)
clean_cloud = (clean_full[ci] - mu) / sig
dart_cloud = (dart_full[di] - mu) / sig
print(f"cloud points: clean={N}, dart={N} (matched).  ruler = clean mu/sig")
tc = cKDTree(clean_cloud); td = cKDTree(dart_cloud)

# self-spread of each cloud (leave-one-out 1-NN, same standardization) -> noise fattening
def loo_p(tree, cloud):
    d, _ = tree.query(cloud, k=2)  # nearest is self(0); take 2nd
    return np.percentile(d[:, 1], [50, 95, 99])
print(f"clean cloud self-NN p50/95/99 = {loo_p(tc, clean_cloud)}")
print(f"dart  cloud self-NN p50/95/99 = {loo_p(td, dart_cloud)}   (larger spread = DART fills more space)")

# test states
test = stack("data/dart_test_huge_full2ins.hdf5", stride=2)
ts = (test - mu) / sig
d_clean = tc.query(ts)[0]
d_dart = td.query(ts)[0]
print(f"\n{len(ts)} test states.")
print(f"{'d_clean bin':14} {'N':>7} {'median d_dart':>13} {'p90 d_dart':>11} {'%(d_dart<1.79)':>15}")
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]:
    m = (d_clean >= lo) & (d_clean < hi); n = int(m.sum())
    if n < 10:
        print(f"[{lo},{hi})       {n:>7}  -- too few"); continue
    md = np.median(d_dart[m]); p90 = np.percentile(d_dart[m], 90)
    frac = 100 * np.mean(d_dart[m] < 1.79)
    print(f"[{lo},{hi})       {n:>7} {md:>13.2f} {p90:>11.2f} {frac:>14.0f}%")
print("\ninterpretation: if off-clean states (d_clean>2) have SMALL d_dart, the DART")
print("manifold covers exactly the region that is off the clean manifold.")
