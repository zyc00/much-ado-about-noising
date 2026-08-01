"""#3a: Did 20k CLEAN data expand the support into the off-clean (recovery) region,
or just densify the same manifold? Build 2k-clean and 20k-clean clouds with a COMMON
standardization (2k mu/sig) and MATCHED point counts (so 1-NN distance isn't confounded
by density). For the held-out test states, compare d_2k vs d_20k. If off-support states
(d_2k>2) still have large d_20k, then 10x clean data did NOT cover the recovery region
-> 20k's high SR cannot come from support expansion. Pure obs geometry; no env/model."""
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


c2_full = stack("data/tool_hang_full2ins_2000.hdf5", ndemo=200)
mu, sig = c2_full.mean(0), c2_full.std(0) + 1e-6
c20_full = stack("data/tool_hang_full2ins_20000.hdf5", ndemo=2000)
N = min(len(c2_full), len(c20_full))
rng = np.random.RandomState(0)
c2 = (c2_full[rng.choice(len(c2_full), N, replace=False)] - mu) / sig
c20 = (c20_full[rng.choice(len(c20_full), N, replace=False)] - mu) / sig
print(f"matched clouds: 2k={N} pts, 20k={N} pts (common 2k mu/sig)")
t2 = cKDTree(c2); t20 = cKDTree(c20)


def loo(tree, cloud):
    d, _ = tree.query(cloud, k=2); return np.percentile(d[:, 1], [50, 95, 99])
print(f"2k  self-NN p50/95/99 = {loo(t2, c2)}")
print(f"20k self-NN p50/95/99 = {loo(t20, c20)}   (if ~same, 20k did NOT extend the manifold)")

test = stack("data/dart_test_huge_full2ins.hdf5", stride=2)
ts = (test - mu) / sig
d2 = t2.query(ts)[0]; d20 = t20.query(ts)[0]
print(f"\n{len(ts)} test states.  d_2k bin -> what is d_20k (matched size)?")
print(f"{'d_2k bin':12} {'N':>7} {'median d_20k':>12} {'p90 d_20k':>10} {'%(d_20k<1.79)':>14}")
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]:
    m = (d2 >= lo) & (d2 < hi); n = int(m.sum())
    if n < 10:
        print(f"[{lo},{hi})     {n:>7}  -- too few"); continue
    print(f"[{lo},{hi})     {n:>7} {np.median(d20[m]):>12.2f} {np.percentile(d20[m],90):>10.2f} {100*np.mean(d20[m]<1.79):>13.0f}%")
print("\nif median d_20k ~ d_2k bin range (i.e. off-support stays off), 20k did NOT cover")
print("the recovery region -> its SR gain is NOT support expansion (compare to DART table).")
