"""Fine-grained bias/std vs distance (esp. the 2-4 region split into 0.25-width bins).
Reports per bin: N, |mu_MSE|, |mu_MIP|, |mu_MSE-mu_MIP| (residual bias), |std|, and a
bootstrap 90% CI on |mu_MSE| to flag reliability. Uses the executed (first) action."""
import numpy as np
np.set_printoptions(precision=3, suppress=True)
E = np.load("analysis/recovery/err_vectors.npz")
dist = E["dist"]; eM = E["eMSE"][:, 0, :]; eP = E["eMIP"][:, 0, :]
edges = [0, 0.5, 1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 6, 8]
rng = np.random.RandomState(0)
print(f"total states {len(dist)}")
print(f"{'bin':12} {'N':>6} {'|mu_MSE|':>9} {'[boot90]':>13} {'|mu_MIP|':>9} {'|mu_res|':>9} {'|std_MSE|':>9} {'|std_MIP|':>9}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dist >= lo) & (dist < hi); n = int(m.sum())
    if n < 10:
        print(f"[{lo},{hi})        {n:>6}   -- too few"); continue
    XM = eM[m]; XP = eP[m]
    bM = np.linalg.norm(XM.mean(0)); bP = np.linalg.norm(XP.mean(0))
    res = np.linalg.norm(XM.mean(0) - XP.mean(0))
    sM = np.linalg.norm(XM.std(0)); sP = np.linalg.norm(XP.std(0))
    boots = [np.linalg.norm(XM[rng.randint(0, n, n)].mean(0)) for _ in range(300)]
    lo_b, hi_b = np.percentile(boots, [5, 95])
    flag = "" if (hi_b - lo_b) < 0.3 * max(bM, 1e-6) or n > 300 else " (shaky)"
    print(f"[{lo},{hi})        {n:>6} {bM:>9.3f} [{lo_b:>5.3f},{hi_b:>5.3f}] {bP:>9.3f} {res:>9.3f} {sM:>9.3f} {sP:>9.3f}{flag}")
