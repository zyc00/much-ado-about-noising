"""Estimate per-distance-bin ACTION-CHUNK error covariance (AS*6 = 48-dim) and check
reliability. Effective N = #states (one 48-d chunk-error per state; chunk actions are
NOT independent so we keep them as one vector, not pooled). 48x48 needs N >> 48.
Reliability: N vs dim, split-half Frobenius + top-eigvec angle, bootstrap eig CI, rank."""
import numpy as np
np.set_printoptions(precision=3, suppress=True)
E = np.load("analysis/recovery/err_vectors.npz")
dist = E["dist"]
D = E["eMSE"].shape[1] * E["eMSE"].shape[2]   # AS*6
eMSE = E["eMSE"].reshape(len(dist), -1)        # (N, 48)
eMIP = E["eMIP"].reshape(len(dist), -1)
BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
rng = np.random.RandomState(0)


def topvec_angle(A, B):
    _, va = np.linalg.eigh(A); _, vb = np.linalg.eigh(B)
    return np.degrees(np.arccos(np.clip(abs(np.dot(va[:, -1], vb[:, -1])), 0, 1)))


def report(name, e):
    print(f"\n### {name}  (chunk-error dim = {D})")
    for b in BINS:
        m = (dist >= b[0]) & (dist < b[1]); X = e[m]; n = len(X)
        if n < 20:
            print(f"  d[{b[0]},{b[1]}): N_states={n:4d}  -- skip"); continue
        S = np.cov(X, rowvar=False)
        w = np.linalg.eigvalsh(S)
        rank = int((w > 1e-6 * w[-1]).sum())
        idx = rng.permutation(n); h = n // 2
        SA = np.cov(X[idx[:h]], rowvar=False); SB = np.cov(X[idx[h:]], rowvar=False)
        fro = np.linalg.norm(SA - SB) / max(np.linalg.norm(S), 1e-9)
        ang = topvec_angle(SA, SB)
        tops = [np.linalg.eigvalsh(np.cov(X[rng.randint(0, n, n)], rowvar=False))[-1] for _ in range(150)]
        lo, hi = np.percentile(tops, [5, 95]); rel = (hi - lo) / max(w[-1], 1e-9)
        status = "OK" if (n > 3 * D and fro < 0.3 and ang < 25 and rel < 0.5) else ("LOW-N" if n < 2 * D else "SHAKY")
        print(f"  d[{b[0]},{b[1]}): N={n:4d} (need>~{3*D}) rank={rank}/{D}  splithalf ΔF={fro:.2f} top∠={ang:4.1f}°  "
              f"boot-rel={rel:.2f}  -> {status}")


report("MSE chunk-error cov", eMSE)
report("MIP chunk-error cov", eMIP)
print(f"\nNeed N_states >~ {3*D} per bin for a usable {D}x{D} cov (rank-full needs N>{D}).")
print("Reliable if: N>3*dim, split-half ΔF<0.3, top-eigvec angle<25deg, bootstrap rel<0.5.")
