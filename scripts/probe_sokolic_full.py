"""Sokolic Corollary 1 applied FAITHFULLY, with all three terms.

  GE <= sqrt( log2 * N_y * 2^{k+1} * C_M^k / (gamma^k * m) )

k   : intrinsic dimension of the data manifold (TwoNN estimator on the demo
      windows) -- never measured before; gamma enters as gamma^{-k}, so the
      Jacobian's effect is EXPONENTIALLY amplified by k.
gamma: robustness radius = (task action tolerance) / ||J||_2, per arm.
K   : the covering number of the region the algorithm must be robust over.
      In closed loop this is NOT the demo manifold but the region the policy
      ITSELF VISITS -- which is arm-dependent and spans orders of magnitude
      here (MIP-2k never leaves d<3; L2-200 orbits to d>10^3). Estimated by
      greedy covering of each arm's captured rollout states at radius rho.

Prints SOKF lines. Env: SF_ARMS, SF_TAGS, SF_TOL.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
from scipy.spatial import cKDTree

TAGS = {"L2-200": "l2mp200v2", "MIP-200": "mipmp200",
        "HT-200": "htmp200", "HG-200": "hgmp200",
        "L2-2k": "l2mp2k", "MIP-2k": "mipmp2k"}
SPEC = {"L2-200": 3.7616, "MIP-200": 2.2246, "HT-200": 1.8127,
        "HG-200": 2.9665, "L2-2k": 5.3870, "MIP-2k": 4.5545}
NWIN = {"L2-200": 7431, "MIP-200": 7431, "HT-200": 7431, "HG-200": 7431,
        "L2-2k": 74111, "MIP-2k": 74111}
SR = {"L2-200": 63, "MIP-200": 94, "HT-200": 75, "HG-200": 87,
      "L2-2k": 94, "MIP-2k": 100}
TOL = float(os.environ.get("SF_TOL", "0.04"))   # ~2mm / 50mm action scale


def twonn(X, frac=0.9):
    """TwoNN intrinsic-dimension estimator (Facco et al.)."""
    t = cKDTree(X)
    d, _ = t.query(X, k=3)
    r1, r2 = d[:, 1], d[:, 2]
    m = (r1 > 1e-12)
    mu = np.sort(r2[m] / r1[m])
    n = int(len(mu) * frac)
    mu = mu[:n]
    F = np.arange(1, n + 1) / len(mu)
    x, y = np.log(mu), -np.log(1 - F * (n / (n + 1)))
    return float((x @ y) / (x @ x))


def greedy_cover(X, rho, cap=4000):
    """number of radius-rho balls needed to cover X (greedy upper bound)"""
    if len(X) > cap:
        X = X[np.random.RandomState(0).choice(len(X), cap, replace=False)]
    t = cKDTree(X)
    uncovered = np.ones(len(X), bool)
    n = 0
    while uncovered.any():
        i = int(np.argmax(uncovered))
        n += 1
        for j in t.query_ball_point(X[i], rho):
            uncovered[j] = False
    return n


FV = "analysis/failvids/"
# --- k of the demo manifold (from the on-support rollout states, which lie
#     on the demonstration manifold) ---------------------------------------
z0 = np.load(FV + "mipmp2k_trajs.npz")
seeds0 = sorted({int(k[1:]) for k in z0.files if k.startswith("W")})
W0 = np.concatenate([z0[f"W{s}"] for s in seeds0])
D0 = np.concatenate([z0[f"D{s}"] for s in seeds0])
on0 = W0[D0 < 2].reshape((D0 < 2).sum(), -1)
rs = np.random.RandomState(0)
on0 = on0[rs.choice(len(on0), min(3000, len(on0)), replace=False)]
on0 = (on0 - on0.mean(0)) / (on0.std(0) + 1e-6)
k_hat = twonn(on0)
print(f"SOKF intrinsic_dim k_hat {k_hat:.2f} (n={len(on0)})", flush=True)

print(f"SOKF using tolerance {TOL} action units (~{TOL*50:.1f} mm)", flush=True)
rows = []
for name, spec in SPEC.items():
    gamma = TOL / spec
    m = NWIN[name]
    # bound (dropping constants common to all arms): gamma^{-k} / m, in logs
    log_bound = 0.5 * (-k_hat * np.log(gamma) - np.log(m))
    row = dict(name=name, spec=spec, gamma=gamma, logB=log_bound, SR=SR[name])
    if name in TAGS:
        z = np.load(FV + TAGS[name] + "_trajs.npz")
        sd = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
        W = np.concatenate([z[f"W{s}"] for s in sd])
        W = W.reshape(len(W), -1)
        W = (W - on0.mean(0) * 0) / 1.0
        Wn = (W - W.mean(0)) / (W.std(0) + 1e-6)
        row["cover"] = greedy_cover(Wn, rho=3.0)
        row["cover6"] = greedy_cover(Wn, rho=6.0)
        # composite bound: log N(visited; gamma) = k*(log R - log gamma),
        # with R ~ rho * N(rho)^(1/k)
        R = 6.0 * row["cover6"] ** (1.0 / k_hat)
        row["logC"] = 0.5 * (k_hat * (np.log(R) - np.log(gamma)) - np.log(m))
        row["R"] = R
    rows.append(row)
for r in sorted(rows, key=lambda r: r["logB"]):
    c = (f"cov3 {r.get('cover','-')} cov6 {r.get('cover6','-')} "
         f"R {r.get('R', float('nan')):.2f} logCOMPOSITE "
         f"{r.get('logC', float('nan')):+.2f}")
    print(f"SOKF {r['name']:8} ||J||2 {r['spec']:.2f} gamma {r['gamma']:.5f} "
          f"logJonly {r['logB']:+.2f} | {c} | SR {r['SR']}", flush=True)
print("SOKF done", flush=True)
