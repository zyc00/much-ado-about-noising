"""Chunking-as-target-disambiguation probe (data-level, no model).
For H in {1,2,4,8,16,32}: does the H-step chunk label space (a) separate phases
("sparser" cross-phase action distribution), (b) make quiet-window targets
injective in time-to-transition? Metrics per H:
- quiet-query (r in [-10,10)) label-NN: fraction with |dr|<=10 (phase-matched),
  median |dr| of the NN, fraction class-matched (within bands)
- class-pair median per-step-normalized label distance (SET-SHO/SET-MID/...)
- Spearman rho(|r_i-r_j|, label dist) among quiet pairs (progress coding)
Distances are L2 / sqrt(H) (per-step RMS) so magnitudes compare across H.
Raw clipped actions (global affine normalizer irrelevant for NN structure)."""
import sys

import h5py
import numpy as np
import torch

DSP = "data/tool_hang_full2ins_2000.hdf5"
CLS = ["APP", "SET", "SHO", "MID", "PST"]
BANDS = {"APP": (-40, -16), "SET": (-10, 0), "SHO": (2, 10), "MID": (14, 50), "PST": (72, 120)}
def cls_of(r):
    for c, (lo, hi) in BANDS.items():
        if lo <= r < hi:
            return c
    return None

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:150]
A, R, D = [], [], []
for di, k in enumerate(keys):
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t - 1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    A.append(a); R.append(np.arange(len(a)) - c1); D.append(np.full(len(a), di))
h.close()

rng = np.random.RandomState(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("H | quiet-NN phase-match(|dr|<=10) | NN |dr| p50/p90 | NN class-match | "
      "rho(dr,dist) quiet | pair p50: SET-SHO SET-MID SHO-MID SET-PST", flush=True)
for H in [1, 2, 4, 8, 16, 32]:
    ch, rr, dd = [], [], []
    for a, r, d in zip(A, R, D):
        T = len(a)
        for t in range(0, T, 2):
            c = a[t:t + H]
            if len(c) < H:
                c = np.pad(c, ((0, H - len(c)), (0, 0)), "edge")
            ch.append(c.reshape(-1)); rr.append(r[t]); dd.append(d[t])
    ch = np.stack(ch); rr = np.array(rr); dd = np.array(dd)
    if len(ch) > 8000:
        sel = rng.choice(len(ch), 8000, replace=False)
        ch, rr, dd = ch[sel], rr[sel], dd[sel]
    V = torch.tensor(ch, device=dev) / np.sqrt(H)
    Dm = torch.cdist(V, V)
    Dm[torch.tensor(dd[:, None] == dd[None, :], device=dev)] = 1e9
    q = np.where((rr >= -10) & (rr < 10))[0]
    nn = Dm[q].argmin(dim=1).cpu().numpy()
    dr = np.abs(rr[nn] - rr[q])
    pm = float((dr <= 10).mean())
    qc = np.array([cls_of(r_) for r_ in rr[q]])
    nc = np.array([cls_of(r_) for r_ in rr[nn]])
    both = (qc != None) & (nc != None)  # noqa: E711
    cm = float((qc[both] == nc[both]).mean())
    # class-pair medians
    pmids = {}
    for c1_, c2_ in [("SET", "SHO"), ("SET", "MID"), ("SHO", "MID"), ("SET", "PST")]:
        i1 = np.where([cls_of(r_) == c1_ for r_ in rr])[0]
        i2 = np.where([cls_of(r_) == c2_ for r_ in rr])[0]
        i1 = i1[rng.permutation(len(i1))[:300]]; i2 = i2[rng.permutation(len(i2))[:300]]
        pmids[(c1_, c2_)] = float(torch.cdist(V[i1], V[i2]).median())
    # progress coding among quiet pairs
    qq = q[rng.permutation(len(q))[:600]]
    Dq = Dm[qq][:, qq].cpu().numpy()
    drq = np.abs(rr[qq][:, None] - rr[qq][None, :])
    iu = np.triu_indices(len(qq), 1)
    x, y = drq[iu], Dq[iu]
    m = y < 1e8
    rx = np.argsort(np.argsort(x[m])).astype(np.float64)
    ry = np.argsort(np.argsort(y[m])).astype(np.float64)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    print(f"H={H:2d} | {pm:.0%} | {np.median(dr):.0f}/{np.percentile(dr, 90):.0f} | {cm:.0%} | "
          f"{rho:.2f} | {pmids[('SET','SHO')]:.3f} {pmids[('SET','MID')]:.3f} "
          f"{pmids[('SHO','MID')]:.3f} {pmids[('SET','PST')]:.3f}", flush=True)
print("CHUNKDUAL-DONE")
