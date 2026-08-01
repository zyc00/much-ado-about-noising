"""Cross-phase action-collision probability vs chunk horizon H (data-level).
User's quantity: P(||chunk_i - chunk_j|| < eps | phase_i != phase_j) — the mass
of similar-action pairs across phases (not NN). Distances per-step-normalized
(L2/sqrt(H)). Also: quiet-step subset (|a_pos|<0.05, |dr|>10), the ambiguity
share P(cross-phase | similar), and the per-class-pair collision matrix.
Bands: APP[-40,-16) SET[-10,0) SHO[2,10) MID[14,50) PST[72,120)."""
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
EPS = (0.1, 0.3)
print("collision = P(per-step-normalized chunk dist < eps); bands-pairs = both in a class band, class differs;"
      " quiet = |a_pos|<0.05 & |dr|>10; amb = P(cross-band | d<eps)", flush=True)
for H in [1, 2, 4, 8, 16, 32]:
    ch, rr, dd, qmask = [], [], [], []
    for a, r, d in zip(A, R, D):
        T = len(a)
        for t in range(0, T, 2):
            c = a[t:t + H]
            if len(c) < H:
                c = np.pad(c, ((0, H - len(c)), (0, 0)), "edge")
            ch.append(c.reshape(-1)); rr.append(r[t]); dd.append(d[t])
            qmask.append(np.linalg.norm(a[t, :3]) < 0.05)
    ch = np.stack(ch); rr = np.array(rr); dd = np.array(dd); qmask = np.array(qmask)
    if len(ch) > 8000:
        sel = rng.choice(len(ch), 8000, replace=False)
        ch, rr, dd, qmask = ch[sel], rr[sel], dd[sel], qmask[sel]
    cls = np.array([cls_of(r_) for r_ in rr], dtype=object)
    V = torch.tensor(ch, device=dev) / np.sqrt(H)
    N = len(V)
    # pairwise in blocks; accumulate boolean counts
    inband = cls != None  # noqa: E711
    cls_id = np.array([CLS.index(c) if c is not None else -1 for c in cls])
    cid = torch.tensor(cls_id, device=dev)
    rt = torch.tensor(rr, device=dev, dtype=torch.float32)
    dt = torch.tensor(dd, device=dev)
    qt = torch.tensor(qmask, device=dev)
    tot_cross = tot_same = 0
    hit_cross = {e: 0 for e in EPS}; hit_same = {e: 0 for e in EPS}
    q_tot = 0; q_hit = {e: 0 for e in EPS}
    pairhit = np.zeros((5, 5)); pairtot = np.zeros((5, 5))
    B = 1024
    for i0 in range(0, N, B):
        Dm = torch.cdist(V[i0:i0 + B], V)
        # mask self/same-demo
        sd = dt[i0:i0 + B, None] == dt[None, :]
        ci = cid[i0:i0 + B, None]; cj = cid[None, :]
        both = (ci >= 0) & (cj >= 0) & ~sd
        cross = both & (ci != cj); same = both & (ci == cj)
        tot_cross += int(cross.sum()); tot_same += int(same.sum())
        for e in EPS:
            close = Dm < e
            hit_cross[e] += int((cross & close).sum()); hit_same[e] += int((same & close).sum())
        # quiet subset
        qq = qt[i0:i0 + B, None] & qt[None, :] & ~sd & ((rt[i0:i0 + B, None] - rt[None, :]).abs() > 10)
        q_tot += int(qq.sum())
        for e in EPS:
            q_hit[e] += int((qq & (Dm < e)).sum())
        # class-pair collision at eps=0.3
        close3 = (Dm < 0.3)
        for a_ in range(5):
            for b_ in range(5):
                m = (ci == a_) & (cj == b_) & ~sd
                pairtot[a_, b_] += int(m.sum()); pairhit[a_, b_] += int((m & close3).sum())
    amb = {e: hit_cross[e] / max(1, hit_cross[e] + hit_same[e]) for e in EPS}
    print(f"H={H:2d} | cross-band P(d<.1)={hit_cross[0.1]/max(1,tot_cross):.4f} P(d<.3)={hit_cross[0.3]/max(1,tot_cross):.4f}"
          f" | same-band P(d<.3)={hit_same[0.3]/max(1,tot_same):.4f}"
          f" | quiet-crossR P(d<.1)={q_hit[0.1]/max(1,q_tot):.4f} P(d<.3)={q_hit[0.3]/max(1,q_tot):.4f}"
          f" | amb(.1)={amb[0.1]:.2f} amb(.3)={amb[0.3]:.2f}", flush=True)
    P = pairhit / np.maximum(pairtot, 1)
    top = sorted(((P[a_, b_], CLS[a_], CLS[b_]) for a_ in range(5) for b_ in range(a_ + 1, 5)), reverse=True)[:4]
    print(f"H={H:2d} | top colliding class pairs P(d<.3): " +
          " ".join(f"{a_}-{b_}={p:.4f}" for p, a_, b_ in top), flush=True)
print("CHUNKDUAL2-DONE")
