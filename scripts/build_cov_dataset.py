"""Targeted-coverage substitution at fixed demo budget (user-designed):
from the 20k pool, greedily select M demos that best cover the SPARSE
regions of MP-200 (largest cross-episode r_nn windows); drop the M most
REDUNDANT original demos (densest, protecting the first 40 = eval
anchors); write a 200-demo dataset mp200cov.

Score of a candidate demo = number of still-uncovered sparse targets t
with min-dist(demo, t) < 0.6 * r_nn(t) (it supplies a neighbor
substantially closer than the best existing one). Greedy with coverage
marking. Prints COV lines. Env: CV_M (swap count, default 60), CV_TOPQ
(sparse-target quantile, default 0.30), CV_CAND (candidate demos, 2800).
"""
import os
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
from scipy.spatial import cKDTree

M = int(os.environ.get("CV_M", "60"))
TOPQ = float(os.environ.get("CV_TOPQ", "0.30"))
NCAND = int(os.environ.get("CV_CAND", "2800"))
SRC = "data/tool_hang_full2ins_mp_200.hdf5"
POOL = "data/tool_hang_full2ins_mp_20k.hdf5"
OUT = "data/tool_hang_full2ins_mp200cov.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def windows(h, dn):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    if len(S) < 12:
        return None
    return np.stack([np.stack([S[i - 1], S[i]]).reshape(-1)
                     for i in range(1, len(S) - 9, 2)])


hs = h5py.File(SRC, "r")
names = sorted(hs["data"].keys(), key=lambda d: int(d.split("_")[1]))
W, EID = [], []
for ti, dn in enumerate(names):
    w = windows(hs, dn)
    W.append(w)
    EID.append(np.full(len(w), ti))
Wa, Ea = np.concatenate(W), np.concatenate(EID)
mu, sd = Wa.mean(0), Wa.std(0) + 1e-6
Wn = (Wa - mu) / sd
tree = cKDTree(Wn)
d, nb = tree.query(Wn, k=24)
rnn = np.full(len(Wn), np.inf)
for i in range(len(Wn)):
    m = Ea[nb[i]] != Ea[i]
    if m.any():
        rnn[i] = d[i][m][0]
thr = np.quantile(rnn[np.isfinite(rnn)], 1 - TOPQ)
targets = np.where(rnn >= thr)[0]
print(f"COV base windows {len(Wn)}, sparse targets {len(targets)} "
      f"(r_nn >= {thr:.2f})", flush=True)

# per-demo redundancy (mean r_nn of its windows), protect demos 0..39
dem_r = np.array([rnn[Ea == ti][np.isfinite(rnn[Ea == ti])].mean()
                  for ti in range(len(names))])
droppable = np.argsort(dem_r)  # densest (most redundant) first
drop = [ti for ti in droppable if ti >= 40][:M]
print(f"COV dropping {len(drop)} densest demos (mean r_nn "
      f"{dem_r[drop].mean():.2f} vs kept {np.delete(dem_r, drop).mean():.2f})",
      flush=True)

# candidates from the pool (skip first 200 = the originals)
hp = h5py.File(POOL, "r")
pnames = sorted(hp["data"].keys(), key=lambda d: int(d.split("_")[1]))
cand_names = pnames[200:200 + NCAND]
Tn = Wn[targets]
ttree = cKDTree(Tn)
tr = rnn[targets]
cover_lists = []
for ci, dn in enumerate(cand_names):
    w = windows(hp, dn)
    if w is None:
        cover_lists.append(set())
        continue
    wn = (w - mu) / sd
    dd, tt = ttree.query(wn, k=8)
    cov = set()
    for j in range(len(wn)):
        for k in range(8):
            if dd[j][k] < 0.6 * tr[tt[j][k]]:
                cov.add(int(tt[j][k]))
    cover_lists.append(cov)
    if ci % 500 == 0:
        print(f"COV scanned {ci}/{len(cand_names)}", flush=True)

selected, covered = [], set()
for _ in range(M):
    gains = [len(c - covered) for c in cover_lists]
    b = int(np.argmax(gains))
    if gains[b] == 0:
        break
    selected.append(b)
    covered |= cover_lists[b]
    cover_lists[b] = set()
print(f"COV selected {len(selected)} pool demos covering "
      f"{len(covered)}/{len(targets)} sparse targets "
      f"({len(covered)/len(targets)*100:.0f}%)", flush=True)

# write output: originals (minus dropped, anchors first) + selected
keep = [dn for ti, dn in enumerate(names) if ti not in set(drop)]
with h5py.File(OUT, "w") as ho:
    g = ho.create_group("data")
    for k, v in hs["data"].attrs.items():
        g.attrs[k] = v
    for k, v in hs.attrs.items():
        ho.attrs[k] = v
    n = 0
    for dn in keep:
        hs.copy(f"data/{dn}", g, name=f"demo_{n}")
        n += 1
    for b in selected:
        hp.copy(f"data/{cand_names[b]}", g, name=f"demo_{n}")
        n += 1
print(f"COV wrote {OUT}: {len(keep)} kept + {len(selected)} pool = "
      f"{len(keep) + len(selected)} demos", flush=True)
print("COV done", flush=True)
