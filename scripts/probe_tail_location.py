"""WHERE in the task do the gradient-tail samples live? For each of N dataset samples:
(a) tail score = |label - obs-kNN label mean| (the state-unpredictable component),
(b) task window from the obs: NEAR-GATE (held frame, lat<80mm), CARRY (held, lat>=80mm),
    REACH (not held). If the tail decile is enriched in NEAR-GATE, the noise sits exactly
    on the servo-critical windows — the data-side reason L2 loses the servo specifically."""
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs = []
rows = []  # (obs53, act7, window)
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    for t in range(0, T - 1, 3):
        held = c1 + 5 <= t < r1
        rows.append((ov[t], a[t], t, held))
h.close()
off = np.median(np.stack(offs), 0)
X = np.stack([r[0] for r in rows]); A = np.stack([r[1] for r in rows])
wins = []
for (s53, act, t, held) in rows:
    if not held: wins.append("REACH")
    else:
        v = s53[FP] - (s53[BP] + off)
        wins.append("NEARGATE" if np.linalg.norm(v[:2]) < 0.080 else "CARRY")
wins = np.array(wins)
# tail score via obs-kNN label deviation (torch cdist)
import torch
FX = torch.tensor(X); D = torch.cdist(FX, FX)
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
knn = A[nbr].mean(1)
tail = np.linalg.norm(A - knn, axis=1)
q90 = np.quantile(tail, 0.9)
print(f"N={len(rows)} | window shares: " + " ".join(f"{w}={np.mean(wins==w):.2f}" for w in ["REACH","CARRY","NEARGATE"]))
for w in ["REACH", "CARRY", "NEARGATE"]:
    m = wins == w
    enr = np.mean(tail[m] >= q90) / 0.10
    print(f"TAILLOC {w}: tail p50={np.median(tail[m]):.3f} p90={np.quantile(tail[m],0.9):.3f} | tail-decile enrichment x{enr:.2f}", flush=True)
# rotation-channel share of the tail within neargate
rotdev = np.linalg.norm((A - knn)[:, 3:6], axis=1); posdev = np.linalg.norm((A - knn)[:, 0:3], axis=1)
m = wins == "NEARGATE"
print(f"TAILLOC NEARGATE rot-vs-pos deviation: rot p50={np.median(rotdev[m]):.3f} pos p50={np.median(posdev[m]):.3f}", flush=True)
print("TAILLOC-DONE")
