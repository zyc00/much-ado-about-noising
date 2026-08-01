"""Tool-grasp diagnosis: at the second gripper-close (c2), measure eef offset from the
tool (3D, xy, z) — demos vs each policy's lifted vs not-lifted grasp attempts."""
import numpy as np, h5py, os
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
TP = slice(35, 38); EE = slice(44, 47)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
demo_offs = []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if len(cl) < 2: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if r1 is None: continue
    c2 = next((t for t in cl if t > r1 + 5), None)
    if c2 is None: continue
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    demo_offs.append(ov[c2, EE] - ov[c2, TP])
h.close()
D = np.stack(demo_offs)
mu = np.median(D, 0)
def rep(tag, offs):
    if not offs: return
    O = np.stack(offs) - mu
    r3 = np.linalg.norm(O, axis=1) * 1000; rxy = np.linalg.norm(O[:, :2], axis=1) * 1000; rz = O[:, 2] * 1000
    print(f"TOOLGRASP {tag} (n={len(offs)}): |off3D| p50={np.median(r3):.0f}mm p90={np.quantile(r3,0.9):.0f}"
          f" | xy p50={np.median(rxy):.0f} | z p50={np.median(rz):+.0f}", flush=True)
rep("demos(vs own median)", list(D))
for name in os.environ.get("NAMES", "hheterot_s5 hheterot_s1000 hmipL286k").split():
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0; lifted_o, missed_o = [], []
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        T = min(len(o), len(a)); g = a[:T, 6]
        cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
        op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
        if len(cl) >= 2:
            c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
            c2 = next((t for t in cl if t > (r1 or 1e9) + 5), None) if r1 else None
            if c2 is not None and c2 < T - 10:
                off = o[c2, EE] - o[c2, TP]
                tz = o[c2:T, TP.start + 2]
                (lifted_o if (tz - tz[0] > 0.02).any() else missed_o).append(off)
        i += 1
    rep(f"{name}-LIFTED", lifted_o); rep(f"{name}-MISSED", missed_o)
print("TOOLGRASP-DONE")
