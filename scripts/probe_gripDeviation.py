"""Is the failure manufactured at the GRASP? In-hand pose = frame position in the eef frame
(obs cols 14-16, rel = R_eef^T(frame_pos - eef_pos)) during the held pre-gate segment.
Compare distributions: demos vs each model's assembled vs failed episodes.
Readout: deviation of the episode's median in-hand pose from the DEMO median (mm), and
in-hand pose drift within the episode."""
import os
import numpy as np, h5py
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
FR = slice(14, 17)   # frame rel eef (gripper frame)
BP = slice(7, 10); FP = slice(21, 24)

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
demo_grip = []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    demo_grip.append(np.median(ov[c1+10:r1, FR], axis=0))
h.close()
demo_grip = np.stack(demo_grip)
mu = np.median(demo_grip, 0)
scat = np.linalg.norm(demo_grip - mu, axis=1)
print(f"DEMO in-hand pose: median={np.round(mu,4)} scatter p50={np.median(scat)*1000:.1f}mm p90={np.quantile(scat,0.9)*1000:.1f}mm (n={len(demo_grip)})", flush=True)

for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    dev_pass, dev_fail, drift_pass, drift_fail = [], [], [], []
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        L = min(len(o), len(a)); gc = a[:L, 6]
        run = 0; seg = []
        for t in range(L):
            run = run + 1 if gc[t] >= 0 else 0
            if run >= 15:
                seg.append(o[t, FR])
        if len(seg) > 30:
            seg = np.stack(seg)
            gp = np.median(seg, 0)
            dev = np.linalg.norm(gp - mu) * 1000
            drift = np.linalg.norm(seg[-15:].mean(0) - seg[:15].mean(0)) * 1000
            (dev_pass if int(m[1]) else dev_fail).append(dev)
            (drift_pass if int(m[1]) else drift_fail).append(drift)
        i += 1
    def s(x): return f"p50={np.median(x):.1f}mm p90={np.quantile(x,0.9):.1f}mm (n={len(x)})" if x else "none"
    print(f"GRIP {name}: PASS dev {s(dev_pass)} drift {s(drift_pass)}", flush=True)
    print(f"GRIP {name}: FAIL dev {s(dev_fail)} drift {s(drift_fail)}", flush=True)
print("GRIP-DONE")
