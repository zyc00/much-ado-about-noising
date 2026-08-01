"""Handoff-state quality analysis: compare the observation at the A->B switch
against the demonstration distribution at c1+16, per policy, split by outcome.
Groups: frame-in-hand pose (frame_to_eef pos 14-16 / quat 17-20), eef pos 44-46,
gripper qpos 51-52. Prints per-policy p50 z-norms for success vs failure episodes."""
import glob, os, sys
import numpy as np, h5py

DS = "data/tool_hang_full2ins_2000.hdf5"
GROUPS = {"fpos": [14, 15, 16], "fquat": [17, 18, 19, 20], "eef": [44, 45, 46], "grip": [51, 52]}

h = h5py.File(DS, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ref = []
for k in keys[:300]:
    d = h[f"data/{k}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    t = cl[0] + 16
    if t >= len(g): continue
    ov = np.concatenate([np.asarray(d["obs"][q][t]) for q in OK])
    ref.append(ov)
h.close()
R = np.stack(ref); mu, sd = R.mean(0), R.std(0) + 1e-8
print(f"reference n={len(R)}")

for pol in sorted(os.listdir("logs/swdump")):
    rows = {1: {g: [] for g in GROUPS}, 0: {g: [] for g in GROUPS}}
    n = {1: 0, 0: 0}
    for f in sorted(glob.glob(f"logs/swdump/{pol}/sw_*.npz")):
        z = np.load(f)
        sw = int(z["sw_step"])
        if sw < 0: continue
        w = z["obs_traj"][min(sw + 1, len(z["obs_traj"]) - 1)]
        asm = int(z["asm"]); n[asm] += 1
        for g, dims in GROUPS.items():
            zn = np.linalg.norm((w[dims] - mu[dims]) / sd[dims]) / np.sqrt(len(dims))
            rows[asm][g].append(zn)
    line = f"SW {pol}: n_succ={n[1]} n_fail={n[0]}"
    for g in GROUPS:
        s = np.median(rows[1][g]) if rows[1][g] else float("nan")
        fl = np.median(rows[0][g]) if rows[0][g] else float("nan")
        line += f" | {g} z_succ/z_fail={s:.2f}/{fl:.2f}"
    print(line, flush=True)
print("ANALYZE-DONE")
