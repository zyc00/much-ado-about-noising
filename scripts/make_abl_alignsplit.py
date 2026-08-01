"""Build the align/push decomposition arms:
  ABL_g_t_align: (0, t_push)              -- grasp + transit + align hover, NO insertion push
  ABL_g_t_push : (0, c1+70) u (t_push, T) -- grasp + transit + push only, align hover removed
t_push = last step at the frame-z hover plateau (fz >= max(fz[c1+60:]) - 5mm)."""
import h5py, numpy as np

src = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[-1]))

def write_ds(path, plan):
    out = h5py.File(path, "w")
    grp = out.create_group("data")
    for a in src["data"].attrs: grp.attrs[a] = src["data"].attrs[a]
    total = 0; di = 0
    for k in keys:
        d = src[f"data/{k}"]
        a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
        cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
        if not cl: continue
        c1 = cl[0]; T = len(g)
        fz = np.asarray(d["obs/object"])[:, 23]
        if c1 + 70 >= T: continue
        peak = fz[c1+60:].max()
        plateau = np.where(fz >= peak - 0.005)[0]
        plateau = plateau[plateau >= c1 + 60]
        if not len(plateau): continue
        t_push = int(plateau[-1])
        for (t0, t1) in plan(c1, t_push, T):
            t1 = min(t1, T)
            if t1 - t0 < 20: continue
            nd = grp.create_group(f"demo_{di}"); di += 1
            for name in ["actions", "dones", "rewards", "states"]:
                nd.create_dataset(name, data=np.asarray(d[name])[t0:t1])
            og = nd.create_group("obs")
            for ok in d["obs"]:
                og.create_dataset(ok, data=np.asarray(d["obs"][ok])[t0:t1])
            nd.attrs["num_samples"] = t1 - t0
            total += t1 - t0
    grp.attrs["total"] = total
    mg = out.create_group("mask")
    mg.create_dataset("train", data=np.array([f"demo_{i}".encode() for i in range(di)]))
    out.close()
    print(path, "demos:", di, "total:", total)

write_ds("data/tool_hang_ABL_g_t_align.hdf5", lambda c1, tp, T: [(0, tp)])
write_ds("data/tool_hang_ABL_g_t_push.hdf5",  lambda c1, tp, T: [(0, c1 + 70), (tp, T)])
