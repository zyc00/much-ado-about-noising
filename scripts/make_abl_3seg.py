"""Build ablation datasets: B = grasp+transit (0->c1+70); C = grasp(0->c1+16) + insert(c1+70->end) as separate demos."""
import h5py, numpy as np, json
src = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[-1]))
def write_ds(path, plan):
    """plan(demo) -> list of (t0, t1) sub-demos"""
    out = h5py.File(path, "w")
    grp = out.create_group("data")
    for a in src["data"].attrs: grp.attrs[a] = src["data"].attrs[a]
    total = 0; di = 0
    for k in keys:
        d = src[f"data/{k}"]
        a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
        cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
        if not cl: continue
        for (t0, t1) in plan(cl[0], len(g)):
            t1 = min(t1, len(g))
            if t1 - t0 < 20: continue
            nd = grp.create_group(f"demo_{di}"); di += 1
            nd.create_dataset("actions", data=np.asarray(d["actions"])[t0:t1])
            nd.create_dataset("dones", data=np.asarray(d["dones"])[t0:t1])
            nd.create_dataset("rewards", data=np.asarray(d["rewards"])[t0:t1])
            nd.create_dataset("states", data=np.asarray(d["states"])[t0:t1])
            og = nd.create_group("obs")
            for ok in d["obs"]:
                og.create_dataset(ok, data=np.asarray(d["obs"][ok])[t0:t1])
            nd.attrs["num_samples"] = t1 - t0
            total += t1 - t0
    grp.attrs["total"] = total
    if "mask" in src:
        mg = out.create_group("mask")
        names = [f"demo_{i}".encode() for i in range(di)]
        mg.create_dataset("train", data=np.array(names))
    out.close()
    print(path, "demos:", di, "total:", total)
write_ds("data/tool_hang_ABL_3seg.hdf5", lambda c1, T: [(0, c1 + 16), (c1 + 24, c1 + 62), (c1 + 70, T)])
src.close()
