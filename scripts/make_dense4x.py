"""C2 construction: density imbalance. Copy f2i and append 3 approach-only
clips per demo (steps with r < -12 relative to closure), quadrupling approach
sample density without touching amplitudes or chunk structure."""
import h5py
import numpy as np

SRC = "data/tool_hang_full2ins_2000.hdf5"
DST = "data/tool_hang_f2i_dense4x.hdf5"
src = h5py.File(SRC, "r")
dst = h5py.File(DST, "w")
g = dst.create_group("data")
for k, v in src["data"].attrs.items():
    g.attrs[k] = v
n_out = 0
keys = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[-1]))
for k in keys:
    grp = src[f"data/{k}"]
    a = np.asarray(grp["actions"])
    gr = a[:, 6]
    cl = [t for t in range(1, len(gr)) if gr[t - 1] < 0 and gr[t] >= 0]
    src.copy(grp, g, name=f"demo_{n_out}"); n_out += 1
    if not cl: continue
    cut = max(cl[0] - 12, 8)
    for _ in range(3):
        ng = g.create_group(f"demo_{n_out}"); n_out += 1
        ng.attrs["num_samples"] = cut
        ng.create_dataset("actions", data=a[:cut])
        og = ng.create_group("obs")
        for ok in grp["obs"]:
            og.create_dataset(ok, data=np.asarray(grp[f"obs/{ok}"])[:cut])
        for extra in ("dones", "rewards", "states"):
            if extra in grp:
                ng.create_dataset(extra, data=np.asarray(grp[extra])[:cut])
print("demos out:", n_out)
dst.close(); src.close()
