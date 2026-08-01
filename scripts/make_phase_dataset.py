"""Build the oracle-phase dataset: full2ins_2000 with a 3-dim phase one-hot appended
to robot0_gripper_qpos (2->5 dims, obs_dim 53->56) so all existing dim indices stay valid.
Phases per demo (c1 = first gripper sign crossing in actions):
  [1,0,0] grasp   t <  c1+16
  [0,1,0] transit c1+16 <= t < c1+70
  [0,0,1] insert  t >= c1+70
"""
import h5py, numpy as np

SRC = "data/tool_hang_full2ins_2000.hdf5"
DST = "data/tool_hang_full2ins_2000_phase.hdf5"
src = h5py.File(SRC, "r")
keys = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[-1]))
out = h5py.File(DST, "w")
grp = out.create_group("data")
for a in src["data"].attrs: grp.attrs[a] = src["data"].attrs[a]
total = 0; di = 0
for k in keys:
    d = src[f"data/{k}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]; T = len(g)
    ph = np.zeros((T, 3), dtype=np.float64)
    ph[:min(c1+16, T), 0] = 1
    ph[min(c1+16, T):min(c1+70, T), 1] = 1
    ph[min(c1+70, T):, 2] = 1
    nd = grp.create_group(f"demo_{di}"); di += 1
    for name in ["actions", "dones", "rewards", "states"]:
        nd.create_dataset(name, data=np.asarray(d[name]))
    og = nd.create_group("obs")
    for ok in d["obs"]:
        arr = np.asarray(d["obs"][ok])
        if ok == "robot0_gripper_qpos":
            arr = np.concatenate([arr, ph], axis=1)
        og.create_dataset(ok, data=arr)
    nd.attrs["num_samples"] = T
    total += T
grp.attrs["total"] = total
if "mask" in src:
    mg = out.create_group("mask")
    mg.create_dataset("train", data=np.array([f"demo_{i}".encode() for i in range(di)]))
out.close()
print(DST, "demos:", di, "total:", total)
