"""Dump demo EEF position trajectories (reference tube) for the MP datasets
so local trajectory analysis can measure lateral deviation vs height."""
import h5py
import numpy as np

OUT = "/mnt/pfs/yuchen/demo_eef.npz"
out = {}
for name, f in [("d200", "data/tool_hang_full2ins_mp_200.hdf5"),
                ("d2k", "data/tool_hang_full2ins_mp_2000.hdf5")]:
    h = h5py.File(f, "r")
    demos = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[:300]
    P = [np.asarray(h[f"data/{d}/obs/robot0_eef_pos"]) for d in demos]
    A = [np.asarray(h[f"data/{d}/actions"]) for d in demos]
    out[name] = np.concatenate(P).astype(np.float32)
    out[name + "_len"] = np.array([len(p) for p in P])
    out[name + "_act"] = np.concatenate(A).astype(np.float32)
    h.close()
    print(f"DEMOEEF {name} demos={len(P)} steps={len(out[name])}", flush=True)
np.savez_compressed(OUT, **out)
print("DEMOEEF-DONE", flush=True)
