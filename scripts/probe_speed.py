"""Per-phase speed profile of MP-200: eef translational speed (mm/step) and
rotational speed (rad/step) per progress bin, p50 over 20 demos."""
import h5py
import numpy as np

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[:20]
NB = 12
tv = [[] for _ in range(NB)]
rv = [[] for _ in range(NB)]
for dn in names:
    o = h[f"data/{dn}/obs"]
    E = np.asarray(o["robot0_eef_pos"])
    Q = np.asarray(o["robot0_eef_quat"])
    L = len(E)
    for i in range(1, L - 1):
        b = min(int(NB * i / L), NB - 1)
        tv[b].append(1000 * np.linalg.norm(E[i + 1] - E[i]))
        d = abs(float(np.dot(Q[i + 1], Q[i])))
        rv[b].append(2 * np.arccos(min(1.0, d)))
h.close()
PH = ["reach", "reach", "pick", "pick", "lift", "transit", "transit",
      "rotate", "rotate", "align", "insert", "settle"]
print("SP bin phase   trans(mm/step)p50  rot(rad/step)p50")
for b in range(NB):
    print(f"SP {b:2d} {PH[b]:8s} {np.median(tv[b]):6.1f} "
          f"{np.median(rv[b]):.4f}")
print("SP done")
