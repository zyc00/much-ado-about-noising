import numpy as np

z = np.load("analysis/manifold/loss_traj.npz")
for arm in ["L2", "HT", "MIP"]:
    for snap in ["60000", "300000"]:
        rs, ds = [], []
        for k in z.files:
            if k.startswith(f"{arm}{snap}_") and k.endswith("_r"):
                r = z[k]
                L = len(r)
                dec = np.minimum((10 * np.arange(L) / L).astype(int), 9)
                rs.append(r)
                ds.append(dec)
        r = np.concatenate(rs)
        dec = np.concatenate(ds)
        pk = np.isin(dec, [3, 4, 8, 9])
        print(f"C60 {arm}@{snap} pocket p50 {np.median(r[pk]):.2e} "
              f"nonpocket {np.median(r[~pk]):.2e} ratio "
              f"{np.median(r[pk]) / np.median(r[~pk]):.2f} | pocket "
              f"loss-share {(r[pk].sum()) / r.sum():.2f}", flush=True)
print("C60 done", flush=True)
