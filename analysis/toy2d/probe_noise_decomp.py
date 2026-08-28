"""Per-type aleatoric decomposition of tool-hang human demo actions.
Types: T1 speed (tangential jitter + pauses), T2 direction (transverse
jitter), T3 flip (gripper toggle-timing dispersion). Pure h5py/numpy.
Smoothing defines the 'intended' signal; residual = noise proxy."""
import h5py
import numpy as np

HUM = ("/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/"
       "blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4")
K = 5  # smoothing half-window


def smooth(x, k=K):
    out = np.copy(x).astype(float)
    for i in range(len(x)):
        lo, hi = max(0, i - k), min(len(x), i + k + 1)
        out[i] = x[lo:hi].mean(axis=0)
    return out


def kurt(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    s = x.std() + 1e-12
    return float((x ** 4).mean() / s ** 4 - 3.0)


f = h5py.File(HUM, "r")
demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
print(f"{len(demos)} demos")

seg_stats = {s: {"tan": [], "trs": [], "pause": 0, "n": 0}
             for s in ("slow", "fast")}
flip_states, grip_frac = [], []
tan_all, trs_all = [], []

for d in demos:
    g = f["data"][d]
    act = g["actions"][:]           # (T, 7) delta: dpos3, drot3, grip
    eefz = g["obs"]["robot0_eef_pos"][:, 2]
    dp = act[:, :3]
    sm = smooth(dp)
    res = dp - sm                   # high-frequency label residual
    smn = np.linalg.norm(sm, axis=1) + 1e-9
    u = sm / smn[:, None]
    tan = (res * u).sum(1)          # pace jitter (along intended)
    trs = np.linalg.norm(res - tan[:, None] * u, axis=1)  # direction jitter
    speed = np.linalg.norm(dp, axis=1)
    thr = np.percentile(speed, 40)
    slow = speed < thr
    for m, name in ((slow, "slow"), (~slow, "fast")):
        seg_stats[name]["tan"].extend(tan[m].tolist())
        seg_stats[name]["trs"].extend(trs[m].tolist())
        seg_stats[name]["pause"] += int((speed[m] < 0.1 * np.median(speed)).sum())
        seg_stats[name]["n"] += int(m.sum())
    tan_all.extend(tan.tolist())
    trs_all.extend(trs.tolist())
    # T3: gripper toggles
    gr = act[:, 6]
    tog = np.where(np.abs(np.diff(np.sign(gr))) > 0)[0]
    for t in tog:
        flip_states.append(eefz[t])
    grip_frac.append(float(np.var(gr) / (np.var(act) * act.shape[1] + 1e-12)))

tan_all, trs_all = np.array(tan_all), np.array(trs_all)
e_tan, e_trs = float((tan_all ** 2).mean()), float((trs_all ** 2).mean())
tot = e_tan + e_trs
print("\n=== T1 vs T2: residual ENERGY shares (position-action labels) ===")
print(f"tangential (SPEED)   : {e_tan / tot:.2%}   kurtosis {kurt(tan_all):.1f}")
print(f"transverse (DIRECTION): {e_trs / tot:.2%}   kurtosis {kurt(trs_all):.1f}")
for name in ("slow", "fast"):
    s = seg_stats[name]
    ta, tr = np.array(s["tan"]), np.array(s["trs"])
    et, er = (ta ** 2).mean(), (tr ** 2).mean()
    print(f"[{name}] tan share {et / (et + er):.2%} (kurt {kurt(ta):.1f}) | "
          f"trs kurt {kurt(tr):.1f} | pause frac {s['pause'] / max(s['n'], 1):.2%}")

fs = np.array(flip_states)
print("\n=== T3: gripper flip timing ===")
print(f"toggles/demo: {len(fs) / len(demos):.2f}")
print(f"flip-height dispersion (sd across demos): {fs.std():.4f} m "
      f"(range {fs.min():.3f}-{fs.max():.3f})")
print(f"gripper share of total action variance: {np.mean(grip_frac):.2%}")
