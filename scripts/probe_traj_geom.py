"""Trajectory geometry of the captured rollouts (MSE/MIP x 200/2k).

Reference tube: demo EEF positions from the MP datasets, binned by height.
Per episode: descent profile, lateral deviation vs the demo tube at matched
height, insertion dwell, retry count, realized speed/jerk by phase, tube
distance. Aggregated by arm x outcome. Also writes a COMPACT figure npz
(downsampled P, D, O per arm) for local plotting.
Prints GEOM / GEOMEP lines.
"""
import numpy as np

FV = "analysis/failvids/"
Z_INS = 0.86      # insertion phase: below this height
Z_UP = 0.90       # retreat threshold for retry counting
ARMS = [("L2-200", "l2mp200v2"), ("MIP-200", "mipmp200"),
        ("L2-2k", "l2mp2k"), ("MIP-2k", "mipmp2k")]

d = np.load("/mnt/pfs/yuchen/demo_eef.npz")
DP = d["d200"]
zb = np.round(np.arange(0.78, 0.941, 0.01), 2)
ref = {}
for lo in zb[:-1]:
    m = (DP[:, 2] >= lo) & (DP[:, 2] < lo + 0.01)
    if m.sum() > 20:
        ref[lo] = (DP[m][:, :2].mean(0), np.linalg.norm(
            DP[m][:, :2] - DP[m][:, :2].mean(0), axis=1).mean())
for k in sorted(ref):
    print(f"REFTUBE z {k:.2f} center ({ref[k][0][0]:+.3f},{ref[k][0][1]:+.3f})"
          f" mean_radius {ref[k][1]*1000:.1f}mm", flush=True)


def lat_err(p):
    """lateral distance from the demo tube center at this height (m)"""
    k = np.round(np.floor(p[2] * 100) / 100, 2)
    if k not in ref:
        return np.nan
    return float(np.linalg.norm(p[:2] - ref[k][0]))


fig = {}
for name, tag in ARMS:
    z = np.load(FV + f"{tag}_trajs.npz")
    seeds = sorted({int(k[1:]) for k in z.files if k.startswith("P")})
    rows = []
    for sd in seeds:
        P, D = z[f"P{sd}"], z[f"D{sd}"]
        ok = bool(z[f"O{sd}"][0])
        n = min(len(P), len(D))
        P, D = P[:n], D[:n]
        ins = P[:, 2] < Z_INS
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)
        jerk = np.linalg.norm(np.diff(np.diff(P, axis=0), axis=0), axis=1)
        # retries: transitions below Z_INS -> back above Z_UP
        below = P[:, 2] < Z_INS
        above = P[:, 2] > Z_UP
        nret, armed = 0, False
        for i in range(n):
            if below[i]:
                armed = True
            elif armed and above[i]:
                nret += 1
                armed = False
        le = np.array([lat_err(p) for p in P])
        le_ins = le[ins & ~np.isnan(le)]
        izmin = int(np.argmin(P[:, 2]))
        rows.append(dict(
            sd=sd, ok=ok, n=n, zmin=float(P[:, 2].min()),
            dwell=int(ins.sum()), nret=nret,
            le_zmin=float(le[izmin]) if not np.isnan(le[izmin]) else np.nan,
            le_ins_mean=float(le_ins.mean()) if len(le_ins) else np.nan,
            le_ins_max=float(le_ins.max()) if len(le_ins) else np.nan,
            spd_ins=float(step[ins[:-1]].mean()) if ins[:-1].sum() else np.nan,
            spd_tr=float(step[~ins[:-1]].mean()) if (~ins[:-1]).sum() else np.nan,
            jrk_ins=float(jerk[ins[:-2]].mean()) if ins[:-2].sum() else np.nan,
            maxd=float(D.max()), d_at_zmin=float(D[izmin]),
        ))
        print(f"GEOMEP {name} sd{sd} ok={int(ok)} n={n} zmin={P[:,2].min():.3f} "
              f"dwell={int(ins.sum())} retry={nret} "
              f"lat@zmin={le[izmin]*1000 if not np.isnan(le[izmin]) else -1:.1f}mm "
              f"lat_ins_max={(le_ins.max()*1000 if len(le_ins) else -1):.1f}mm "
              f"spd_ins={step[ins[:-1]].mean()*1000 if ins[:-1].sum() else -1:.2f}mm "
              f"maxd={D.max():.1f}", flush=True)
    fig[f"{tag}_P"] = np.concatenate([z[f"P{sd}"][::3].astype(np.float32)
                                      for sd in seeds])
    fig[f"{tag}_L"] = np.array([len(z[f"P{sd}"][::3]) for sd in seeds])
    fig[f"{tag}_D"] = np.concatenate([z[f"D{sd}"][::3].astype(np.float32)
                                      for sd in seeds])
    fig[f"{tag}_O"] = np.array([int(z[f"O{sd}"][0]) for sd in seeds])

    def agg(sel, lab):
        r = [x for x in rows if sel(x)]
        if not r:
            return
        def m(k):
            v = np.array([x[k] for x in r], dtype=float)
            v = v[~np.isnan(v)]
            return v.mean() if len(v) else np.nan
        print(f"GEOM {name} {lab} n={len(r)} "
              f"zmin={m('zmin'):.3f} dwell={m('dwell'):.0f} retry={m('nret'):.2f} "
              f"lat@zmin={m('le_zmin')*1000:.1f}mm ins_mean={m('le_ins_mean')*1000:.1f}mm "
              f"ins_max={m('le_ins_max')*1000:.1f}mm "
              f"spd_ins={m('spd_ins')*1000:.2f}mm/step spd_tr={m('spd_tr')*1000:.2f} "
              f"jerk_ins={m('jrk_ins')*1000:.3f} maxd={m('maxd'):.1f} "
              f"d@zmin={m('d_at_zmin'):.2f}", flush=True)

    agg(lambda x: True, "ALL")
    agg(lambda x: x["ok"], "SUCC")
    agg(lambda x: not x["ok"], "FAIL")
np.savez_compressed("/mnt/pfs/yuchen/traj_fig.npz", **fig)
print("GEOM done", flush=True)
