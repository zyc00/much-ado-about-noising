"""Phase-transition landmark dispersion across rollouts, per arm.
Landmarks per episode (from dumps): pick-entry (z-min before grasp),
grasp step, lift->transit corner (first reach of 95% lift apex), rotation
onset (sustained quat angular speed after grasp) with its height z.
Reports cross-episode std over SUCCESSES + each FAILURE's deviation from
the success median (in success-std units). Prints LM lines."""
import glob
import sys

import numpy as np

EEF = slice(44, 47)
QUA = slice(47, 51)
GRI = slice(51, 53)


def qangle(q1, q2):
    d = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) *
                                     np.linalg.norm(q2) + 1e-9))
    return 2 * np.arccos(min(1.0, d))


def landmarks(obs):
    g = obs[:, GRI].sum(1)
    thr = 0.5 * (g.max() + g.min())
    closed = np.where(g < thr)[0]
    if not len(closed):
        return None
    tg = int(closed[0])
    z = obs[:, 46]
    w0 = max(0, tg - 40)
    tpick = w0 + int(np.argmin(z[w0:tg + 5]))
    pick = obs[tpick, EEF].copy()
    wend = min(len(obs) - 1, tg + 150)
    zapex = z[tg:wend].max()
    reach = np.where(z[tg:wend] >= z[tg] + 0.95 * (zapex - z[tg]))[0]
    tcorn = tg + (int(reach[0]) if len(reach) else 0)
    corner = obs[tcorn, EEF].copy()
    ang = np.array([qangle(obs[i, QUA], obs[i + 1, QUA])
                    for i in range(tg, len(obs) - 1)])
    k = 5
    sm = np.convolve(ang, np.ones(k) / k, mode="valid")
    on = np.where(sm > 0.02)[0]
    if not len(on):
        return None
    trot = tg + int(on[0])
    rot = obs[trot, EEF].copy()
    return {"pick": pick, "corner": corner, "rot": rot,
            "tg": tg, "tcorn": tcorn, "trot": trot}


for tag, dr in [("HT", "logs/rd_ht"), ("HG", "logs/rd_hg"),
                ("MIP", "logs/rd_mip")]:
    succ, fail = [], []
    for f in sorted(glob.glob(f"{dr}/ep_*.npz")):
        z = np.load(f)
        lm = landmarks(z["obs"])
        if lm is None:
            continue
        (succ if int(z["asm"]) else fail).append((int(z["seed"]), lm))
    for key, dim, nm in [("pick", 2, "pick z"), ("pick", None, "pick xyz"),
                         ("corner", 2, "corner z"),
                         ("rot", 2, "rot-onset z"),
                         ("rot", None, "rot-onset xyz")]:
        S = np.array([lm[key] for _, lm in succ])
        med = np.median(S, 0)
        if dim is None:
            sd = float(np.median(np.linalg.norm(S - med, axis=1)))
            devs = [(seed, float(np.linalg.norm(lm[key] - med) /
                                 (sd + 1e-9))) for seed, lm in fail]
        else:
            sd = float(S[:, dim].std())
            devs = [(seed, float(abs(lm[key][dim] - med[dim]) /
                                 (sd + 1e-9))) for seed, lm in fail]
        dv = " ".join(f"{s}:{d:.1f}" for s, d in devs)
        print(f"LM {tag} {nm} succ-std {1000 * sd:.1f}mm | fail-dev(sd) "
              f"{dv}", flush=True)
    ts = np.array([lm["trot"] - lm["tg"] for _, lm in succ])
    print(f"LM {tag} rot-onset timing (steps after grasp) succ p50 "
          f"{np.median(ts):.0f} std {ts.std():.1f}", flush=True)
print("LM done", flush=True)
