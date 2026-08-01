"""Stall anatomy from TRAJDUMP traces: dwell at near-zero command, longest low-|cmd|
run, and eef progress in the final phase, success vs failure."""
import glob, os
import numpy as np
PREFIX = os.environ.get("PREFIX", "analysis/traj_vis/trmh_ht")
lows, runs, prog, succs = [], [], [], []
for f in sorted(glob.glob(f"{PREFIX}_ep*.npz")):
    z = np.load(f)
    O, A, S = z["obs"], z["act"], z["succ"]
    for e in range(len(S)):
        a = A[e]
        m = np.maximum(np.linalg.norm(a[:, 0:3], axis=1), np.linalg.norm(a[:, 7:10], axis=1))
        low = m < 0.15
        # longest consecutive low-command run
        best, cur = 0, 0
        for v in low:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        # eef displacement over final 300 steps (both arms; obs layout: object then eefs)
        od = O.shape[-1] - 18
        e0 = O[e][:, od:od+3]; e1 = O[e][:, od+9:od+12]
        k = min(len(e0) - 1, 38)  # obs recorded per 8-step chunk; ~300 env steps
        d300 = np.linalg.norm(e0[-1]-e0[-k], axis=-1) + np.linalg.norm(e1[-1]-e1[-k], axis=-1)
        lows.append(low.mean()); runs.append(best); prog.append(d300); succs.append(S[e])
lows, runs, prog, succs = map(np.array, (lows, runs, prog, succs))
ok, ko = succs > 0, succs == 0
print(f"STALL {os.path.basename(PREFIX)}: n_succ={ok.sum()} n_fail={ko.sum()}")
print(f"  frac steps |cmd|<0.15 : succ {np.median(lows[ok]):.2f} | fail {np.median(lows[ko]):.2f}")
print(f"  longest low-cmd run   : succ {np.median(runs[ok]):.0f} | fail {np.median(runs[ko]):.0f} (p90 fail {np.percentile(runs[ko],90):.0f})")
print(f"  eef displacement last300: succ {np.median(prog[ok]):.3f} | fail {np.median(prog[ko]):.3f}")
print("STALL-DONE")
