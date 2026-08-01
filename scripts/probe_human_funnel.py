"""Task-semantic behavioral census: what does the learned policy DO, episode by episode?

Stage funnel per episode (from executed actions + eef positions + env flags):
  S1 close-attempt   : any gripper-close command issued
  S2 grasp-hold      : closure held >= 30 consecutive steps
  S3 lift            : eef z rises >= 4cm while closed
  S4 transport       : xy displacement >= 12cm after lift
  S5 frame-assembled : env asm flag (episode-level)
  S6 release+regrasp : release after S5 followed by another closure (tool pickup)
  S7 success         : env success flag
Plus: number of grasp cycles (close events), time-to-first-close, terminal stage, and
what the policy is doing in its final 100 steps (net displacement + command magnitude).
Models: chiunet pairs s5 and s5001.
"""
import os
import numpy as np

def census(name):
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    rows = []
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; p = z[f"ep{i}_pos"]; m = z[f"ep{i}_meta"]
        L = min(len(a), len(p) - 1)
        g = a[:L, 6]
        closes = [t for t in range(1, L) if g[t-1] < 0 and g[t] >= 0]
        opens = [t for t in range(1, L) if g[t-1] >= 0 and g[t] < 0]
        s1 = len(closes) > 0
        s2 = False; hold_start = None
        for c in closes:
            nxt_open = next((t for t in opens if t > c), L)
            if nxt_open - c >= 30: s2 = True; hold_start = c; break
        s3 = False; s4 = False
        if s2:
            seg_end = next((t for t in opens if t > hold_start), L)
            zrise = p[hold_start:seg_end, 2].max() - p[hold_start, 2] if seg_end > hold_start else 0
            s3 = zrise >= 0.04
            xy = np.linalg.norm(p[hold_start:seg_end, :2] - p[hold_start, :2], axis=1).max() if seg_end > hold_start else 0
            s4 = s3 and xy >= 0.12
        s5 = bool(m[1]); s7 = bool(m[0])
        s6 = False
        if s5:
            rel = next((t for t in opens if t > (hold_start or 0)), None)
            if rel is not None:
                s6 = any(c > rel for c in closes)
        stage = 7 if s7 else (6 if s6 else (5 if s5 else (4 if s4 else (3 if s3 else (2 if s2 else (1 if s1 else 0))))))
        tail = slice(max(0, L - 100), L)
        net_tail = np.linalg.norm(p[min(L, len(p)-1)] - p[max(0, L-100)])
        mag_tail = float(np.median(np.linalg.norm(a[tail, :3], axis=1)))
        rows.append(dict(stage=stage, cycles=len(closes), tfc=closes[0] if closes else -1,
                         net_tail=net_tail, mag_tail=mag_tail, succ=s7, asm=s5))
        i += 1
    return rows

for name in ["hMSE_s5", "hMIP_s5", "hMSE_s5001", "hMIP_s5001"]:
    if not os.path.exists(f"analysis/traj_vis/human_{name}.npz"):
        print(f"FUNNEL {name}: no dump"); continue
    rows = census(name)
    n = len(rows)
    reached = [sum(1 for r in rows if r["stage"] >= s) for s in range(1, 8)]
    term = {}
    for r in rows: term[r["stage"]] = term.get(r["stage"], 0) + 1
    cyc = np.array([r["cycles"] for r in rows])
    nt = np.array([r["net_tail"] for r in rows]); mt = np.array([r["mag_tail"] for r in rows])
    print(f"FUNNEL {name} (n={n}): reach S1..S7 = " + "/".join(str(x) for x in reached), flush=True)
    print(f"  terminal-stage counts: " + ", ".join(f"S{k}:{v}" for k, v in sorted(term.items())), flush=True)
    print(f"  grasp cycles p50={np.median(cyc):.0f} max={cyc.max()} | final-100-steps: net-move p50={np.median(nt)*100:.1f}cm cmd-mag p50={np.median(mt):.3f}", flush=True)
print("FUNNEL-DONE")
