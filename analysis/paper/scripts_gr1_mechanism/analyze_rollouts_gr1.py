"""Closed-loop execution analysis of GR1 rollout dumps (HT464 / MSE / HT928), per task and pooled."""
import numpy as np, glob, os, sys
np.set_printoptions(precision=3, suppress=True, linewidth=200)
files = sorted(glob.glob("trajdump/*.npz"))
def episodes(z):
    """split each env stream into episodes using done flags; returns list of dicts"""
    done = z["done"]; succ = z["success"]; S, E = done.shape; eps = []
    arm = np.concatenate([z["act.action.left_arm"], z["act.action.right_arm"]], -1)  # (S,E,8,14)
    hand = np.concatenate([z["act.action.left_hand"], z["act.action.right_hand"]], -1)  # (S,E,8,12)
    st_arm = np.concatenate([z["obs.state.left_arm"], z["obs.state.right_arm"]], -1)[:, :, 0]  # (S,E,14)
    st_hand = np.concatenate([z["obs.state.left_hand"], z["obs.state.right_hand"]], -1)[:, :, 0]
    for e in range(E):
        start = 0
        for s in range(S):
            if done[s, e] or s == S - 1:
                sl = slice(start, s + 1)
                eps.append(dict(env=e, t0=start, t1=s, n=s + 1 - start, success=bool(succ[start:s + 1, e].any()), complete=bool(done[s, e]),
                                arm=arm[sl, e], hand=hand[sl, e], st_arm=st_arm[sl, e], st_hand=st_hand[sl, e]))
                start = s + 1
    return eps
rows = {}
for f in files:
    tag = os.path.basename(f)[:-4]; label, task = tag.split("_", 1); z = np.load(f, allow_pickle=True); eps = [e for e in episodes(z) if e["complete"]]
    # per-episode metrics
    for e in eps:
        off = e["arm"] - e["st_arm"][:, None, :]            # commanded offset from current state, (n,8,14) rad
        e["off_rms_step"] = np.sqrt((off ** 2).mean((0, 2)))  # per chunk step
        e["off_rms"] = float(np.sqrt((off ** 2).mean()))
        e["off_step7"] = float(np.sqrt((off[:, 7] ** 2).mean()))
        mv = np.linalg.norm(np.diff(e["st_arm"], axis=0), axis=-1)  # measured joint motion per macro-step (8 env steps)
        e["motion_mean"] = float(mv.mean()) if len(mv) else 0.0
        e["stall_frac"] = float((mv < 0.02).mean()) if len(mv) else 0.0   # fraction of macro-steps with almost no arm motion
        # hand: commanded closure = mean over hand dims; flips = sign changes of the commanded closure crossing mid-level
        hc = e["hand"].mean(-1)  # (n, 8)
        lvl = hc.reshape(-1); e["hand_flips"] = int((np.diff(lvl > np.median(np.concatenate([hc.reshape(-1) for _ in [0]])) ) != 0).sum())
        # within-chunk hand indecision: chunks whose 8 steps straddle the half-way level
        rng_ = hc.max(1) - hc.min(1); e["hand_inchunk_swing"] = float(rng_.mean())
    rows[tag] = eps
    n = len(eps); sr = np.mean([e["success"] for e in eps]) if n else float("nan")
    print(f"{label:6s} {task[22:-6]:28s} eps={n:2d} SR={sr:.2f} len(mean)={np.mean([e['n'] for e in eps]):6.1f} | arm cmd offset rms {np.mean([e['off_rms'] for e in eps]):.3f} rad (step7 {np.mean([e['off_step7'] for e in eps]):.3f}) | measured motion/macro-step {np.mean([e['motion_mean'] for e in eps]):.3f} rad, stall frac {np.mean([e['stall_frac'] for e in eps]):.2f} | hand in-chunk swing {np.mean([e['hand_inchunk_swing'] for e in eps]):.3f}")
print("\nper-episode detail (env, len, success, offset rms, motion, stall):")
for tag, eps in rows.items():
    print(" ", tag[:40], " ".join(f"[e{e['env']} n{e['n']} {'S' if e['success'] else 'F'} off{e['off_rms']:.2f} mv{e['motion_mean']:.2f} st{e['stall_frac']:.2f}]" for e in eps))
# pooled by label: success vs failure characteristics
print("\npooled by model: success episodes vs failure episodes")
for label in ["ht464", "mse", "ht928"]:
    eps = [e for tag, v in rows.items() if tag.startswith(label + "_") for e in v]
    for flag in [True, False]:
        sub = [e for e in eps if e["success"] == flag]
        if sub: print(f"  {label:6s} {'success' if flag else 'failure'} n={len(sub):2d} len {np.mean([e['n'] for e in sub]):6.1f} offset rms {np.mean([e['off_rms'] for e in sub]):.3f} motion {np.mean([e['motion_mean'] for e in sub]):.3f} stall {np.mean([e['stall_frac'] for e in sub]):.2f} hand swing {np.mean([e['hand_inchunk_swing'] for e in sub]):.3f}")
    if eps: print(f"  {label:6s} offset rms by chunk step (all eps): {np.mean([e['off_rms_step'] for e in eps], 0)}")
