"""GR1 training-set fit analysis (corrected normalization). Usage: python analyze_fit_v2.py fit_dump_gr1_v2.npz"""
import sys, json, numpy as np
from gr1_norm_utils import lohi, unnorm, norm, load_stats
np.set_printoptions(precision=3, suppress=True, linewidth=220)
z = np.load(sys.argv[1] if len(sys.argv) > 1 else "fit_dump_gr1_v2.npz", allow_pickle=True); T, D = 8, 29
GT = z["gt"].reshape(-1, T, D); N = len(GT)
models = [k[:-5] for k in z.files if k.endswith("_pred")]
P = {m: z[m + "_pred"].mean(1).reshape(N, T, D) for m in models}
A = load_stats("statsA.json"); lo, hi = lohi(A)
ARM, HAND, WAIST = slice(0, 14), slice(14, 26), slice(26, 29)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
ds = z["ident_ds"]; print(f"N={N}; models {models}; datasets: {dict(zip(*np.unique(ds, return_counts=True)))}; episodes {len(set(zip(ds, z['ident_ep'])))}")
print("forward loss per model (should match training logs):", {m: round(float(z[m + '_loss'].mean()), 4) for m in models if m + "_loss" in z.files})
print("\n[A] ARM: GT rms / residual rms / R^2 by chunk step")
print("  GT rms   " + " ".join(f"{rms(GT[:, t, ARM]):.3f}" for t in range(T)))
for m in models:
    r = GT[:, :, ARM] - P[m][:, :, ARM]
    print(f"  {m:6s} resid " + " ".join(f"{rms(r[:, t]):.3f}" for t in range(T)) + "   R^2 " + " ".join(f"{1 - np.mean(r[:, t]**2) / np.mean(GT[:, t, ARM]**2):.2f}" for t in range(T)) + f"   all-steps resid {rms(r):.4f}")
print("\n[B] joint-group residual rms (all steps): left arm | right arm | hands | waist")
for m in models: print(f"  {m:6s} " + " | ".join(f"{rms(GT[:, :, s] - P[m][:, :, s]):.4f}" for s in [slice(0, 7), slice(7, 14), HAND, WAIST]))
print("\n[1] ARM residual rms by |GT offset| bin (normalized units); bin population in %")
edges = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 10]; g = np.abs(GT[:, :, ARM]).ravel(); b = np.digitize(g, edges[1:-1])
for m in models:
    r = (GT[:, :, ARM] - P[m][:, :, ARM]).ravel()
    print(f"  {m:6s} " + "  ".join(f"[{edges[i]:.2f},{edges[i+1]:.2f}) {rms(r[b == i]) if (b == i).any() else float('nan'):.3f} ({np.mean(b == i)*100:4.1f}%)" for i in range(6)))
print("\n[2] amplitude slope pred ~ slope*GT (per chunk step) and corr, ARM (1.0 = full amplitude)")
for m in models:
    sl = [np.sum(P[m][:, t, ARM] * GT[:, t, ARM]) / np.sum(GT[:, t, ARM] ** 2) for t in range(T)]
    xa, ya = GT[:, :, ARM].ravel(), P[m][:, :, ARM].ravel()
    print(f"  {m:6s} steps " + " ".join(f"{s:.2f}" for s in sl) + f"   overall {np.sum(xa*ya)/np.sum(xa*xa):.3f}  corr {np.corrcoef(xa, ya)[0,1]:.3f}")
print("\n[C] ARM per (sample, step) with |GT|>0.05: |pred|/|GT| median, cos(pred,GT) median")
for m in models:
    p = P[m][:, :, ARM].reshape(-1, 14); g = GT[:, :, ARM].reshape(-1, 14); keep = np.linalg.norm(g, axis=1) > 0.05
    ratio = np.linalg.norm(p, axis=1)[keep] / np.linalg.norm(g, axis=1)[keep]; cos = np.sum(p * g, 1)[keep] / (np.linalg.norm(p, axis=1)[keep] * np.linalg.norm(g, axis=1)[keep] + 1e-9)
    print(f"  {m:6s} |pred|/|GT| median {np.median(ratio):.3f} (q25 {np.quantile(ratio,0.25):.2f}, q75 {np.quantile(ratio,0.75):.2f})  cos median {np.median(cos):.3f}")
if "mse" in models:
    print("\n[D] projection of (pred_m - pred_mse) onto MSE's residual (GT - pred_mse), ARM per step (1 = closes MSE's residual fully)")
    for m in [x for x in models if x != "mse"]:
        d = P[m][:, :, ARM] - P["mse"][:, :, ARM]; e = GT[:, :, ARM] - P["mse"][:, :, ARM]
        print(f"  {m:6s} " + " ".join(f"{np.sum(d[:, t]*e[:, t])/np.sum(e[:, t]**2):5.2f}" for t in range(T)) + f"   |d| rms {rms(d):.4f} vs MSE resid rms {rms(e):.4f}")
    print("\n[E] per-sample ARM chunk residual: HT-MSE delta by MSE-residual decile (negative = HT better)")
    rm = np.sqrt(np.mean((GT[:, :, ARM] - P["mse"][:, :, ARM]) ** 2, axis=(1, 2)))
    dec = np.digitize(rm, np.quantile(rm, np.linspace(0, 1, 11)[1:-1]))
    hdr = "  decile mse_rms " + " ".join(f"{m:>8s}" for m in models if m != "mse"); print(hdr)
    for k in range(10):
        s = dec == k; print(f"  {k:6d} {rm[s].mean():.4f} " + " ".join(f"{(np.sqrt(np.mean((GT[s][:, :, ARM]-P[m][s][:, :, ARM])**2, axis=(1,2))) - rm[s]).mean():+8.4f}" for m in models if m != "mse"))
    for m in [x for x in models if x != "mse"]:
        rh = np.sqrt(np.mean((GT[:, :, ARM] - P[m][:, :, ARM]) ** 2, axis=(1, 2))); print(f"  {m}: better than MSE on {np.mean(rh < rm)*100:.0f}% of samples; median ratio MSE/{m} {np.median(rm/rh):.3f}")
# ---- hand transitions inside chunk ----
H = GT[:, :, HAND]; lv = np.unique(np.round(H, 2)); 
print(f"\n[3] HAND channels: distinct GT levels (rounded) {len(lv)}; residual rms by model: " + ", ".join(f"{m} {rms(GT[:, :, HAND]-P[m][:, :, HAND]):.4f}" for m in models))
jump = np.abs(np.diff(H, axis=1)).max(axis=(1, 2)); has = jump > 0.3; print(f"  samples with an in-chunk hand jump (>0.3): {has.sum()} of {N}")
if has.sum() > 0:
    for m in models:
        pr = P[m][has][:, :, HAND]; g = H[has]
        # timing: step of max |diff| in GT vs pred
        tg = np.abs(np.diff(g, axis=1)).max(-1).argmax(1); tp = np.abs(np.diff(pr, axis=1)).max(-1).argmax(1); jp = np.abs(np.diff(pr, axis=1)).max(axis=(1, 2))
        print(f"  {m:6s} pred jump size median {np.median(jp):.2f} (GT {np.median(jump[has]):.2f}); |timing err| median {np.median(np.abs(tg-tp)):.0f} steps; residual on these samples {rms(g-pr):.4f} vs on others {rms(H[~has]-P[m][~has][:, :, HAND]):.4f}")
# ---- physical / EE space ----
try:
    from gr1_fk_lib import fk
    lo_s = np.concatenate([np.array(A["state"][k]["q01"]) for k in ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]]); hi_s = np.concatenate([np.array(A["state"][k]["q99"]) for k in ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]])
    st = unnorm(z["state"][:, :29], lo_s, hi_s)  # physical state (rad)
    def ee_end(offs):  # offs: (N,T,29) physical offsets -> right/left wrist positions at step 7 and at step 0 state
        out = np.zeros((N, 2, 3))
        for i in range(N):
            q = st[i]; l0, r0 = fk(q[:7], q[7:14], q[26:29]); qe = q + offs[i, 7]; l1, r1 = fk(qe[:7], qe[7:14], q[26:29] if True else qe[26:29])
            out[i, 0] = r1 - r0; out[i, 1] = l1 - l0
        return out
    gphys = unnorm(GT, lo, hi); ee_gt = ee_end(gphys)
    print("\n[F] end-effector (wrist) displacement over the chunk (step 7 vs current state), cm: GT amplitude median {:.1f} (right) {:.1f} (left)".format(100*np.median(np.linalg.norm(ee_gt[:, 0], axis=1)), 100*np.median(np.linalg.norm(ee_gt[:, 1], axis=1))))
    for m in models:
        ee = ee_end(unnorm(P[m], lo, hi)); err = np.linalg.norm(ee - ee_gt, axis=2) * 100; amp = np.linalg.norm(ee, axis=2) / (np.linalg.norm(ee_gt, axis=2) + 1e-6)
        print(f"  {m:6s} wrist endpoint error median {np.median(err[:, 0]):.2f} cm (right) {np.median(err[:, 1]):.2f} (left); mean {err[:, 0].mean():.2f} / {err[:, 1].mean():.2f}; displacement ratio pred/GT median {np.median(amp[:, 0]):.2f} (right)")
    np.savez("fit_v2_ee.npz", ee_gt=ee_gt, **{m: ee_end(unnorm(P[m], lo, hi)) for m in models})
except Exception as ex: print("FK section skipped:", ex)
# ---- candidate concrete examples ----
if "mse" in models and "ht464" in models:
    rh = np.sqrt(np.mean((GT[:, :, ARM] - P["ht464"][:, :, ARM]) ** 2, axis=(1, 2))); ga = np.sqrt(np.mean(GT[:, :, ARM] ** 2, axis=(1, 2)))
    score = (rm - rh) * (ga > np.quantile(ga, 0.5)); o = np.argsort(-score)[:8]
    print("\n[G] candidate examples (HT much better than MSE, large GT motion): idx, dataset, ep, step, GT amp, res_mse, res_ht, res_flow")
    for i in o: print(f"  {i:3d} {ds[i][12:40]:28s} ep{z['ident_ep'][i]:4d} s{z['ident_step'][i]:4d}  amp {ga[i]:.3f}  mse {rm[i]:.3f}  ht {rh[i]:.3f}" + (f"  flow {np.sqrt(np.mean((GT[i][:, ARM]-P['flow'][i][:, ARM])**2)):.3f}" if "flow" in models else ""))
    np.save("fit_v2_candidates.npy", o)
