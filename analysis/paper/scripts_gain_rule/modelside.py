"""Model-side residual statistics from existing dumps (validation of the data-side proxy).
GR1: fit_dump_gr1_v3.npz (400 states, 5 heads, gt/mse_pred [400,232] = 8 steps x 29 dims).
fractal: fit_dump_fractal.npz (2000 states, gt/flow_pred/ht224_pred [2000,56] = 8 x 7).
pi0.5 / OFT: resid_dist_*.npz carry only per-sample mean squares (ms) -> tail + dispersion only."""
import json, sys, numpy as np
sys.path.insert(0, "/mnt/pfs/yuchen/gainrule")
from resid_stats import chunk_stats
out = {}
d = np.load("/mnt/pfs/yuchen/groot/fit_dump_gr1_v3.npz")
gt = d["gt"].reshape(-1, 8, 29)
for head in ["mse", "ht464", "ht928", "l1", "flow"]:
    p = d[f"{head}_pred"]; p = p.mean(1) if p.ndim == 3 else p
    R = gt - p.reshape(-1, 8, 29)
    out[f"gr1_{head}"] = chunk_stats(R, list(range(14, 26)), head)
d = np.load("/mnt/pfs/yuchen/groot/fit_dump_fractal.npz")
gt = d["gt"].reshape(-1, 8, 7)
for head in ["flow", "ht224"]:
    out[f"fractal_{head}"] = chunk_stats(gt - d[f"{head}_pred"].reshape(-1, 8, 7), [6], head)
for name, f in [("pi05_ht", "/mnt/pfs/yuchen/pi05/resid_dist_pi05.npz"), ("oft_ht", "/mnt/pfs/yuchen/oft/resid_dist_oft.npz")]:
    ms = np.load(f)["ms"].astype(float); s = np.sort(ms)[::-1]; n = len(ms)
    lm = np.log(ms + 1e-4 * np.median(ms))
    out[name] = {"N": n, "rms": float(np.sqrt(ms.mean())), "top1_share": float(s[:max(1, n // 100)].sum() / s.sum()),
                 "top5_share": float(s[:max(1, n // 20)].sum() / s.sum()), "sd_logm": float(lm.std()), "frac_m_gt4med": float((ms > 4 * np.median(ms)).mean())}
json.dump(out, open("/mnt/pfs/yuchen/gainrule/out/modelside.json", "w"), indent=1)
for k, v in out.items():
    vv = v.get("all", v)
    print(f"{k:14s} rms {vv['rms']:.3f} top1 {vv['top1_share']:.3f} (gauss {vv.get('top1_share_gauss', float('nan')):.3f}) kurt {vv.get('kurt_elem', float('nan')):.2f} t_df {vv.get('t_df', float('nan')):.1f} rel_time {vv.get('rel_time', float('nan')):.2f} sd_logm {vv['sd_logm']:.2f}")
