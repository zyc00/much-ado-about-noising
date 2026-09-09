"""v2: headroom-controlled tests. Does a residual statistic of the DATA predict the gain of flow / HT over MSE?"""
import json, glob, os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/outcomes.json"))
OUT = os.environ.get("OUT", "out_c1")
ST = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{HERE}/{OUT}/*.json") if "modelside" not in f}
MS = json.load(open(f"{HERE}/out/modelside.json"))
tasks = D["tasks"]; H_RM = int(os.environ.get("H_RM", "10"))
FAMS = ["knn", "zoh", "sg"]; TAGS = ["all", "cont"]
BASE = ["rms", "top1_share", "top5_share", "kurt_elem", "t_df", "rel_time", "rel_dim", "sd_logm", "hetero_signal", "frac_m_gt4med"]

def feats(name, H):
    s = ST.get(name, {}).get(f"H{H}", {}); out = {}
    for fam in FAMS:
        if fam not in s: continue
        for tag in TAGS:
            v = s[fam].get(tag)
            if not v: continue
            for k in BASE: out[f"{fam}.{tag}.{k}"] = v[k]
            out[f"{fam}.{tag}.tail_x"] = v["top1_share"] / v["top1_share_gauss"]
            out[f"{fam}.{tag}.sd_excess"] = v["sd_logm"] - v["sd_logm_gauss"]
            out[f"{fam}.{tag}.log_tdf"] = np.log(v["t_df"])
    return out

cells = []
for mod in ["state", "image"]:
    for bb, v in D["robomimic_late_avg"][mod].items():
        for i, t in enumerate(tasks):
            cells.append(dict(task=t, mod=mod, bb=bb, flow=v["flow"][i], mse=v["mse"][i], ht=v["ht"][i]))
rows = []
for t in tasks:
    c = [x for x in cells if x["task"] == t]
    r = dict(task=t, mse=np.mean([x["mse"] for x in c]), flow=np.mean([x["flow"] for x in c]), ht=np.mean([x["ht"] for x in c]))
    for a, b, nm in [("flow", "mse", "g_flow"), ("ht", "mse", "g_ht"), ("ht", "flow", "g_htflow")]:
        r[nm] = np.mean([x[a] - x[b] for x in c]); r[nm + "_npos"] = sum(x[a] - x[b] > 0.005 for x in c); r[nm + "_nneg"] = sum(x[a] - x[b] < -0.005 for x in c)
        for mod in ["state", "image"]: r[f"{nm}_{mod}"] = np.mean([x[a] - x[b] for x in c if x["mod"] == mod])
    r["headroom"] = 1 - r["mse"]
    r["rel_flow"] = r["g_flow"] / r["headroom"] if r["headroom"] > 0.03 else np.nan
    r["rel_ht"] = r["g_ht"] / r["headroom"] if r["headroom"] > 0.03 else np.nan
    r.update(feats(t, H_RM)); rows.append(r)
def col(k, rs=None): return np.array([r.get(k, np.nan) for r in (rs or rows)], float)
pred_keys = sorted(k for k in rows[0] if any(k.startswith(f) for f in FAMS))

print(f"[stats dir {OUT}]")
print(f"=== robomimic outcomes (late-average SR; mean over 3 backbones x 2 modalities; npos/nneg = cells out of 6 with gain > +0.005 / < -0.005), H={H_RM}")
print(f"{'task':13s} {'MSE':>5s} {'head':>5s} | {'flow-MSE':>8s} {'+/-':>4s} {'rel':>5s} | {'HT-MSE':>7s} {'+/-':>4s} {'rel':>5s} | {'HT-flow':>7s} {'+/-':>4s}")
for r in rows:
    print(f"{r['task']:13s} {r['mse']:5.2f} {r['headroom']:5.2f} | {r['g_flow']:+8.3f} {r['g_flow_npos']}/{r['g_flow_nneg']:<2d} {r['rel_flow']:5.2f} | {r['g_ht']:+7.3f} {r['g_ht_npos']}/{r['g_ht_nneg']:<2d} {r['rel_ht']:5.2f} | {r['g_htflow']:+7.3f} {r['g_htflow_npos']}/{r['g_htflow_nneg']:<2d}")

show = ["knn.cont.rms", "knn.cont.tail_x", "knn.cont.kurt_elem", "knn.cont.t_df", "knn.cont.rel_time", "knn.cont.sd_excess", "zoh.cont.tail_x", "zoh.cont.rel_time", "sg.cont.tail_x", "sg.cont.rel_time", "knn.all.tail_x"]
print(f"\n=== data residual statistics (H={H_RM}); tail_x = top-1% chunk share / Gaussian reference; rel_time = split-half (time) reliability of chunk scale; sd_excess = sd(log m) beyond chi2")
print(f"{'task':13s} " + " ".join(f"{k.replace('.cont',''):>13s}" for k in show))
for r in rows: print(f"{r['task']:13s} " + " ".join(f"{r.get(k, np.nan):13.3f}" for k in show))

def resid_on(y, x):
    ok = np.isfinite(x) & np.isfinite(y); b = np.polyfit(x[ok], y[ok], 1); out = np.full_like(y, np.nan); out[ok] = y[ok] - np.polyval(b, x[ok]); return out
def table(outcome, label, rs, control=None, top=10):
    y = col(outcome, rs); hr = col("headroom", rs)
    if control: y = resid_on(y, hr)
    res = []
    for k in pred_keys:
        x = col(k, rs)
        if control: x = resid_on(x, hr)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5 or np.std(x[ok]) == 0: continue
        rho, p = spearmanr(x[ok], y[ok]); res.append((k, rho, p, int(ok.sum())))
    res.sort(key=lambda z: -abs(z[1]))
    print(f"\n--- Spearman with {label}{' (both residualized on headroom = partial)' if control else ''}, n={len(rs)}: top {top}")
    for k, rho, p, n in res[:top]: print(f"   {k:28s} rho {rho:+.2f}  p {p:.3f}  (n={n})")
    return res
nonceil = [r for r in rows if r["headroom"] > 0.03]
print(f"\nheadroom alone: " + "  ".join(f"{o}: rho {spearmanr(col('headroom'), col(o))[0]:+.2f}" for o in ["g_flow", "g_ht", "g_htflow"]))
for outcome, label in [("g_flow", "flow - MSE"), ("g_ht", "HT - MSE"), ("g_htflow", "HT - flow")]:
    table(outcome, label, rows, control=True)
for outcome, label in [("rel_flow", "relative flow gain (flow-MSE)/(1-MSE)"), ("rel_ht", "relative HT gain (HT-MSE)/(1-MSE)")]:
    table(outcome, label, nonceil, top=8)

# ph vs mh pairs (same task, different demonstrator pool): the noise contrast with the task held fixed
print("\n=== paired ph -> mh contrast (same task; mh = multi-human, noisier)")
print(f"{'task':10s} {'d tail_x':>9s} {'d kurt':>7s} {'d rel_t':>8s} {'d sdx':>6s} | {'d rel_flow':>10s} {'d rel_ht':>9s} {'d (HT-flow)':>11s}")
byt = {r["task"]: r for r in rows}
for base in ["lift", "can", "square", "transport"]:
    a, b = byt.get(f"{base}-ph"), byt.get(f"{base}-mh")
    if not a or not b: continue
    d = lambda k: b.get(k, np.nan) - a.get(k, np.nan)
    print(f"{base:10s} {d('knn.cont.tail_x'):+9.2f} {d('knn.cont.kurt_elem'):+7.2f} {d('knn.cont.rel_time'):+8.3f} {d('knn.cont.sd_excess'):+6.2f} | {d('rel_flow'):+10.3f} {d('rel_ht'):+9.3f} {d('g_htflow'):+11.3f}")

# LOO rules
def loo(ykey, xkeys, rs):
    y = col(ykey, rs); X = np.column_stack([np.ones(len(rs))] + [col(k, rs) for k in xkeys]); ok = np.isfinite(X).all(1) & np.isfinite(y); y, X = y[ok], X[ok]
    if len(y) < 6: return np.nan, np.nan, len(y)
    pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False; pred[i] = X[i] @ np.linalg.lstsq(X[m], y[m], rcond=None)[0]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum(), len(y)
print("\n=== leave-one-out R^2:  gain = a + b*headroom (+ c*stat)")
cands = [k for k in pred_keys if k.split(".")[2] in ("tail_x", "rel_time", "sd_excess", "kurt_elem", "rms", "hetero_signal", "log_tdf")]
for outcome in ["g_flow", "g_ht", "g_htflow"]:
    l0, i0, n = loo(outcome, ["headroom"], rows); print(f"  {outcome:9s} headroom only: LOO R2 {l0:+.2f} (in-sample {i0:+.2f}, n={n})")
    best = sorted([(loo(outcome, ["headroom", k], rows), k) for k in cands], key=lambda z: -np.nan_to_num(z[0][0], nan=-9))[:5]
    for (l, ins, n), k in best: print(f"      + {k:24s} LOO R2 {l:+.2f} (in-sample {ins:+.2f})")

# VLA table + model-side check
print("\n=== VLA stacks: data-side statistics at own H; gains in SR points")
print(f"{'stack':22s} {'H':>3s} {'MSE':>5s} {'flow':>5s} {'HT':>5s} {'HTc2':>5s} | {'f-M':>5s} {'H-M':>5s} {'H-f':>5s} | {'knn.tail_x':>10s} {'knn.kurt':>8s} {'knn.rel_t':>9s} {'knn.sdx':>7s} {'zoh.tail_x':>10s} {'zoh.rel_t':>9s} {'sg.tail_x':>9s}")
vla = []
for v in D["vla"]:
    f = feats(v["dataset"], v["H"]); vla.append((v, f))
    g = lambda a, b: f"{(v[a]-v[b]):+5.1f}" if v.get(a) is not None and v.get(b) is not None else "   --"
    fmt = lambda k: f"{f.get(k, np.nan):.3f}" if k in f else "  --"
    print(f"{v['stack']:22s} {v['H']:3d} {str(v['mse']):>5s} {str(v['flow']):>5s} {str(v['ht']):>5s} {str(v['ht_c2']):>5s} | {g('flow','mse')} {g('ht','mse')} {g('ht','flow')} | {fmt('knn.cont.tail_x'):>10s} {fmt('knn.cont.kurt_elem'):>8s} {fmt('knn.cont.rel_time'):>9s} {fmt('knn.cont.sd_excess'):>7s} {fmt('zoh.cont.tail_x'):>10s} {fmt('zoh.cont.rel_time'):>9s} {fmt('sg.cont.tail_x'):>9s}")
print("\n=== model-side residuals (existing dumps) vs data-side proxy")
for k in ["gr1_mse", "gr1_flow", "fractal_flow", "fractal_ht224"]:
    v = MS[k]["cont"]; print(f"  {k:14s} rms {v['rms']:.3f} tail_x {v['top1_share']/v['top1_share_gauss']:.2f} kurt {v['kurt_elem']:.1f} rel_time {v['rel_time']:.3f} sd_excess {v['sd_logm']-v['sd_logm_gauss']:.2f}")
for k in ["pi05_ht", "oft_ht"]:
    v = MS[k]; print(f"  {k:14s} rms {v['rms']:.3f} top1 {v['top1_share']:.3f} sd_logm {v['sd_logm']:.2f}")

# figure
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
pairs = [("knn.cont.tail_x", "tail heaviness of state-conditional residual (top-1% share / Gaussian)"), ("knn.cont.rel_time", "time split-half reliability of chunk scale (heteroscedasticity)"), ("headroom", "headroom = 1 - MSE success")]
for j, (xk, xl) in enumerate(pairs):
    for i, (yk, yl) in enumerate([("g_ht", "HT - MSE (SR)"), ("g_htflow", "HT - flow (SR)")]):
        ax = axes[i, j]; x, y = col(xk), col(yk)
        ax.scatter(x, y, c="#1f77b4", s=40)
        for r in rows: ax.annotate(r["task"], (r.get(xk, np.nan), r[yk]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for v, f in vla:
            if v.get("mse") is None and yk == "g_ht": continue
            if v.get("flow") is None and yk == "g_htflow": continue
            yy = (v["ht"] - v["mse"]) / 100 if yk == "g_ht" else (v["ht"] - v["flow"]) / 100
            xx = f.get(xk, np.nan) if xk != "headroom" else (1 - v["mse"] / 100 if v.get("mse") is not None else np.nan)
            ax.scatter([xx], [yy], marker="s", c="#d62728", s=40); ax.annotate(v["stack"].split(" ")[0], (xx, yy), fontsize=7, color="#d62728", xytext=(3, -8), textcoords="offset points")
        ok = np.isfinite(x) & np.isfinite(y); rho = spearmanr(x[ok], y[ok])[0]
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8); ax.axhline(0, color="gray", lw=0.5); ax.set_title(f"robomimic Spearman rho = {rho:+.2f} (n={ok.sum()})", fontsize=9)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_gain_rule_{OUT}_H{H_RM}.png", dpi=130); print("\nfigure:", f"{HERE}/fig_gain_rule.png")
json.dump({"rows": rows}, open(f"{HERE}/gain_rule_rows.json", "w"), indent=1, default=float)
