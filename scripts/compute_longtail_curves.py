"""Run on the cluster (pi05 venv): four-panel tail curves + statistics from the WidowX probe files -> small JSON."""
import glob, json, sys, numpy as np
from scipy.special import gammaln
from scipy.stats import norm, t as student_t
RAW = "/mnt/pfs/yuchen/widowx_general_scale_20260906/raw"
def load(pattern):
    parts = {}
    for f in sorted(glob.glob(pattern)):
        with np.load(f, allow_pickle=False) as npz:
            for k in npz.files:
                if k != "metadata": parts.setdefault(k, []).append(npz[k])
    d = {k: np.concatenate(v) for k, v in parts.items()}; o = np.lexsort((d["step"], d["episode"], d["task_id"]))
    return {k: v[o] for k, v in d.items()}
def t_logpdf(v, nu, s): return gammaln((nu+1)/2) - gammaln(nu/2) - .5*np.log(nu*np.pi) - np.log(s) - (nu+1)/2*np.log1p((v/s)**2/nu)
def fit_t(values, grid=np.logspace(0.1, 2.3, 23)):
    rng = np.random.default_rng(0); values = values if len(values) <= 200000 else rng.choice(values, 200000, replace=False)
    best = (None, -np.inf, None)
    for nu in grid:
        s2 = np.mean(values**2)
        for _ in range(25): s2 = np.mean((nu+1)/(nu + values**2/s2) * values**2)
        ll = np.mean(t_logpdf(values, nu, np.sqrt(s2)))
        if ll > best[1]: best = (nu, ll, np.sqrt(s2))
    return best[0], best[2]
def tail_fit(z, lo=1.0, hi=5.0):
    a = np.abs(z); ts = np.linspace(lo, hi, 41); emp = np.array([(a > t).mean() for t in ts]); keep = emp > 0; best = (None, None, np.inf)
    for nu in np.logspace(np.log10(1.5), np.log10(300), 100):
        for sc in np.linspace(0.55, 1.3, 51):
            err = np.mean((np.log(emp[keep]) - np.log(2*student_t.sf(ts[keep]/sc, nu)))**2)
            if err < best[2]: best = (nu, sc, err)
    return best[0], best[1]
def tail(x, fold, own_scale=None):
    N, K = x.shape[:2]; x = x.reshape(N, K, 48).astype(np.float64)
    if own_scale is not None: x = x / own_scale[:, None, None]
    z = np.empty_like(x)
    for f in np.unique(fold):
        rms = np.sqrt(np.mean(x[fold != f].reshape(-1, 48)**2, axis=0)); z[fold == f] = x[fold == f] / rms
    z = z / np.sqrt(np.mean(z**2)); a = np.abs(z); ts = np.linspace(0.5, 6.5, 121); emp = np.array([(a > t).mean() for t in ts])
    gauss, stud, lap = [], [], []
    for f in np.unique(fold):
        tr, te = z[fold != f].ravel(), z[fold == f].ravel(); sg = np.sqrt(np.mean(tr**2)); nu, s = fit_t(tr); b = np.mean(np.abs(tr))
        gauss.append(np.mean(.5*np.log(2*np.pi*sg**2) + te**2/(2*sg**2))); stud.append(np.mean(-t_logpdf(te, nu, s))); lap.append(np.mean(np.log(2*b) + np.abs(te)/b))
    nu_t, s_t = tail_fit(z.ravel())
    return dict(ts=ts.tolist(), emp=emp.tolist(), p_gt3=float((a > 3).mean()), gauss_gt3=float(2*norm.sf(3)), kurtosis=float(np.mean(z**4) - 3),
                laplace_b=float(np.mean(a)), tailfit_nu=float(nu_t), tailfit_scale=float(s_t), heldout_gauss=float(np.mean(gauss)),
                heldout_student=float(np.mean(stud)), heldout_laplace=float(np.mean(lap)), n=int(z.size))
mse, hg, flow = load(f"{RAW}/mse_rank*.npz"), load(f"{RAW}/hg_rank*.npz"), load(f"{RAW}_k1024/flow_rank*.npz")
for k in ("task_id", "episode", "step"): assert (mse[k] == hg[k]).all() and (mse[k] == flow[k]).all()
task, ep = mse["task_id"], mse["episode"]; rng = np.random.default_rng(20260906); fold = np.empty(len(task), int)
for t in np.unique(task):
    for i, e in enumerate(rng.permutation(np.unique(ep[task == t]))): fold[ep == e] = i % 5
S = flow["samples"].astype(np.float32); dev = S - S.mean(axis=1, keepdims=True); del S
coord_rms = np.sqrt(np.mean(dev.reshape(-1, 48)**2, axis=0)); own = np.sqrt(np.mean((dev.reshape(len(task), 1024, 48)/coord_rms)**2, axis=(1, 2)))
lab = flow["samples"][:, ::16].astype(np.float32) - flow["target"].astype(np.float32)[:, None]
out = {"a_hg": tail(hg["residual"][:, None], fold, hg["sigma"]), "b_mse": tail(mse["residual"][:, None], fold, None),
       "c_flow_fixed": tail(dev[:, ::16], fold, None), "d_flow_own": tail(dev[:, ::16], fold, own),
       "flow_to_label_own": tail(lab, fold, own), "flow_own_scale_q10_50_90": np.quantile(own, [.1, .5, .9]).tolist(),
       "corr_log_own_vs_hg_sigma": float(np.corrcoef(np.log(own), np.log(hg["sigma"]))[0, 1]), "states": int(len(task)), "draws": 1024, "draws_in_curve": 64}
json.dump(out, open("/mnt/pfs/yuchen/widowx_general_scale_20260906/longtail_curves.json", "w"))
for k in ["a_hg", "b_mse", "c_flow_fixed", "d_flow_own", "flow_to_label_own"]:
    r = out[k]; print(f"{k:18s} P>3 {100*r['p_gt3']:.2f}% ({r['p_gt3']/r['gauss_gt3']:.1f}x) kurt {r['kurtosis']:.1f} | held-out G {r['heldout_gauss']:.3f} L {r['heldout_laplace']:.3f} t {r['heldout_student']:.3f} | tail nu {r['tailfit_nu']:.1f}", flush=True)
print("DONE")
