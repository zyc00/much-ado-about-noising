#!/usr/bin/env python3
"""Tail statistic from policy outputs to numbers, in one file, two implementations that must agree:

  tail_fast(x, fold, own_scale)      vectorized (used for the paper numbers)
  tail_readable(x, fold, own_scale)  explicit loops, no ravel/reshape, one named step at a time (used to audit the fast one)
  check_equivalence()                runs both on a subset and asserts identical results

Inputs (both functions)
  x          array [N, K, 8, 6]: per state, K vectors of 8 steps x 6 continuous action channels, in the model's training
             action space (a residual has K = 1; Flow has K draws of sample-minus-sample-mean)
  fold       array [N]: episode-fold id of each state; every scale is estimated on the OTHER folds
  own_scale  array [N] or None: the model's own per-state scale (HG: its sigma(x); Flow: RMS spread of its draws)
Steps
  1  divide by own_scale (if given)
  2  divide each of the 48 coordinates by its RMS measured on the other folds
  3  rescale everything to unit RMS and take the magnitude |z|
  4  report P(|z| > 3), the ratio to the Gaussian 0.27 %, kurtosis, and the held-out NLL gain per coordinate of a
     Student-t over a Gaussian (both fitted by maximum likelihood on the other folds)
"""
import glob, numpy as np
from scipy.special import gammaln
from scipy.stats import norm

NU_GRID = np.logspace(0.1, 2.3, 23)

def t_logpdf(v, nu, s):
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(nu * np.pi) - np.log(s) - (nu + 1) / 2 * np.log1p((v / s) ** 2 / nu)

def fit_student_t(values):
    """maximum-likelihood (nu, scale) of a 1-D Student-t: grid on nu, fixed-point iteration for the scale"""
    best_nu, best_ll, best_s = None, -np.inf, None
    for nu in NU_GRID:
        s2 = np.mean(values ** 2)
        for _ in range(25):
            s2 = np.mean((nu + 1) / (nu + values ** 2 / s2) * values ** 2)
        ll = np.mean(t_logpdf(values, nu, np.sqrt(s2)))
        if ll > best_ll: best_nu, best_ll, best_s = nu, ll, np.sqrt(s2)
    return best_nu, best_s

# ----------------------------------------------------------------------------------------------- performance version
def tail_fast(x, fold, own_scale=None):
    N, K = x.shape[:2]; x = x.reshape(N, K, 48).astype(np.float64)
    if own_scale is not None: x = x / own_scale[:, None, None]                                   # step 1
    z = np.empty_like(x)
    for f in np.unique(fold):                                                                     # step 2
        rms = np.sqrt(np.mean(x[fold != f].reshape(-1, 48) ** 2, axis=0)); z[fold == f] = x[fold == f] / rms
    z = z / np.sqrt(np.mean(z ** 2)); a = np.abs(z)                                               # step 3
    p3, g3 = (a > 3).mean(), 2 * norm.sf(3)                                                       # step 4
    gauss, stud = [], []
    for f in np.unique(fold):
        tr, te = z[fold != f].ravel(), z[fold == f].ravel()
        sg = np.sqrt(np.mean(tr ** 2)); nu, s = fit_student_t(tr)
        gauss.append(np.mean(.5 * np.log(2 * np.pi * sg ** 2) + te ** 2 / (2 * sg ** 2))); stud.append(np.mean(-t_logpdf(te, nu, s)))
    return dict(p_gt3=float(p3), gauss_gt3=float(g3), excess=float(p3 / g3), kurtosis=float(np.mean(z ** 4) - 3),
                heldout_nll_gain=float(np.mean(gauss) - np.mean(stud)))

# ----------------------------------------------------------------------------------------------- readable version
def tail_readable(x, fold, own_scale=None):
    n_states, n_draws = x.shape[0], x.shape[1]
    folds = sorted(set(int(f) for f in fold))

    # step 1: each state's vectors divided by that state's own scale (1.0 when the model has no scale)
    scaled = np.zeros((n_states, n_draws, 8, 6))
    for n in range(n_states):
        scale_n = own_scale[n] if own_scale is not None else 1.0
        scaled[n] = x[n] / scale_n

    # step 2: coordinate (step t, channel c) divided by its RMS over all states of the OTHER folds and all their draws
    z = np.zeros((n_states, n_draws, 8, 6))
    for f in folds:
        other_states = [n for n in range(n_states) if fold[n] != f]
        coord_rms = np.zeros((8, 6))
        for t in range(8):
            for c in range(6):
                sq_sum, count = 0.0, 0
                for n in other_states:
                    for k in range(n_draws):
                        sq_sum += scaled[n, k, t, c] ** 2; count += 1
                coord_rms[t, c] = np.sqrt(sq_sum / count)
        for n in range(n_states):
            if fold[n] == f:
                z[n] = scaled[n] / coord_rms

    # step 3: one overall RMS so that a standard normal is the reference; magnitude
    overall_rms = np.sqrt(np.mean(z ** 2))
    z = z / overall_rms
    magnitude = np.abs(z)

    # step 4a: share beyond 3 sigma, ratio to Gaussian, kurtosis
    n_beyond, n_total, fourth_moment = 0, 0, 0.0
    for n in range(n_states):
        for k in range(n_draws):
            for t in range(8):
                for c in range(6):
                    n_total += 1; fourth_moment += z[n, k, t, c] ** 4
                    if magnitude[n, k, t, c] > 3: n_beyond += 1
    p_gt3 = n_beyond / n_total
    gauss_gt3 = 2 * norm.sf(3)
    kurtosis = fourth_moment / n_total - 3

    # step 4b: held-out NLL, Gaussian vs Student-t, each fitted on the other folds
    gauss_nll_per_fold, student_nll_per_fold = [], []
    for f in folds:
        train_values, test_values = [], []
        for n in range(n_states):
            for k in range(n_draws):
                for t in range(8):
                    for c in range(6):
                        (test_values if fold[n] == f else train_values).append(z[n, k, t, c])
        train_values, test_values = np.array(train_values), np.array(test_values)
        gauss_sigma = np.sqrt(np.mean(train_values ** 2))
        nu, s = fit_student_t(train_values)
        gauss_nll_per_fold.append(np.mean(.5 * np.log(2 * np.pi * gauss_sigma ** 2) + test_values ** 2 / (2 * gauss_sigma ** 2)))
        student_nll_per_fold.append(np.mean(-t_logpdf(test_values, nu, s)))
    heldout_nll_gain = np.mean(gauss_nll_per_fold) - np.mean(student_nll_per_fold)
    return dict(p_gt3=float(p_gt3), gauss_gt3=float(gauss_gt3), excess=float(p_gt3 / gauss_gt3), kurtosis=float(kurtosis),
                heldout_nll_gain=float(heldout_nll_gain))

# ----------------------------------------------------------------------------------------------- one function per panel
# Each panel function forms ITS quantity (step 1) and hands it to the common tail computation (steps 2-4).
# `impl` selects tail_fast or tail_readable so every panel can be audited with the loop version.

def panel_hg(residual, sigma, fold, impl=tail_fast):
    """a: heteroscedastic-Gaussian head. residual [N, 8, 6]; sigma [N] = the head's own predicted scale per state."""
    return impl(residual[:, None].astype(np.float64), fold, own_scale=sigma.astype(np.float64))

def panel_mse(residual, fold, impl=tail_fast):
    """b: MSE head. residual [N, 8, 6]; no per-state scale (the homoscedastic assumption MSE makes)."""
    return impl(residual[:, None].astype(np.float64), fold, own_scale=None)

def panel_flow_fixed(samples, fold, impl=tail_fast, every=1):
    """c: Flow head, sample minus the mean of ITS OWN draws at that state, fixed (global) scale only.
    samples [N, K, 8, 6]; `every` thins the draws entering the pooled curve (the mean always uses all K draws)."""
    dev = samples.astype(np.float64) - samples.astype(np.float64).mean(axis=1, keepdims=True)
    return impl(dev[:, ::every], fold, own_scale=None)

def panel_flow_own(samples, fold, impl=tail_fast, every=1):
    """d: Flow head, sample minus the mean of its draws, divided by the Flow model's OWN conditional scale:
    the scalar RMS spread of ALL K draws at that state, measured in the RMS-normalized coordinate space."""
    dev = samples.astype(np.float64) - samples.astype(np.float64).mean(axis=1, keepdims=True)      # [N, K, 8, 6]
    coord_rms = np.sqrt(np.mean(dev ** 2, axis=(0, 1)))                                             # [8, 6] over all states, all draws
    own_scale = np.sqrt(np.mean((dev / coord_rms) ** 2, axis=(1, 2, 3)))                            # [N] over K draws x 48 coordinates
    return impl(dev[:, ::every], fold, own_scale=own_scale)

# ----------------------------------------------------------------------------------------------- data + checks
RAW = "analysis/paper/widowx_heterogeneous_scale/raw"

def load(pattern):
    parts = {}
    for f in sorted(glob.glob(pattern)):
        with np.load(f, allow_pickle=False) as npz:
            for k in npz.files:
                if k != "metadata": parts.setdefault(k, []).append(npz[k])
    d = {k: np.concatenate(v) for k, v in parts.items()}; o = np.lexsort((d["step"], d["episode"], d["task_id"]))
    return {k: v[o] for k, v in d.items()}

def make_folds(task, ep, k=5, seed=20260906):
    rng = np.random.default_rng(seed); fold = np.empty(len(task), int)
    for t in np.unique(task):
        for i, e in enumerate(rng.permutation(np.unique(ep[task == t]))): fold[ep == e] = i % k
    return fold

def check_equivalence(n_states=240, n_draws=8):
    """every panel, fast vs readable, on a subset (first n_states states, first n_draws draws of the K=16 Flow probe)"""
    mse, hg, flow = load(f"{RAW}/mse_rank*.npz"), load(f"{RAW}/hg_rank*.npz"), load(f"{RAW}/flow_rank*.npz")
    fold = make_folds(mse["task_id"], mse["episode"])[:n_states]; S = flow["samples"][:n_states, :n_draws]
    cases = [("a HG", lambda impl: panel_hg(hg["residual"][:n_states], hg["sigma"][:n_states], fold, impl)),
             ("b MSE", lambda impl: panel_mse(mse["residual"][:n_states], fold, impl)),
             ("c Flow fixed", lambda impl: panel_flow_fixed(S, fold, impl)),
             ("d Flow own", lambda impl: panel_flow_own(S, fold, impl))]
    for name, run in cases:
        a, b = run(tail_fast), run(tail_readable)
        for key in a:
            assert abs(a[key] - b[key]) <= 1e-9 * max(1.0, abs(a[key])), (name, key, a[key], b[key])
        print(f"equivalent  {name:14s} P(|z|>3) {100*a['p_gt3']:.3f}%  kurt {a['kurtosis']:.4f}  gain {a['heldout_nll_gain']:.5f}")

if __name__ == "__main__":
    check_equivalence()
    if not glob.glob(f"{RAW}_k1024/flow_rank*.npz"):
        print("full run skipped: raw_k1024/flow_rank*.npz not present yet"); raise SystemExit
    mse, hg, flow = load(f"{RAW}/mse_rank*.npz"), load(f"{RAW}/hg_rank*.npz"), load(f"{RAW}_k1024/flow_rank*.npz")
    fold = make_folds(mse["task_id"], mse["episode"])
    for name, r in [("a  HG residual / own sigma(x)", panel_hg(hg["residual"], hg["sigma"], fold)),
                    ("b  MSE residual, fixed scale", panel_mse(mse["residual"], fold)),
                    ("c  Flow dev, fixed scale", panel_flow_fixed(flow["samples"], fold, every=16)),
                    ("d  Flow dev / own conditional scale", panel_flow_own(flow["samples"], fold, every=16))]:
        print(f"{name:38s} P(|z|>3) {100*r['p_gt3']:.2f}% ({r['excess']:.1f}x Gaussian)  kurtosis {r['kurtosis']:.1f}  held-out t gain {r['heldout_nll_gain']:.3f} nat/dim")
