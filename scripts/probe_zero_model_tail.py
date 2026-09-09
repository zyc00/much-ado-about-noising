"""Zero-output diagnostic using exactly the v4 panel-b coordinate protocol.

The zero is in the policy's normalized action space, not physical actuator units.
No policy training or inference is required: residual = -target from saved probes.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from tail_stats_single import load, make_folds, panel_mse, fit_student_t


def standardize(residual, fold):
    x = residual.reshape(len(residual), 1, 48).astype(np.float64)
    z = np.empty_like(x)
    for f in np.unique(fold):
        rms = np.sqrt(np.mean(x[fold != f].reshape(-1, 48) ** 2, axis=0))
        assert np.all(rms > 0)
        z[fold == f] = x[fold == f] / rms
    return z / np.sqrt(np.mean(z ** 2))


def fit_tail_v4(z):
    # Exact grid, interval, and objective from compute_longtail_curves.tail_fit.
    thresholds = np.linspace(1.0, 5.0, 41)
    empirical = np.array([(np.abs(z) > x).mean() for x in thresholds])
    keep = empirical > 0
    best = (None, None, np.inf)
    for nu in np.logspace(np.log10(1.5), np.log10(300), 100):
        for scale in np.linspace(0.55, 1.3, 51):
            fitted = 2 * student_t.sf(thresholds[keep] / scale, nu)
            error = np.mean((np.log(empirical[keep]) - np.log(fitted)) ** 2)
            if error < best[2]:
                best = (float(nu), float(scale), float(error))
    return dict(tailfit_nu=best[0], tailfit_scale=best[1], tail_log_mse=best[2])


def analyze(residual, fold):
    stats = panel_mse(residual, fold)
    z = standardize(residual, fold)
    stats.update(fit_tail_v4(z.ravel()))
    stats["tailfit_hits_upper_nu_bound"] = bool(np.isclose(stats["tailfit_nu"], 300))
    stats["max_abs_z"] = float(np.max(np.abs(z)))
    stats["tail_fractions"] = {
        str(t): float(np.mean(np.abs(z) > t)) for t in (1, 2, 3, 4, 5, 6)
    }
    nu, scale = fit_student_t(z.ravel())
    stats.update(mle_nu=float(nu), mle_scale=float(scale))
    stats["mle_search"] = "same v4 coarse grid: 23 nu values from 10**0.1 to 10**2.3"
    stats["mle_nu_by_training_fold"] = [
        float(fit_student_t(z[fold != f].ravel())[0]) for f in np.unique(fold)
    ]
    stats["standardized_mean"] = float(z.mean())
    return stats


def main():
    raw = Path("analysis/paper/widowx_heterogeneous_scale/raw")
    mse = load(str(raw / "mse_rank*.npz"))
    fold = make_folds(mse["task_id"], mse["episode"])
    zero = analyze(-mse["target"].astype(np.float64), fold)
    reference = analyze(mse["residual"].astype(np.float64), fold)
    saved = json.loads(Path("analysis/paper/longtail_motivation/longtail_curves.json").read_text())["b_mse"]
    for key in ("p_gt3", "tailfit_nu", "tailfit_scale"):
        np.testing.assert_allclose(reference[key], saved[key], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reference["heldout_nll_gain"], saved["heldout_gauss"] - saved["heldout_student"], atol=1e-10)
    result = {
        "definition": "f(o)=0 in saved model-normalized action space; residual=-target",
        "data": str(raw), "states": len(fold), "coordinates": int(mse["target"].size),
        "channels": "six continuous channels, gripper excluded, eight-step chunks",
        "protocol": "v4 panel b: same five episode folds, coordinate RMS from other folds, pooled unit RMS; fixed zero location",
        "v4_mse_reproduction": "passed", "zero_model": zero, "trained_mse": reference,
    }
    out = Path("analysis/paper/longtail_motivation/zero_model_tail_summary.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
