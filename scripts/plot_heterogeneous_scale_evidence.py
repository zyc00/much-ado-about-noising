#!/usr/bin/env python3
"""ICLR-width evidence chain for input-dependent continuous-action scale.

Panels (a--c) use exactly matched held-out GR1 states.  Panel (d) uses the
larger frozen-MSE audit and cross-fits a variance predictor across disjoint
episode halves while keeping the action mean fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.ensemble import ExtraTreesRegressor


BLUE = "#2C7BB6"
ORANGE = "#D55E00"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D0D0D0"
TASK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
SEED = 20260907


def configure_plot() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_parts(raw_dir: Path) -> dict[str, np.ndarray]:
    paths = sorted(raw_dir.glob("gr1_k32_rank*.npz"))
    if len(paths) != 4:
        raise FileNotFoundError(f"Expected four GR1 K=32 parts in {raw_dir}; found {len(paths)}")
    parts: dict[str, list[np.ndarray]] = {}
    metadata = []
    for path in paths:
        with np.load(path, allow_pickle=True) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["flow_draws_per_state"] != 32:
                raise ValueError(f"{path} does not contain the K=32 probe")
            metadata.append(meta)
            for key in archive.files:
                if key != "metadata":
                    parts.setdefault(key, []).append(archive[key])
    merged = {key: np.concatenate(values) for key, values in parts.items()}
    order = np.lexsort((merged["progress"], merged["task_id"]))
    merged = {key: value[order] for key, value in merged.items()}
    if len(merged["task_id"]) != 240 or len(np.unique(merged["task_id"])) != 24:
        raise ValueError("Matched probe must contain 10 states for each of 24 tasks")
    return merged


def task_relative(values: np.ndarray, task: np.ndarray) -> np.ndarray:
    """Divide positive values by their within-task geometric mean."""
    values = np.asarray(values, dtype=float)
    relative = np.empty_like(values)
    for task_id in np.unique(task):
        keep = task == task_id
        center = np.exp(np.mean(np.log(values[keep])))
        relative[keep] = values[keep] / center
    return relative


def within_task_spearman(x: np.ndarray, y: np.ndarray, task: np.ndarray) -> tuple[float, float]:
    ranked_x = np.empty(len(x), dtype=float)
    ranked_y = np.empty(len(y), dtype=float)
    for task_id in np.unique(task):
        keep = task == task_id
        ranked_x[keep] = rankdata(x[keep])
        ranked_y[keep] = rankdata(y[keep])
    result = spearmanr(ranked_x, ranked_y)
    return float(result.statistic), float(result.pvalue)


def within_task_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    task: np.ndarray,
    draws: int = 10000,
    seed_offset: int = 0,
) -> float:
    """Two-sided association test that shuffles labels only within tasks."""
    groups = [np.flatnonzero(task == task_id) for task_id in np.unique(task)]
    ranked_x = np.empty(len(x), dtype=float)
    ranked_y = np.empty(len(y), dtype=float)
    for indices in groups:
        ranked_x[indices] = rankdata(x[indices])
        ranked_y[indices] = rankdata(y[indices])
    ranked_x -= ranked_x.mean()
    ranked_y -= ranked_y.mean()
    denominator = np.sqrt(np.dot(ranked_x, ranked_x) * np.dot(ranked_y, ranked_y))
    observed = np.dot(ranked_x, ranked_y) / denominator
    rng = np.random.default_rng(SEED + 10 + seed_offset)
    exceed = 0
    permuted = np.empty_like(ranked_y)
    for _ in range(draws):
        for indices in groups:
            permuted[indices] = ranked_y[rng.permutation(indices)]
        candidate = np.dot(ranked_x, permuted) / denominator
        exceed += abs(candidate) >= abs(observed)
    return float((exceed + 1) / (draws + 1))


def quantile_curve(
    x: np.ndarray,
    y: np.ndarray,
    task: np.ndarray,
    bins: int = 6,
    draws: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    group = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    tasks = np.unique(task)

    def summarize(selected_tasks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for b in range(bins):
            xv, yv = [], []
            for task_id in selected_tasks:
                keep = (task == task_id) & (group == b)
                if keep.any():
                    xv.append(np.exp(np.mean(np.log(x[keep]))))
                    yv.append(np.exp(np.mean(np.log(y[keep]))))
            xs.append(np.mean(xv))
            ys.append(np.mean(yv))
        return np.asarray(xs), np.asarray(ys)

    centers, curve = summarize(tasks)
    rng = np.random.default_rng(SEED)
    boot = np.empty((draws, bins), dtype=float)
    for draw in range(draws):
        _, boot[draw] = summarize(tasks[rng.integers(len(tasks), size=len(tasks))])
    return centers, curve, np.quantile(boot, [0.025, 0.975], axis=0)


def progress_curve(
    y: np.ndarray,
    task: np.ndarray,
    progress: np.ndarray,
    draws: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    stage = np.minimum(9, (10 * progress).astype(int))
    tasks = np.unique(task)

    def summarize(selected_tasks: np.ndarray) -> np.ndarray:
        task_curves = []
        for task_id in selected_tasks:
            keep_task = task == task_id
            row = [np.exp(np.mean(np.log(y[keep_task & (stage == b)]))) for b in range(10)]
            task_curves.append(row)
        return np.mean(task_curves, axis=0)

    curve = summarize(tasks)
    rng = np.random.default_rng(SEED + 1)
    boot = np.asarray(
        [summarize(tasks[rng.integers(len(tasks), size=len(tasks))]) for _ in range(draws)]
    )
    return curve, np.quantile(boot, [0.025, 0.975], axis=0)


def progress_permutation(
    y: np.ndarray, task: np.ndarray, progress: np.ndarray, draws: int = 20000
) -> tuple[float, float, float]:
    """Test a nonparametric ten-stage effect after removing task intercepts."""
    stage = np.minimum(9, (10 * progress).astype(int))
    centered = np.empty(len(y), dtype=float)
    log_y = np.log(y)
    for task_id in np.unique(task):
        keep = task == task_id
        centered[keep] = log_y[keep] - np.mean(log_y[keep])
    stage_mean = np.asarray([np.mean(centered[stage == b]) for b in range(10)])
    statistic = float(np.var(stage_mean))
    fitted = stage_mean[stage]
    r2 = float(1 - np.sum((centered - fitted) ** 2) / np.sum(centered**2))
    rng = np.random.default_rng(SEED + 2)
    exceed = 0
    for _ in range(draws):
        permuted = np.empty_like(centered)
        for task_id in np.unique(task):
            indices = np.flatnonzero(task == task_id)
            permuted[indices] = centered[rng.permutation(indices)]
        candidate = np.var([np.mean(permuted[stage == b]) for b in range(10)])
        exceed += candidate >= statistic
    pvalue = (exceed + 1) / (draws + 1)
    radius_ratio = float(np.exp(stage_mean.max() - stage_mean.min()))
    return pvalue, r2, radius_ratio


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def task_progress_audit(
    y: np.ndarray,
    task: np.ndarray,
    progress: np.ndarray,
    draws: int = 10000,
) -> dict[str, np.ndarray | float | int]:
    """Estimate and test a separate ten-stage residual curve for every task."""
    tasks = np.unique(task)
    stage = np.minimum(9, (10 * progress).astype(int))
    curves = np.empty((len(tasks), 10), dtype=float)
    ranges = np.empty(len(tasks), dtype=float)
    pvalues = np.empty(len(tasks), dtype=float)
    r2 = np.empty(len(tasks), dtype=float)
    for row, task_id in enumerate(tasks):
        keep = task == task_id
        curves[row] = [
            np.exp(np.mean(np.log(y[keep & (stage == stage_id)])))
            for stage_id in range(10)
        ]
        pvalues[row], r2[row], ranges[row] = progress_permutation(
            y[keep],
            np.zeros(np.sum(keep), dtype=int),
            progress[keep],
            draws=draws,
        )
    qvalues = benjamini_hochberg(pvalues)
    return {
        "curves": curves,
        "ranges": ranges,
        "pvalues": pvalues,
        "qvalues": qvalues,
        "significant": qvalues < 0.05,
        "median_range": float(np.median(ranges)),
        "uncorrected_significant": int(np.sum(pvalues < 0.05)),
        "fdr_significant": int(np.sum(qvalues < 0.05)),
    }


def load_full_mse(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data["metadata"]))
    start = int(metadata["executed_start"])
    stop = start + int(metadata["executed_horizon"])
    channels = metadata["continuous_channels"]
    residual = (
        data["pred_final"][:, start:stop, channels]
        - data["gt"][:, start:stop, channels]
    ).astype(np.float64)
    if not data["valid_time"][:, start:stop].all():
        raise ValueError("A padded action entered the full MSE audit")
    data["energy"] = np.mean(residual**2, axis=(1, 2))
    return data


def crossfit_gaussian_likelihood(data: dict[str, np.ndarray]) -> dict:
    """Compare one variance per task with an input-conditioned variance.

    The action mean is frozen. The scale model sees proprioception, normalized
    episode progress, and the task-specific training-fold baseline. No query
    action or query residual enters its prediction.
    """
    energy = data["energy"]
    state = data["state"].reshape(len(energy), -1).astype(np.float64)
    task_names = np.unique(data["task"])
    task = np.searchsorted(task_names, data["task"])
    split = data["split"].astype(int)
    progress = data["step"] / np.maximum(data["length"] - 1, 1)
    features = np.column_stack((state, progress, progress**2, progress**3))
    ll_fixed = np.full(len(energy), np.nan)
    ll_hetero = np.full(len(energy), np.nan)

    for fold in (0, 1):
        train = split == fold
        test = ~train
        baseline = np.asarray(
            [np.mean(energy[train & (task == task_id)]) for task_id in task], dtype=float
        )
        target = np.log(energy / baseline)
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=20,
            max_features=0.5,
            n_jobs=-1,
            random_state=SEED + fold,
        )
        model.fit(features[train], target[train])
        log_relative_variance = model.predict(features)
        # One scalar calibration is the Gaussian MLE on the fitting fold.
        calibration = np.mean(
            energy[train]
            / (baseline[train] * np.exp(log_relative_variance[train]))
        )
        hetero_variance = baseline * calibration * np.exp(log_relative_variance)
        ll_fixed[test] = -0.5 * (
            np.log(2 * np.pi) + np.log(baseline[test]) + energy[test] / baseline[test]
        )
        ll_hetero[test] = -0.5 * (
            np.log(2 * np.pi)
            + np.log(hetero_variance[test])
            + energy[test] / hetero_variance[test]
        )

    if not np.isfinite(ll_fixed).all() or not np.isfinite(ll_hetero).all():
        raise ValueError("Cross-fitted likelihood contains missing values")
    task_fixed = np.asarray([np.mean(ll_fixed[task == i]) for i in range(len(task_names))])
    task_hetero = np.asarray([np.mean(ll_hetero[task == i]) for i in range(len(task_names))])
    task_gain = task_hetero - task_fixed
    rng = np.random.default_rng(SEED + 3)
    indices = rng.integers(len(task_names), size=(30000, len(task_names)))
    boot_fixed = task_fixed[indices].mean(axis=1)
    boot_hetero = task_hetero[indices].mean(axis=1)
    boot_gain = task_gain[indices].mean(axis=1)
    return {
        "fixed": float(task_fixed.mean()),
        "hetero": float(task_hetero.mean()),
        "fixed_ci": np.quantile(boot_fixed, [0.025, 0.975]).tolist(),
        "hetero_ci": np.quantile(boot_hetero, [0.025, 0.975]).tolist(),
        "gain": float(task_gain.mean()),
        "gain_ci": np.quantile(boot_gain, [0.025, 0.975]).tolist(),
        "positive_tasks": int(np.sum(task_gain > 0)),
        "tasks": int(len(task_names)),
        "samples": int(len(energy)),
        "dimensions": int(np.prod(residual_shape(data))),
        "task_fixed": task_fixed.tolist(),
        "task_hetero": task_hetero.tolist(),
        "task_gain": task_gain.tolist(),
    }


def residual_shape(data: dict[str, np.ndarray]) -> tuple[int, int]:
    metadata = json.loads(str(data["metadata"]))
    return int(metadata["executed_horizon"]), len(metadata["continuous_channels"])


def full_mse_diagnostics(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct the full held-out residual/label audit used in panels a--b."""
    metadata = json.loads(str(data["metadata"]))
    start = int(metadata["executed_start"])
    stop = start + int(metadata["executed_horizon"])
    channels = metadata["continuous_channels"]
    label = np.sqrt(
        np.mean(data["gt"][:, start:stop, channels].astype(np.float64) ** 2, axis=(1, 2))
    )
    residual = np.sqrt(data["energy"])
    task_names = np.unique(data["task"])
    task = np.searchsorted(task_names, data["task"])
    progress = data["step"] / np.maximum(data["length"] - 1, 1)
    return {
        "label": task_relative(label, task),
        "residual": task_relative(residual, task),
        "task": task,
        "task_names": task_names,
        "progress": progress,
    }


def draw(
    matched: dict[str, np.ndarray],
    full_mse: dict[str, np.ndarray],
    likelihood: dict,
    output: Path,
) -> None:
    configure_plot()
    task = matched["task_id"].astype(int)
    label = task_relative(matched["label_scale"], task)
    mse = task_relative(matched["mse_residual"], task)
    flow = task_relative(matched["flow_radius"], task)
    flow_error = task_relative(matched["flow_residual"], task)
    progress = matched["progress"]
    full = full_mse_diagnostics(full_mse)

    mse_label_rho, mse_label_p = within_task_spearman(
        full["label"], full["residual"], full["task"]
    )
    mse_label_perm_p = within_task_permutation_p(
        full["label"], full["residual"], full["task"], seed_offset=1
    )
    flow_label_rho, flow_label_p = within_task_spearman(label, flow, task)
    flow_label_perm_p = within_task_permutation_p(
        label, flow, task, seed_offset=2
    )
    flow_error_label_rho, flow_error_label_p = within_task_spearman(label, flow_error, task)
    mse_progress_p, mse_progress_r2, mse_progress_range = progress_permutation(
        full["residual"], full["task"], full["progress"]
    )
    flow_progress_p, flow_progress_r2, flow_progress_range = progress_permutation(
        flow, task, progress
    )
    flow_error_progress_p, _, flow_error_progress_range = progress_permutation(
        flow_error, task, progress
    )

    x_curve, label_curve, label_ci = quantile_curve(
        full["label"], full["residual"], full["task"], bins=8
    )
    mse_stage_full, mse_stage_full_ci = progress_curve(
        full["residual"], full["task"], full["progress"]
    )
    per_task_progress = task_progress_audit(
        full["residual"], full["task"], full["progress"]
    )
    flow_x_curve, flow_label_curve, flow_label_ci = quantile_curve(label, flow, task)
    mse_stage, mse_stage_ci = progress_curve(mse, task, progress)
    flow_stage, flow_stage_ci = progress_curve(flow, task, progress)
    stage_profile_r, stage_profile_p = pearsonr(np.log(mse_stage), np.log(flow_stage))

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.15, 2.26),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.10, 1.04]},
    )
    axa, axb, axc, axd = axes
    rng = np.random.default_rng(SEED + 4)

    # (a) Residual scale versus label scale.
    sample = rng.choice(len(full["label"]), size=min(1800, len(full["label"])), replace=False)
    axa.scatter(
        full["label"][sample],
        full["residual"][sample],
        s=4.5,
        color=BLUE,
        alpha=0.075,
        linewidths=0,
        rasterized=True,
    )
    axa.fill_between(x_curve, label_ci[0], label_ci[1], color=BLUE, alpha=0.14, linewidth=0)
    axa.plot(x_curve, label_curve, "o-", color=BLUE, lw=1.6, ms=3.1, mec="white", mew=0.45)
    axa.axhline(1, color="0.55", lw=0.65, ls=(0, (3, 2)), zorder=0)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(0.76, 1.38)
    axa.set_ylim(0.12, 12)
    axa.set_xticks([0.8, 1.0, 1.25], ["0.8", "1", "1.25"])
    axa.xaxis.set_minor_locator(NullLocator())
    axa.yaxis.set_minor_locator(NullLocator())
    axa.set_yticks([0.2, 0.5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"])
    axa.set_xlabel("Task-normalized label RMS")
    axa.set_ylabel("Task-normalized residual RMS")
    axa.set_title("a  Residual follows label scale", loc="left", fontweight="bold")
    axa.text(
        0.04,
        0.95,
        rf"$\rho={mse_label_rho:.2f}$; perm. $p<10^{{-4}}$",
        transform=axa.transAxes,
        ha="left",
        va="top",
        color=BLUE,
        fontweight="bold",
    )

    # (b) Keep task-specific stage structure visible. Every gray curve is one
    # task; color marks exactly the tasks that survive BH-FDR correction.
    stage_x = (np.arange(10) + 0.5) / 10
    curves = np.asarray(per_task_progress["curves"])
    significant = np.asarray(per_task_progress["significant"])
    for curve in curves[~significant]:
        axb.plot(stage_x, curve, color="0.72", lw=0.55, alpha=0.40, zorder=1)
    for color, curve in zip(TASK_COLORS, curves[significant]):
        axb.plot(
            stage_x,
            curve,
            "o-",
            color=color,
            lw=1.25,
            ms=2.1,
            mec="white",
            mew=0.25,
            zorder=3,
        )
    axb.axhline(1, color="0.55", lw=0.65, ls=(0, (3, 2)), zorder=0)
    axb.set_yscale("log")
    axb.set_ylim(0.56, 2.05)
    axb.set_yticks([0.6, 1, 2], ["0.6", "1", "2"])
    axb.yaxis.set_minor_locator(NullLocator())
    axb.set_xlim(-0.03, 1.03)
    axb.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    axb.set_xlabel("Episode progress (%)")
    axb.set_title("b  Residual scale depends on stage", loc="left", fontweight="bold")
    axb.text(
        0.04,
        0.95,
        rf"{per_task_progress['fdr_significant']}/24 tasks: FDR $q<.05$"
        + "\n"
        + rf"median within-task range: {per_task_progress['median_range']:.2f}$\times$",
        transform=axb.transAxes,
        ha="left",
        va="top",
        color="0.20",
        fontweight="bold",
        fontsize=6.5,
        linespacing=1.00,
    )

    # (c) Main axis: Flow sampling spread versus label scale. Inset: the
    # episode-progress profiles of Flow spread and MSE residual on the exact
    # same states. This shows both requested sources of scale structure.
    axc.scatter(label, flow, s=7, color=ORANGE, alpha=0.15, linewidths=0)
    axc.fill_between(
        flow_x_curve,
        flow_label_ci[0],
        flow_label_ci[1],
        color=ORANGE,
        alpha=0.14,
        linewidth=0,
    )
    axc.plot(
        flow_x_curve,
        flow_label_curve,
        "o-",
        color=ORANGE,
        lw=1.6,
        ms=3.0,
        mec="white",
        mew=0.45,
    )
    axc.axhline(1, color="0.55", lw=0.65, ls=(0, (3, 2)), zorder=0)
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlim(0.76, 1.38)
    axc.set_ylim(0.12, 12)
    axc.set_xticks([0.8, 1.0, 1.25], ["0.8", "1", "1.25"])
    axc.xaxis.set_minor_locator(NullLocator())
    axc.set_yticks([0.2, 0.5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"])
    axc.yaxis.set_minor_locator(NullLocator())
    axc.set_xlabel("Task-normalized label RMS")
    axc.set_ylabel("Task-normalized Flow spread")
    axc.set_title("c  Flow reflects the same scale", loc="left", fontweight="bold")
    axc.text(
        0.04,
        0.95,
        rf"label scale: $\rho={flow_label_rho:.2f}$; $p<.005$",
        transform=axc.transAxes,
        ha="left",
        va="top",
        color=ORANGE,
        fontweight="bold",
    )

    inset = axc.inset_axes([0.48, 0.13, 0.50, 0.31])
    inset.set_facecolor((1, 1, 1, 0.90))
    inset.plot(stage_x, mse_stage, "o-", color=BLUE, lw=0.9, ms=1.7)
    inset.plot(stage_x, flow_stage, "o-", color=ORANGE, lw=1.15, ms=1.9)
    inset.set_yscale("log")
    inset.set_xlim(0, 1)
    inset.set_xticks([0, 1], ["0%", "100%"])
    inset.set_yticks([])
    inset.yaxis.set_minor_locator(NullLocator())
    inset.tick_params(length=1.8, pad=1, labelsize=5.5)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.text(
        0.02,
        0.96,
        "progress profiles"
        + "\n"
        + rf"Flow--MSE $r={stage_profile_r:.2f}$",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=4.9,
        fontweight="bold",
        linespacing=0.92,
    )

    # (d) Paired held-out likelihood improvements. Each task's uniform-scale
    # Gaussian is its own zero, making the within-task comparison explicit.
    task_gain = np.asarray(likelihood["task_gain"])
    order = np.argsort(task_gain)
    for j, gain in enumerate(task_gain[order]):
        color = ORANGE if gain > 0 else "0.62"
        axd.plot([0, 1], [0, gain], color=color, lw=0.65, alpha=0.48, zorder=1)
        axd.scatter([1], [gain], s=7, color=color, edgecolor="white", linewidth=0.25, zorder=2)
    axd.axhline(0, color="0.25", lw=0.75, ls=(0, (3, 2)), zorder=0)
    gain_lo, gain_hi = likelihood["gain_ci"]
    axd.errorbar(
        [1.11],
        [likelihood["gain"]],
        yerr=[[likelihood["gain"] - gain_lo], [gain_hi - likelihood["gain"]]],
        fmt="D",
        ms=3.5,
        color=ORANGE,
        ecolor=ORANGE,
        capsize=2.0,
        lw=1.15,
        zorder=4,
    )
    axd.set_xlim(-0.08, 1.22)
    axd.set_ylim(min(-0.17, task_gain.min() - 0.01), max(0.125, task_gain.max() + 0.01))
    axd.set_xticks([0, 1], ["Uniform $\sigma$\nwithin task", "Hetero. $\sigma(o)$"])
    axd.set_ylabel("$\Delta$ held-out log likelihood\n(nat / action dim)")
    axd.set_title(
        "d  Heterogeneous Gaussian fits better",
        loc="left",
        fontweight="bold",
        fontsize=7.7,
    )
    axd.text(
        0.03,
        0.96,
        rf"{likelihood['positive_tasks']}/{likelihood['tasks']} tasks improve"
        + "\n"
        + rf"$\Delta$LL $=+{likelihood['gain']:.3f}$"
        + rf" $[{gain_lo:.3f},{gain_hi:.3f}]$",
        transform=axd.transAxes,
        ha="left",
        va="top",
        color=ORANGE,
        fontweight="bold",
        linespacing=1.05,
        fontsize=6.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.6},
    )

    for ax in axes:
        ax.grid(axis="y", color="0.91", linewidth=0.55, zorder=0)
        ax.tick_params(length=2.5, pad=2)
    fig.text(
        0.5,
        0.995,
        "GR00T N1.7 / GR1  ·  24 RoboCasa tasks  ·  continuous action channels",
        ha="center",
        va="top",
        fontsize=7.6,
        color="0.32",
    )
    fig.subplots_adjust(left=0.061, right=0.995, bottom=0.235, top=0.82, wspace=0.39)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=320)
    plt.close(fig)

    summary = {
        "full_mse_audit": {
            "states": int(len(full["label"])),
            "tasks": int(len(np.unique(full["task"]))),
            "heldout_sampling": "120 policy-held-out states per task",
            "continuous_channels": 29,
            "mse_residual_vs_label_scale": {
                "within_task_spearman": mse_label_rho,
                "analytic_p": mse_label_p,
                "within_task_permutation_p": mse_label_perm_p,
            },
            "mse_residual_vs_progress": {
                "stage_range": mse_progress_range,
                "permutation_p": mse_progress_p,
                "task_adjusted_r2": mse_progress_r2,
                "median_within_task_stage_range": per_task_progress["median_range"],
                "uncorrected_significant_tasks": per_task_progress[
                    "uncorrected_significant"
                ],
                "fdr_significant_tasks": per_task_progress["fdr_significant"],
                "task_pvalues": np.asarray(per_task_progress["pvalues"]).tolist(),
                "task_qvalues": np.asarray(per_task_progress["qvalues"]).tolist(),
                "fdr_significant_task_names": np.asarray(full["task_names"])[
                    np.asarray(per_task_progress["significant"])
                ].tolist(),
            },
        },
        "matched_flow_probe": {
            "states": int(len(task)),
            "tasks": int(len(np.unique(task))),
            "heldout_sampling": "one policy-held-out state per task and episode-progress decile",
            "flow_samples_per_state": 32,
            "continuous_channels": 29,
            "gripper_handling": "GR1 hand joints retained because they are continuous joint commands",
            "flow_spread_vs_label_scale": {
                "within_task_spearman": flow_label_rho,
                "analytic_p": flow_label_p,
                "within_task_permutation_p": flow_label_perm_p,
            },
            "flow_spread_vs_progress": {
                "stage_range": flow_progress_range,
                "permutation_p": flow_progress_p,
                "task_adjusted_r2": flow_progress_r2,
            },
            "mse_flow_stage_profile": {
                "pearson_r": float(stage_profile_r),
                "p": float(stage_profile_p),
            },
            "flow_mean_to_label_rms": {
                "vs_label_scale_within_task_spearman": flow_error_label_rho,
                "vs_label_scale_p": flow_error_label_p,
                "progress_stage_range": flow_error_progress_range,
                "progress_permutation_p": flow_error_progress_p,
            },
        },
        "gaussian_goodness_of_fit": likelihood,
        "interpretation": (
            "Unequal held-out residual scales and higher cross-fitted likelihood support an "
            "input-dependent residual scale. Flow spread is corroborating evidence, not used "
            "to define the MSE scale groups. These diagnostics do not by themselves isolate "
            "irreducible demonstration noise from mean-model error."
        ),
    }
    output.with_name(output.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("analysis/paper/heterogeneous_scale_probe/raw"),
    )
    parser.add_argument(
        "--full-mse",
        type=Path,
        default=Path("analysis/paper/mse_scale/gr1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper/heterogeneous_scale_probe/fig_heterogeneous_scale"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched = load_parts(args.raw)
    full = load_full_mse(args.full_mse)
    likelihood = crossfit_gaussian_likelihood(full)
    draw(matched, full, likelihood, args.output)
    print(json.dumps(likelihood, indent=2))


if __name__ == "__main__":
    main()
