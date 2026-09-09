#!/usr/bin/env python3
"""Held-out shape audit for same-observation stochastic action samples.

This analysis asks a narrower question than the existing multimodality probe:
after removing discrete gripper/hand commands, are continuous arm chunks best
described by one Gaussian basin, one heavy-tailed basin, or two Gaussian
components?  It uses held-out log likelihood rather than in-sample BIC alone.

Two complementary axis families are used.

* Fixed random axes are chosen without looking at a state's samples.  Four-fold
  cross-validation compares Normal, Student-t, and two-Gaussian densities.
* A targeted PC1 stress test learns the maximum-variance direction from an
  independent discovery split and evaluates density shape only on untouched
  samples.  WidowX has an explicit 32-draw discovery / 64-draw confirmation
  split; GR1 and pi0.5 use both directions of a 16/16 split.

The script writes a machine-readable JSON report and a compact paper-oriented
figure.  It does not interpret failure to reject as proof of exact Gaussianity.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import kurtosis, shapiro, skew


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "analysis/paper/gaussianity"


@dataclass
class SampleSet:
    name: str
    short: str
    sample: np.ndarray
    discovery: np.ndarray | None
    selection: str


def normal_logpdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    variance = max(float(variance), 1e-6)
    return -0.5 * (math.log(2.0 * math.pi * variance) + (x - mean) ** 2 / variance)


def student_logpdf(
    x: np.ndarray, df: float, location: float, scale: float
) -> np.ndarray:
    scale = max(float(scale), 1e-6)
    standardized = (x - location) / scale
    return (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
        - (df + 1.0) / 2.0 * np.log1p(standardized**2 / df)
    )


def fit_student(z: np.ndarray) -> tuple[float, float, float]:
    """Fast robust grid fit; enough for an out-of-sample shape comparator."""
    z = np.asarray(z, dtype=np.float64)
    location = float(np.median(z))
    mad = float(np.median(np.abs(z - location)))
    if mad < 1e-4:
        mad = max(float(z.std()), 1e-3) * 0.67449
    best: tuple[float, float, float, float] | None = None
    for df in (2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 100.0):
        # q_0.75 for t(df), accurately approximated over this range by the
        # tabulated normal quantile plus its first two asymptotic corrections.
        q = 0.67448975 + (0.67448975**3 + 0.67448975) / (4.0 * df)
        q += (5.0 * 0.67448975**5 + 16.0 * 0.67448975**3 + 3.0 * 0.67448975) / (
            96.0 * df**2
        )
        base_scale = max(mad / q, 1e-3)
        # A small scale grid substantially improves likelihood while keeping
        # the fit deterministic and cheap enough for thousands of CV folds.
        for multiplier in (0.75, 0.9, 1.0, 1.1, 1.3):
            scale = base_scale * multiplier
            ll = float(student_logpdf(z, df, location, scale).sum())
            if best is None or ll > best[0]:
                best = (ll, df, location, scale)
    assert best is not None
    return best[1], best[2], best[3]


def fit_gmm2(z: np.ndarray) -> dict[str, Any]:
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    total_variance = max(float(z.var()), 1e-6)
    floor = max(0.02 * total_variance, 1e-4)
    best = None
    for quantiles in ((0.2, 0.8), (0.3, 0.7), (0.4, 0.6)):
        means = np.quantile(z, quantiles).astype(np.float64)
        variances = np.full(2, total_variance * 0.6, dtype=np.float64)
        weights = np.full(2, 0.5, dtype=np.float64)
        old_ll = -np.inf
        for _ in range(100):
            logp = np.column_stack(
                [
                    math.log(max(weights[j], 1e-10))
                    + normal_logpdf(z, float(means[j]), float(variances[j]))
                    for j in range(2)
                ]
            )
            log_norm = logsumexp(logp, axis=1)
            responsibilities = np.exp(logp - log_norm[:, None])
            ll = float(log_norm.sum())
            counts = responsibilities.sum(axis=0).clip(min=1e-6)
            weights = counts / n
            means = (responsibilities * z[:, None]).sum(axis=0) / counts
            variances = (
                responsibilities * (z[:, None] - means) ** 2
            ).sum(axis=0) / counts
            variances = np.maximum(variances, floor)
            if abs(ll - old_ll) < 1e-7:
                break
            old_ll = ll
        if best is None or ll > best[0]:
            best = (ll, weights.copy(), means.copy(), variances.copy())
    assert best is not None
    _, weights, means, variances = best
    order = np.argsort(means)
    return {
        "weights": weights[order],
        "means": means[order],
        "variances": variances[order],
    }


def gmm_logpdf(x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    return logsumexp(
        np.column_stack(
            [
                math.log(max(float(fit["weights"][j]), 1e-10))
                + normal_logpdf(
                    x, float(fit["means"][j]), float(fit["variances"][j])
                )
                for j in range(2)
            ]
        ),
        axis=1,
    )


def heldout_scores(z_train: np.ndarray, z_test: np.ndarray) -> dict[str, float]:
    location0 = float(z_train.mean())
    scale = max(float(z_train.std()), 1e-6)
    z_train = (z_train - location0) / scale
    z_test = (z_test - location0) / scale
    normal = float(normal_logpdf(z_test, float(z_train.mean()), float(z_train.var())).mean())
    df, location, t_scale = fit_student(z_train)
    student = float(student_logpdf(z_test, df, location, t_scale).mean())
    mixture_fit = fit_gmm2(z_train)
    mixture = float(gmm_logpdf(z_test, mixture_fit).mean())
    return {
        "normal": normal,
        "student": student,
        "mixture": mixture,
        "student_minus_normal": student - normal,
        "mixture_minus_normal": mixture - normal,
        "mixture_minus_student": mixture - student,
        "student_df": float(df),
        "mixture_min_weight": float(np.min(mixture_fit["weights"])),
    }


def cv_projection(z: np.ndarray, folds: int = 4) -> dict[str, float]:
    z = np.asarray(z, dtype=np.float64)
    # Stable order-independent folds: sorting a fixed random permutation is
    # unnecessary because action draws are already exchangeable.
    fold_ids = np.arange(len(z)) % folds
    rows = []
    for fold in range(folds):
        test = fold_ids == fold
        rows.append(heldout_scores(z[~test], z[test]))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def fixed_axes(dimension: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axes = rng.choice((-1.0, 1.0), size=(count, dimension))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes


def top_axis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = x.mean(axis=0)
    _, singular, vh = np.linalg.svd(x - center, full_matrices=False)
    if not len(singular) or singular[0] < 1e-10:
        axis = np.zeros(x.shape[1], dtype=np.float64)
        axis[0] = 1.0
    else:
        axis = vh[0]
    return center, axis


def bh_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def normality_summary(
    state_p: list[float], projected_values: list[np.ndarray]
) -> dict[str, Any]:
    q = bh_adjust(state_p)
    skewness = np.asarray([skew(z, bias=False) for z in projected_values])
    excess = np.asarray([kurtosis(z, bias=False) for z in projected_values])
    return {
        "test": "Shapiro-Wilk; within-state axes Bonferroni corrected, then BH across states",
        "raw_p_le_0.05": int(np.sum(np.asarray(state_p) <= 0.05)),
        "bh_q_le_0.05": int(np.sum(q <= 0.05)),
        "n_states": len(state_p),
        "median_abs_skew": float(np.median(np.abs(skewness))),
        "median_excess_kurtosis": float(np.median(excess)),
    }


def model_summary(rows: list[dict[str, float]], state_ids: list[int]) -> dict[str, Any]:
    keys = (
        "student_minus_normal",
        "mixture_minus_normal",
        "mixture_minus_student",
        "student_df",
    )
    result: dict[str, Any] = {"n_rows": len(rows), "n_states": len(set(state_ids))}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "fraction_positive": float(np.mean(values > 0)),
        }
    # State is the independent unit.  Aggregate axes/split directions within
    # each state before reporting which model wins.
    state_values = []
    for state in sorted(set(state_ids)):
        indices = [i for i, value in enumerate(state_ids) if value == state]
        state_values.append(
            [
                np.mean([rows[i]["student_minus_normal"] for i in indices]),
                np.mean([rows[i]["mixture_minus_normal"] for i in indices]),
                np.mean([rows[i]["mixture_minus_student"] for i in indices]),
            ]
        )
    state_values_array = np.asarray(state_values)
    result["state_fraction_student_beats_normal"] = float(
        np.mean(state_values_array[:, 0] > 0)
    )
    result["state_fraction_mixture_beats_normal"] = float(
        np.mean(state_values_array[:, 1] > 0)
    )
    result["state_fraction_mixture_beats_student"] = float(
        np.mean(state_values_array[:, 2] > 0)
    )
    result["state_deltas"] = state_values_array.tolist()
    return result


def audit_sample_set(sample_set: SampleSet, n_axes: int) -> dict[str, Any]:
    sample = np.asarray(sample_set.sample, dtype=np.float64)
    n_states, n_samples, dimension = sample.shape
    axes = fixed_axes(dimension, n_axes, seed=991 + dimension)
    fixed_rows: list[dict[str, float]] = []
    fixed_states: list[int] = []
    fixed_p: list[float] = []
    fixed_projected: list[np.ndarray] = []
    for state in range(n_states):
        state_p = []
        for axis in axes:
            projected = sample[state] @ axis
            fixed_rows.append(cv_projection(projected))
            fixed_states.append(state)
            state_p.append(float(shapiro(projected).pvalue))
            fixed_projected.append(projected)
        fixed_p.append(min(1.0, min(state_p) * len(state_p)))

    targeted_rows: list[dict[str, float]] = []
    targeted_states: list[int] = []
    targeted_p: list[float] = []
    targeted_projected: list[np.ndarray] = []
    if sample_set.discovery is not None:
        discovery = np.asarray(sample_set.discovery, dtype=np.float64)
        for state in range(n_states):
            center, axis = top_axis(discovery[state])
            projected = (sample[state] - center) @ axis
            targeted_rows.append(cv_projection(projected))
            targeted_states.append(state)
            targeted_p.append(float(shapiro(projected).pvalue))
            targeted_projected.append(projected)
    else:
        # Both 16/16 directions make the axis independent of evaluated draws.
        midpoint = n_samples // 2
        for state in range(n_states):
            state_p = []
            for discover_slice, evaluate_slice in (
                (slice(0, midpoint), slice(midpoint, n_samples)),
                (slice(midpoint, n_samples), slice(0, midpoint)),
            ):
                center, axis = top_axis(sample[state, discover_slice])
                projected = (sample[state, evaluate_slice] - center) @ axis
                targeted_rows.append(cv_projection(projected))
                targeted_states.append(state)
                state_p.append(float(shapiro(projected).pvalue))
                targeted_projected.append(projected)
            targeted_p.append(min(1.0, min(state_p) * len(state_p)))

    return {
        "name": sample_set.name,
        "short": sample_set.short,
        "selection": sample_set.selection,
        "n_states": n_states,
        "samples_per_state": n_samples,
        "continuous_dimension": dimension,
        "fixed_random_axes": model_summary(fixed_rows, fixed_states),
        "independent_pc1": model_summary(targeted_rows, targeted_states),
        "fixed_random_axes_normality": normality_summary(fixed_p, fixed_projected),
        "independent_pc1_normality": normality_summary(targeted_p, targeted_projected),
    }


def widowx_known_mode_stratification(model: dict[str, Any]) -> dict[str, Any]:
    prefix_path = ROOT / "analysis/paper/widowx_mm/rollout_random_prefix4_v1.json"
    with prefix_path.open() as handle:
        prefix = json.load(handle)
    deltas = np.asarray(model["independent_pc1"]["state_deltas"], dtype=np.float64)
    result = {}
    for key in ("detected", "resolved"):
        mask = np.asarray([row["tests"]["arm"][key] for row in prefix["rows"]])
        groups = {}
        for name, selection in ((key, mask), (f"not_{key}", ~mask)):
            values = deltas[selection]
            groups[name] = {
                "n_states": int(selection.sum()),
                "mixture_minus_normal_mean": float(values[:, 1].mean()),
                "mixture_minus_normal_median": float(np.median(values[:, 1])),
                "mixture_beats_normal": int(np.sum(values[:, 1] > 0)),
                "mixture_minus_student_mean": float(values[:, 2].mean()),
                "mixture_minus_student_median": float(np.median(values[:, 2])),
            }
        result[key] = groups
    return result


def load_sets() -> list[SampleSet]:
    widow = np.load(ROOT / "analysis/paper/widowx_mm/rollout_random_v1.npz")
    widow_discovery = widow["discovery"][:, :, :4, :6].reshape(100, 32, -1)
    widow_confirm = widow["confirmation"][:, :, :4, :6].reshape(100, 64, -1)

    gr1 = np.load(DATA_DIR / "gr1_seek3k.npz")
    gr1_chunk = gr1["deep_raw"].reshape(-1, 32, 8, 29)
    gr1_body = np.concatenate(
        [gr1_chunk[:, :, :, :14], gr1_chunk[:, :, :, 26:29]], axis=-1
    ).reshape(-1, 32, 136)
    gr1_hands = gr1_chunk[:, :, :, 14:26].reshape(-1, 32, 96)
    gr1_all = gr1_chunk.reshape(-1, 32, 232)

    pi = np.load(DATA_DIR / "pi05_seek3k.npz")
    pi_raw = pi["deep_raw"].reshape(-1, 32, 50, 7)[:, :, :, :6].reshape(-1, 32, 300)

    return [
        SampleSet(
            "GR00T WidowX Flow",
            "WidowX",
            widow_confirm,
            widow_discovery,
            "uniform 100-state on-policy confirmation sample",
        ),
        SampleSet(
            "GR1 Flow (arm + waist)",
            "GR1 body",
            gr1_body,
            None,
            "body channels at 30 adversarial states retained from a 3,000-state full-action screen",
        ),
        SampleSet(
            "GR1 Flow (articulated hands)",
            "GR1 hands",
            gr1_hands,
            None,
            "hand channels at the same 30 adversarial states",
        ),
        SampleSet(
            "GR1 Flow (all joints)",
            "GR1 all",
            gr1_all,
            None,
            "all 29 channels at the same 30 adversarial states",
        ),
        SampleSet(
            "pi0.5 Flow",
            "pi0.5",
            pi_raw,
            None,
            "36 adversarial states retained from a 3,000-state screen",
        ),
    ]


def make_figure(report: dict[str, Any], output: Path) -> None:
    models = report["models"]
    labels = [model["short"] for model in models]
    x = np.arange(len(models))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), constrained_layout=True)
    for panel, family, title in (
        (axes[0], "fixed_random_axes", "fixed projections"),
        (axes[1], "independent_pc1", "independent PC1 stress test"),
    ):
        values = np.asarray(
            [
                [
                    model[family]["state_fraction_student_beats_normal"],
                    model[family]["state_fraction_mixture_beats_normal"],
                    model[family]["state_fraction_mixture_beats_student"],
                ]
                for model in models
            ]
        )
        for index, (name, color) in enumerate(
            (("t > N", "#4c78a8"), ("2G > N", "#f58518"), ("2G > t", "#b44e65"))
        ):
            panel.bar(x + (index - 1) * width, values[:, index], width, label=name, color=color)
        panel.set_xticks(x, labels)
        panel.set_ylim(0, 1)
        panel.set_ylabel("fraction of states")
        panel.set_title(title)
        panel.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].legend(frameon=False, ncols=3, fontsize=8, loc="upper center")
    fig.suptitle("Held-out density comparison on same-state action chunks", fontsize=11)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "action_basin_report.json")
    args = parser.parse_args()
    models = []
    for sample_set in load_sets():
        print(f"auditing {sample_set.name} ...", flush=True)
        models.append(audit_sample_set(sample_set, args.axes))
    report = {
        "question": (
            "After excluding discrete gripper/hand channels, do same-observation "
            "continuous action chunks require two modes beyond a one-basin density?"
        ),
        "protocol": {
            "models": "Normal vs robust Student-t vs two-Gaussian mixture",
            "criterion": "mean held-out log likelihood per scalar projected draw",
            "fixed_axes_per_state": args.axes,
            "targeted_axis": "PC1 learned from samples disjoint from all evaluated draws",
            "caveat": (
                "This is evidence about low-dimensional marginals and the strongest "
                "independently learned direction, not proof of exact multivariate Gaussianity."
            ),
        },
        "models": models,
        "widowx_known_mode_stratification": widowx_known_mode_stratification(models[0]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(report, handle, indent=2)
    make_figure(report, args.out.with_name("action_basin_heldout.png"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
