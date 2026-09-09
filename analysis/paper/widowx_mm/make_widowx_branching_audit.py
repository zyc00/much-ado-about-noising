#!/usr/bin/env python3
"""Aggregate and visualize same-state WidowX closed-loop branch experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
STEMS = ("widowx_branching_prefix_v1",)
COLORS = ("#2878B5", "#D95319")
WINDOWS = {
    "forced chunk": slice(1, 5),
    "middle": slice(17, 33),
    "late": slice(37, 49),
}
MIN_FORCED_POSITION_GAP_CM = 0.05
MIN_FORCED_ORIENTATION_GAP_DEG = 0.5


def load_cases() -> tuple[list[dict], list[dict]]:
    cases = []
    runs = []
    for stem in STEMS:
        json_path = HERE / f"{stem}.json"
        npz_path = HERE / f"{stem}.npz"
        if not json_path.exists() or not npz_path.exists():
            continue
        metadata = json.loads(json_path.read_text())
        runs.append(
            {
                "stem": stem,
                "n_screened": metadata["n_screened"],
                "n_deep_attempts": metadata["n_deep_attempts"],
                "n_accepted_cases": len(metadata["cases"]),
                "execution_horizon": metadata["execution_horizon"],
                "mode_test_horizon": metadata["mode_test_horizon"],
            }
        )
        if metadata["mode_test_horizon"] != metadata["execution_horizon"]:
            raise ValueError(
                f"{stem} fits modes over {metadata['mode_test_horizon']} steps "
                f"but executes {metadata['execution_horizon']}"
            )
        with np.load(npz_path) as arrays:
            for index, row in enumerate(metadata["cases"]):
                cases.append(
                    {
                        **row,
                        "source_stem": stem,
                        "rgb": arrays[f"case_{index}_rgb"].copy(),
                        "poses": arrays[f"case_{index}_poses"].copy(),
                        "labels": arrays[f"case_{index}_labels"].copy(),
                        "success": arrays[f"case_{index}_success"].copy(),
                        "task_metric": arrays[f"case_{index}_task_metric"].copy(),
                    }
                )
    if not cases:
        raise FileNotFoundError("No complete branching JSON/NPZ pairs found")
    return cases, runs


def loo_nearest_centroid(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    prediction = np.empty(len(labels), dtype=np.int8)
    for held_out in range(len(labels)):
        train = np.arange(len(labels)) != held_out
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (x - mean) / scale
        centroids = [standardized[train & (labels == mode)].mean(axis=0) for mode in (0, 1)]
        distances = [np.linalg.norm(standardized[held_out] - centroid) for centroid in centroids]
        prediction[held_out] = int(distances[1] < distances[0])
    return prediction


def case_predictions(case: dict, window: slice, labels: np.ndarray | None = None) -> np.ndarray:
    labels = case["labels"] if labels is None else labels
    poses = case["poses"]
    xyz_cm = (poses[..., :3] - poses[:, :1, :3]) * 100.0
    rpy_deg = np.rad2deg(np.unwrap(poses[..., 3:6], axis=1))
    rpy_deg = rpy_deg - rpy_deg[:, :1]
    features = np.concatenate((xyz_cm[:, window], rpy_deg[:, window]), axis=-1)
    features = features.reshape(len(labels), -1)
    return loo_nearest_centroid(features, labels)


def forced_pose_effect(case: dict) -> tuple[float, float, bool]:
    labels = case["labels"]
    poses = case["poses"]
    xyz_cm = poses[..., :3] * 100.0
    rpy_deg = np.rad2deg(np.unwrap(poses[..., 3:6], axis=1))
    means_xyz = np.stack([xyz_cm[labels == mode].mean(axis=0) for mode in (0, 1)])
    means_rpy = np.stack([rpy_deg[labels == mode].mean(axis=0) for mode in (0, 1)])
    forced = slice(1, 5)
    position_gap = float(np.linalg.norm(means_xyz[1] - means_xyz[0], axis=-1)[forced].max())
    orientation_gap = float(np.linalg.norm(means_rpy[1] - means_rpy[0], axis=-1)[forced].max())
    measurable = bool(
        position_gap >= MIN_FORCED_POSITION_GAP_CM
        or orientation_gap >= MIN_FORCED_ORIENTATION_GAP_DEG
    )
    return position_gap, orientation_gap, measurable


def spatial_gap_and_within(case: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Return Euclidean centroid gap and radial within-mode RMS, both in cm."""
    labels = case["labels"]
    xyz = case["poses"][..., :3]
    means = np.stack([xyz[labels == mode].mean(axis=0) for mode in (0, 1)])
    gap_cm = np.linalg.norm(means[1] - means[0], axis=-1) * 100.0
    within_cm = np.empty(len(gap_cm), dtype=np.float64)
    for step in range(len(gap_cm)):
        residuals = np.concatenate(
            [xyz[labels == mode, step] - means[mode, step] for mode in (0, 1)]
        )
        within_cm[step] = np.sqrt(np.mean(np.sum(residuals**2, axis=-1))) * 100.0
    late = slice(max(1, 3 * len(gap_cm) // 4), len(gap_cm))
    late_ratio = float(gap_cm[late].mean() / max(within_cm[late].mean(), 1e-12))
    return gap_cm, within_cm, late_ratio


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def endpoint_permutation_p(
    values: np.ndarray, labels: np.ndarray, seed: int, n_perm: int = 5000
) -> float:
    """Two-sided randomization test for the endpoint mean difference."""
    rng = np.random.default_rng(seed)
    observed = abs(float(values[labels == 1].mean() - values[labels == 0].mean()))
    exceed = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        statistic = abs(
            float(values[shuffled == 1].mean() - values[shuffled == 0].mean())
        )
        exceed += statistic >= observed
    return float((exceed + 1) / (n_perm + 1))


def fisher_exact_two_sided(success: np.ndarray, labels: np.ndarray) -> float:
    """Two-sided Fisher exact p-value for two balanced intervention groups."""
    successes = [int(success[labels == mode].sum()) for mode in (0, 1)]
    totals = [int((labels == mode).sum()) for mode in (0, 1)]
    total_success = sum(successes)
    denominator = math.comb(sum(totals), total_success)

    def probability(k: int) -> float:
        if not (0 <= k <= totals[0] and 0 <= total_success - k <= totals[1]):
            return 0.0
        return math.comb(totals[0], k) * math.comb(totals[1], total_success - k) / denominator

    observed_probability = probability(successes[0])
    lower = max(0, total_success - totals[1])
    upper = min(totals[0], total_success)
    return float(
        min(
            1.0,
            sum(
                probability(k)
                for k in range(lower, upper + 1)
                if probability(k) <= observed_probability + 1e-15
            ),
        )
    )


def decoding_summary(cases: list[dict], n_perm: int = 5000) -> dict:
    rng = np.random.default_rng(20260906)
    observed = {}
    for name, window in WINDOWS.items():
        correct = []
        per_case = []
        for case in cases:
            prediction = case_predictions(case, window)
            hits = prediction == case["labels"]
            correct.extend(hits.tolist())
            per_case.append(float(hits.mean()))
        count = int(np.sum(correct))
        total = len(correct)
        observed[name] = {
            "correct": count,
            "total": total,
            "accuracy": count / total,
            "wilson_95": wilson(count, total),
            "per_case_accuracy": per_case,
        }

    null = {name: np.empty(n_perm, dtype=np.float64) for name in WINDOWS}
    for rep in range(n_perm):
        for name, window in WINDOWS.items():
            hits = []
            for case in cases:
                shuffled = rng.permutation(case["labels"])
                prediction = case_predictions(case, window, shuffled)
                hits.extend((prediction == shuffled).tolist())
            null[name][rep] = np.mean(hits)
    for name in WINDOWS:
        value = observed[name]["accuracy"]
        observed[name]["permutation_p_above_chance"] = float(
            (1 + np.sum(null[name] >= value)) / (n_perm + 1)
        )
    return observed


def project_routes(case: dict) -> np.ndarray:
    xyz = (case["poses"][..., :3] - case["poses"][:, :1, :3]) * 100.0
    flat = xyz.reshape(-1, 3)
    _, _, vh = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    return (xyz - flat.mean(axis=0)) @ vh[:2].T


def route_panel(ax: plt.Axes, case: dict, title: str) -> None:
    routes = project_routes(case)
    labels = case["labels"]
    for mode, color in enumerate(COLORS):
        subset = routes[labels == mode]
        for route in subset:
            ax.plot(route[:, 0], route[:, 1], color=color, alpha=0.14, linewidth=0.8)
        mean = subset.mean(axis=0)
        ax.plot(mean[:, 0], mean[:, 1], color=color, linewidth=3.0, label=f"mode {chr(65 + mode)}")
        ax.scatter(mean[4, 0], mean[4, 1], color=color, s=32, marker="o", zorder=4)
        ax.scatter(mean[-1, 0], mean[-1, 1], color=color, s=45, marker="X", zorder=4)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("route PC1 (cm)")
    ax.set_ylabel("route PC2 (cm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.18)
    ax.legend(title="forced initial action", frameon=False, loc="lower right")
    inset = ax.inset_axes([0.67, 0.64, 0.31, 0.33])
    inset.imshow(case["rgb"])
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("0.25")


def main() -> None:
    cases, runs = load_cases()
    motion_cases = [case for case in cases if forced_pose_effect(case)[2]]
    decoding = decoding_summary(motion_cases)
    drawer = [case for case in motion_cases if case["task"] == "widowx_close_drawer"]
    basket = [case for case in motion_cases if "eggplant" in case["task"]]
    drawer_example = max(drawer, key=lambda row: row["confirmation"]["sep"])
    basket_example = max(basket, key=lambda row: row["confirmation"]["sep"])

    fig = plt.figure(figsize=(14.0, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 0.95))
    route_panel(fig.add_subplot(grid[0, 0]), drawer_example, "a  close drawer: strongest confirmed split")
    route_panel(fig.add_subplot(grid[0, 1]), basket_example, "b  basket: strongest confirmed split")

    ax = fig.add_subplot(grid[0, 2])
    names = list(WINDOWS)
    accuracy = [decoding[name]["accuracy"] for name in names]
    lower = [accuracy[i] - decoding[name]["wilson_95"][0] for i, name in enumerate(names)]
    upper = [decoding[name]["wilson_95"][1] - accuracy[i] for i, name in enumerate(names)]
    ax.bar(np.arange(len(names)), accuracy, color=["#58508D", "#A0A0A0", "#4C9F70"], width=0.68)
    ax.errorbar(np.arange(len(names)), accuracy, yerr=[lower, upper], fmt="none", color="black", capsize=4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="chance")
    ax.set_ylim(0.25, 1.02)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("mode decoding from end-effector pose")
    ax.set_title("c  mode-label decoding over time", loc="left", fontweight="bold")
    for i, name in enumerate(names):
        row = decoding[name]
        ax.text(i, accuracy[i] + 0.035, f"{row['correct']}/{row['total']}", ha="center", fontsize=9)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.18)

    ax = fig.add_subplot(grid[1, :2])
    ratios = []
    for case in motion_cases:
        gap, within, _ = spatial_gap_and_within(case)
        ratio = gap / np.maximum(within, 1e-6)
        ratio[0] = np.nan
        ratios.append(ratio)
        color = "#3B75AF" if case["task"] == "widowx_close_drawer" else "#E07A36"
        ax.plot(ratio, color=color, alpha=0.34, linewidth=1.3)
    ratios = np.stack(ratios)
    median_ratio = np.full(ratios.shape[1], np.nan)
    median_ratio[1:] = np.nanmedian(ratios[:, 1:], axis=0)
    ax.plot(median_ratio, color="black", linewidth=2.8, label="median across states")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2, label="between = within spread")
    ax.axvline(4, color="0.2", linestyle=":", linewidth=1.3, label="forced chunk ends")
    ax.set_xlabel("simulator step after same-state intervention")
    ax.set_ylabel("mode-centroid distance / within-mode RMS")
    ax.set_title("d  mode-conditioned spatial separation over time", loc="left", fontweight="bold")
    ax.set_ylim(0, min(6, np.nanpercentile(ratios, 98)))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ax = fig.add_subplot(grid[1, 2])
    for case in motion_cases:
        x = case["confirmation"]["delta_bic"]
        y = spatial_gap_and_within(case)[2]
        marker = "o" if case["task"] == "widowx_close_drawer" else "s"
        color = "#3B75AF" if marker == "o" else "#E07A36"
        ax.scatter(x, y, marker=marker, s=65, color=color, edgecolor="white", linewidth=0.8)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"same-state action evidence  $\Delta$BIC")
    ax.set_ylabel("late spatial between/within")
    ax.set_title("e  action evidence vs. late separation", loc="left", fontweight="bold")
    ax.grid(alpha=0.18)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color="#3B75AF", label="close drawer"),
        plt.Line2D([], [], marker="s", linestyle="none", color="#E07A36", label="eggplant to basket"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best")
    fig.savefig(HERE / "widowx_branching_audit.png", dpi=220)
    fig.savefig(HERE / "widowx_branching_audit.pdf")

    report = {
        "n_cases": len(cases),
        "n_branches": int(sum(len(case["labels"]) for case in cases)),
        "n_motion_cases": len(motion_cases),
        "n_motion_branches": int(sum(len(case["labels"]) for case in motion_cases)),
        "motion_case_rule": (
            f"forced-prefix mode-centroid gap >= {MIN_FORCED_POSITION_GAP_CM} cm "
            f"in position or >= {MIN_FORCED_ORIENTATION_GAP_DEG} deg in orientation"
        ),
        "normalization_note": (
            "late_position_between_within is recomputed from raw trajectories using "
            "Euclidean centroid distance / radial within-mode RMS; the run JSON used "
            "a per-coordinate RMS denominator"
        ),
        "source_runs": runs,
        "decoding": decoding,
        "cases": [
            {
                "uid": case["uid"],
                "task": case["task"],
                "group": case["group"],
                "delta_bic": case["confirmation"]["delta_bic"],
                "separation_sd": case["confirmation"]["sep"],
                "late_position_between_within": spatial_gap_and_within(case)[2],
                "time_aligned_curve_rms_cm": case["summary"]["time_aligned_curve_rms_cm"],
                "late_path_permutation_p": case["summary"]["late_path_permutation_p"],
                "success_count": case["summary"]["success_count"],
                "final_task_metric_delta": case["summary"]["final_task_metric_delta"],
                "final_task_metric_permutation_p": endpoint_permutation_p(
                    case["task_metric"][:, -1],
                    case["labels"],
                    20261000 + index,
                ),
                "success_fisher_p": fisher_exact_two_sided(
                    case["success"], case["labels"]
                ),
                "forced_position_gap_cm": forced_pose_effect(case)[0],
                "forced_orientation_gap_deg": forced_pose_effect(case)[1],
                "produces_measurable_pose_split": forced_pose_effect(case)[2],
            }
            for index, case in enumerate(cases)
        ],
    }
    (HERE / "widowx_branching_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
