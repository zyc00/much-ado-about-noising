#!/usr/bin/env python3
"""Visual audit of candidate same-state modes in the WidowX Flow probe."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from scripts.probe_widowx_multimodality import GROUP_COLUMNS, fit_gmm2, group_view


ROOT = Path(__file__).resolve().parent
ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
# Checkpoint q01/q99 bounds used by its min-max processor. The first three
# channels are converted to cm and rotations to degrees for the raw plots.
ACTION_LOW = np.array((-0.02875255, -0.04170214, -0.02609672, -0.08052875, -0.09249907, -0.20738555, 0.0))
ACTION_HIGH = np.array((0.02830666, 0.04089853, 0.04018052, 0.08173403, 0.07760761, 0.20384654, 1.0))
ACTION_LABELS = (r"$\Delta x$ (cm)", r"$\Delta y$ (cm)", r"$\Delta z$ (cm)", r"$\Delta$roll (deg)", r"$\Delta$pitch (deg)", r"$\Delta$yaw (deg)", "gripper command")
COLORS = ("#2878B5", "#D95319")
EXTREME_CASES = (
    ("close_drawer:11:2", "rotation"),
    ("close_drawer:267:1", "rotation"),
    ("close_drawer:385:0", "arm"),
    ("close_drawer:440:4", "rotation"),
    ("eggplant_basket:73:3", "arm"),
    ("eggplant_basket:148:1", "gripper"),
)
RANDOM_CASES = (
    ("close_drawer:126:2", "arm"),
    ("close_drawer:385:0", "arm"),
    ("close_drawer:458:2", "arm"),
    ("close_drawer:570:3", "arm"),
    ("eggplant_basket:53:3", "arm"),
    ("eggplant_basket:171:2", "arm"),
)
MAIN_CASES = (
    ("close_drawer:126:2", "arm"),
    ("eggplant_basket:53:3", "arm"),
    ("eggplant_basket:53:0", "gripper"),
)


def projection(discovery: np.ndarray, confirmation: np.ndarray, group: str):
    gd = group_view(discovery, GROUP_COLUMNS[group])
    gc = group_view(confirmation, GROUP_COLUMNS[group])
    center = gd.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(gd - center, full_matrices=False)
    if not len(singular_values) or singular_values[0] < 1e-12:
        raise ValueError("Degenerate discovery samples")
    axis = vh[0]
    z = (gc - center) @ axis
    fit = fit_gmm2(z)
    return z, fit


def density_geometry(fit: dict, z: np.ndarray):
    weights, means, variances = fit["weights"], fit["means"], fit["variances"]
    pad = max(float(z.std()), 1e-3)
    grid = np.linspace(float(z.min() - pad), float(z.max() + pad), 5000)
    components = np.stack(
        [
            weight
            / np.sqrt(2 * np.pi * variance)
            * np.exp(-0.5 * (grid - mean) ** 2 / variance)
            for weight, mean, variance in zip(weights, means, variances)
        ]
    )
    density = components.sum(axis=0)
    peaks, _ = find_peaks(density, prominence=density.max() * 1e-5)
    valley_depth = 0.0
    if len(peaks) >= 2:
        chosen = sorted(sorted(peaks, key=lambda i: density[i], reverse=True)[:2])
        valley = density[chosen[0] : chosen[1] + 1].min()
        valley_depth = 1.0 - valley / min(density[chosen[0]], density[chosen[1]])
    return grid, components, density, len(peaks), float(valley_depth)


def within_rms(chunks: np.ndarray, labels: np.ndarray, dims: tuple[int, ...]) -> float:
    residuals = []
    for label in (0, 1):
        group = chunks[labels == label][:, :, dims]
        residuals.append(group - group.mean(axis=0, keepdims=True))
    return float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))


def main() -> None:
    stem = os.environ.get("AUDIT_STEM", "rollout_v1")
    variant = os.environ.get("AUDIT_VARIANT", "all")
    if stem == "rollout_random_v1" and variant == "main":
        cases = MAIN_CASES
        output_stem = "widowx_main_modes"
    elif stem == "rollout_random_v1":
        cases = RANDOM_CASES
        output_stem = "widowx_random_arm_audit"
    else:
        cases = EXTREME_CASES
        output_stem = "widowx_rollout_audit"
    summary = json.loads((ROOT / f"{stem}.json").read_text())
    raw = np.load(ROOT / f"{stem}.npz")
    lookup = {row["uid"]: (i, row) for i, row in enumerate(summary["deep"])}

    fig, axes = plt.subplots(
        len(cases),
        3,
        figsize=(10.8, 2.15 * len(cases)),
        gridspec_kw={"width_ratios": (1.0, 1.45, 1.65)},
        constrained_layout=True,
    )
    audit = []
    for row_no, (uid, group) in enumerate(cases):
        i, result = lookup[uid]
        chunks = raw["confirmation"][i].astype(np.float64)
        z, fit = projection(raw["discovery"][i], chunks, group)
        labels = fit["labels"]
        grid, components, density, n_modes, valley_depth = density_geometry(fit, z)

        # Image of the exact rollout observation supplied to the policy.
        image_ax = axes[row_no, 0]
        image_ax.imshow(raw["rgb"][i])
        image_ax.axis("off")
        task = "close drawer" if result["source_group"] == "close_drawer" else "eggplant to basket"
        image_ax.set_title(f"{task}\n{uid.split(':', 1)[1]}", fontsize=9)

        # Independent confirmation samples projected on discovery PC1.
        density_ax = axes[row_no, 1]
        for label, color in enumerate(COLORS):
            density_ax.hist(
                z[labels == label], bins=12, density=True, alpha=0.22,
                color=color, edgecolor="none"
            )
            density_ax.plot(grid, components[label], color=color, lw=1.5)
            density_ax.scatter(
                z[labels == label], np.full((labels == label).sum(), -0.025 * density.max()),
                marker="|", s=30, color=color, clip_on=False
            )
        density_ax.plot(grid, density, color="black", lw=1.2)
        density_ax.set_yticks([])
        density_ax.set_xlabel(f"{group} discovery-PC1")
        density_ax.set_title(
            f"$\\Delta$BIC={fit['delta_bic']:.1f}, sep={fit['sep']:.1f}$\\sigma$\n"
            f"valley depth={valley_depth:.2f}",
            fontsize=9,
        )

        # Plot the action coordinate carrying most between-mode energy.
        means = np.stack([chunks[labels == label].mean(axis=0) for label in (0, 1)])
        delta = means[1] - means[0]
        allowed_dims = GROUP_COLUMNS[group]
        dominant_dim = max(allowed_dims, key=lambda dim: float(np.sum(delta[:, dim] ** 2)))
        trajectory_ax = axes[row_no, 2]
        steps = np.arange(chunks.shape[1])
        for label, color in enumerate(COLORS):
            values = chunks[labels == label, :, dominant_dim]
            values = (values + 1.0) * 0.5 * (
                ACTION_HIGH[dominant_dim] - ACTION_LOW[dominant_dim]
            ) + ACTION_LOW[dominant_dim]
            if dominant_dim < 3:
                values = 100.0 * values
            elif dominant_dim < 6:
                values = np.rad2deg(values)
            trajectory_ax.plot(steps, values.T, color=color, alpha=0.13, lw=0.65)
            trajectory_ax.plot(steps, values.mean(axis=0), color=color, lw=2.2)
        trajectory_ax.set_xlabel("action-chunk step")
        trajectory_ax.set_ylabel(ACTION_LABELS[dominant_dim])
        trajectory_ax.set_xticks(steps)
        trajectory_ax.spines[["top", "right"]].set_visible(False)
        between_rms = float(np.sqrt(np.mean(delta[:, allowed_dims] ** 2)))
        residual_rms = within_rms(chunks, labels, allowed_dims)
        trajectory_ax.set_title(
            f"between/within RMS={between_rms / max(residual_rms, 1e-12):.2f}",
            fontsize=9,
        )

        test = result["tests"][group]
        audit.append(
            {
                "uid": uid,
                "group": group,
                "delta_bic": fit["delta_bic"],
                "separation_sd": fit["sep"],
                "minimum_weight": fit["min_weight"],
                "fitted_density_modes": n_modes,
                "valley_depth": valley_depth,
                "energy_translation": test["energy_translation"],
                "energy_rotation": test["energy_rotation"],
                "energy_gripper": test["energy_gripper"],
                "dominant_coordinate": ACTION_NAMES[dominant_dim],
                "between_within_rms_ratio": between_rms / max(residual_rms, 1e-12),
            }
        )

    axes[0, 0].text(-0.08, 1.28, "observation", transform=axes[0, 0].transAxes, weight="bold")
    axes[0, 1].text(-0.08, 1.28, "independent mode test", transform=axes[0, 1].transAxes, weight="bold")
    axes[0, 2].text(-0.08, 1.28, "raw sampled action chunks", transform=axes[0, 2].transAxes, weight="bold")
    fig.savefig(ROOT / f"{output_stem}.png", dpi=220)
    fig.savefig(ROOT / f"{output_stem}.pdf")
    (ROOT / f"{output_stem}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
