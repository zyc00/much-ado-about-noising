#!/usr/bin/env python3
"""Summarize the exact-state WidowX continuous-arm mean intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


def metric(cases: list[dict], condition: str, key: str) -> np.ndarray:
    return np.asarray(
        [case["summary"][condition][key] for case in cases], dtype=np.float64
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def paired_summary(cases: list[dict], key: str) -> dict:
    mean = metric(cases, "mean_arm", key)
    medoid = metric(cases, "medoid_arm", key)
    try:
        test = wilcoxon(mean, medoid, alternative="less", method="auto")
        p = float(test.pvalue)
    except ValueError:
        p = 1.0
    return {
        "mean_arm": quantiles(mean),
        "medoid_arm": quantiles(medoid),
        "mean_lower_than_medoid": int(np.sum(mean < medoid)),
        "ties": int(np.sum(mean == medoid)),
        "n_states": len(cases),
        "paired_wilcoxon_mean_less_p": p,
    }


def make_figure(cases: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.15), constrained_layout=True)
    panels = (
        ("position_between_within", "position path"),
        ("orientation_between_within", "orientation path"),
    )
    colors = {
        "widowx_close_drawer": "#4c78a8",
        "widowx_put_eggplant_in_basket": "#f58518",
    }
    for axis, (key, title) in zip(axes, panels):
        mean = metric(cases, "mean_arm", key)
        medoid = metric(cases, "medoid_arm", key)
        for task in colors:
            selection = np.asarray([case["task"] == task for case in cases])
            label = "drawer" if "drawer" in task else "basket"
            axis.scatter(
                medoid[selection],
                mean[selection],
                s=24,
                alpha=0.78,
                color=colors[task],
                edgecolor="white",
                linewidth=0.35,
                label=label,
            )
        limit = max(1.5, float(np.quantile(np.r_[mean, medoid], 0.98)) * 1.08)
        axis.plot([0, limit], [0, limit], color="0.35", linewidth=0.9, linestyle="--")
        axis.axhline(1.0, color="0.75", linewidth=0.8, linestyle=":")
        axis.axvline(1.0, color="0.75", linewidth=0.8, linestyle=":")
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("sample medoid / within-sample RMS")
        axis.set_ylabel("conditional mean / within-sample RMS")
        axis.set_title(title)
        axis.grid(color="0.92", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Exact-state executability of the continuous arm mean", fontsize=11)
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "widowx_arm_mean_summary.json",
    )
    args = parser.parse_args()
    with args.input.open() as handle:
        raw = json.load(handle)
    cases = raw["cases"]
    result = {
        "protocol": {
            "checkpoint": raw["checkpoint"],
            "n_states": len(cases),
            "tasks": raw["tasks"],
            "k_mean": raw["k_mean"],
            "k_reference": raw["k_reference"],
            "execution_horizon": raw["execution_horizon"],
            "gripper": raw["gripper_protocol"],
            "snapshot_replay_max_abs_pose_error": raw[
                "snapshot_replay_max_abs_pose_error"
            ],
        },
        "all_states": {
            "position_between_within": paired_summary(
                cases, "position_between_within"
            ),
            "orientation_between_within": paired_summary(
                cases, "orientation_between_within"
            ),
            "mean_position_inside_reference_rms": int(
                np.sum(metric(cases, "mean_arm", "position_between_within") < 1.0)
            ),
            "mean_orientation_inside_reference_rms": int(
                np.sum(metric(cases, "mean_arm", "orientation_between_within") < 1.0)
            ),
            "mean_outside_95pct_reference_paths": int(
                np.sum(
                    (metric(cases, "mean_arm", "position_percentile") >= 0.95)
                    | (metric(cases, "mean_arm", "orientation_percentile") >= 0.95)
                )
            ),
        },
        "by_task": {},
    }
    for task in raw["tasks"]:
        selected = [case for case in cases if case["task"] == task]
        result["by_task"][task] = {
            "n_states": len(selected),
            "position_between_within": paired_summary(
                selected, "position_between_within"
            ),
            "orientation_between_within": paired_summary(
                selected, "orientation_between_within"
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(result, handle, indent=2)
    make_figure(cases, args.out.with_name("widowx_arm_mean.png"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
