#!/usr/bin/env python3
"""Deprecated specialist diagnostic, not the general-MSE motivation figure.

Panels a, b, and d use episode-held-out BridgeData V2 examples evaluated by
three equally trained MSE specialists (small/mid/large action-magnitude
quintiles). Panel c uses actual SIMPLER rollout states from the released Flow
checkpoint. All panels exclude the gripper and use only the six continuous
end-effector channels.
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
from scipy.stats import rankdata, spearmanr


BLUE = "#2C7BB6"
GRAY = "#7A7A7A"
ORANGE = "#D55E00"
GROUP_COLORS = {0: "#2C7BB6", 2: "#8C8C8C", 4: "#D55E00"}
GROUP_LABELS = {0: "Q1: small", 2: "Q3: medium", 4: "Q5: large"}
TASK_COLORS = {0: "#2C7BB6", 1: "#009E73"}
SEED = 20260908


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.4,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_specialists(results: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        "group": [],
        "action_scale": [],
        "residual_scale": [],
        "residual": [],
        "episode": [],
    }
    for group in (0, 2, 4):
        path = results / f"widowx_q{group}.npz"
        with np.load(path, allow_pickle=False) as z:
            residual = z["residual"].astype(np.float64)
            rows["group"].append(np.full(len(residual), group, dtype=int))
            rows["action_scale"].append(z["action_magnitude"].astype(np.float64))
            rows["residual_scale"].append(
                np.sqrt(np.mean(residual**2, axis=(1, 2)))
            )
            rows["residual"].append(residual)
            rows["episode"].append(z["episode"].astype(int))
    return {key: np.concatenate(value) for key, value in rows.items()}


def task_relative(values: np.ndarray, task: np.ndarray) -> np.ndarray:
    relative = np.empty_like(values, dtype=float)
    for task_id in np.unique(task):
        keep = task == task_id
        relative[keep] = values[keep] / np.exp(np.mean(np.log(values[keep])))
    return relative


def clustered_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    cluster: np.ndarray,
    draws: int = 5000,
) -> tuple[float, list[float]]:
    observed = float(spearmanr(x, y).statistic)
    unique = np.unique(cluster)
    groups = [np.flatnonzero(cluster == value) for value in unique]
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        bootstrap[draw] = spearmanr(x[indices], y[indices]).statistic
    return observed, np.quantile(bootstrap, [0.025, 0.975]).tolist()


def within_task_spearman(x: np.ndarray, y: np.ndarray, task: np.ndarray) -> float:
    rx = np.empty(len(x), dtype=float)
    ry = np.empty(len(y), dtype=float)
    for task_id in np.unique(task):
        keep = task == task_id
        rx[keep] = rankdata(x[keep])
        ry[keep] = rankdata(y[keep])
    return float(np.corrcoef(rx, ry)[0, 1])


def flow_cluster_ci(
    x: np.ndarray,
    y: np.ndarray,
    task: np.ndarray,
    cluster: np.ndarray,
    draws: int = 5000,
) -> tuple[float, list[float]]:
    observed = within_task_spearman(x, y, task)
    unique = np.unique(cluster)
    groups = [np.flatnonzero(cluster == value) for value in unique]
    rng = np.random.default_rng(SEED + 1)
    bootstrap = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        bootstrap[draw] = within_task_spearman(x[indices], y[indices], task[indices])
    return observed, np.quantile(bootstrap, [0.025, 0.975]).tolist()


def quantile_summary(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    bins: int = 7,
    draws: int = 3000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    group = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)

    def summarize(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([np.median(x[indices][group[indices] == b]) for b in range(bins)]),
            np.asarray(
                [np.sqrt(np.mean(y[indices][group[indices] == b] ** 2)) for b in range(bins)]
            ),
        )

    base = np.arange(len(x))
    centers, curve = summarize(base)
    unique = np.unique(clusters)
    by_cluster = [np.flatnonzero(clusters == value) for value in unique]
    rng = np.random.default_rng(SEED + 2)
    boot = np.empty((draws, bins), dtype=float)
    for draw in range(draws):
        indices = np.concatenate(
            [by_cluster[i] for i in rng.integers(len(by_cluster), size=len(by_cluster))]
        )
        _, boot[draw] = summarize(indices)
    return centers, curve, np.quantile(boot, [0.025, 0.975], axis=0)


def likelihood_audit(data: dict[str, np.ndarray]) -> dict:
    """Cross-fit one shared variance versus one variance per scale regime."""
    energy = data["residual_scale"] ** 2
    group = data["group"]
    episode = data["episode"]
    episodes = np.unique(episode)
    rng = np.random.default_rng(SEED + 3)
    rng.shuffle(episodes)
    split = np.isin(episode, episodes[::2]).astype(int)
    fixed_ll = np.full(len(energy), np.nan)
    hetero_ll = np.full(len(energy), np.nan)
    for fold in (0, 1):
        train = split == fold
        test = ~train
        fixed_variance = np.mean(energy[train])
        group_variance = {
            value: np.mean(energy[train & (group == value)]) for value in np.unique(group)
        }
        hetero_variance = np.asarray([group_variance[value] for value in group[test]])
        fixed_ll[test] = -0.5 * (
            np.log(2 * np.pi * fixed_variance) + energy[test] / fixed_variance
        )
        hetero_ll[test] = -0.5 * (
            np.log(2 * np.pi * hetero_variance) + energy[test] / hetero_variance
        )

    delta = hetero_ll - fixed_ll
    labels = [0, 2, 4, -1]
    names = ["Q1", "Q3", "Q5", "Overall"]
    result = []
    for row, name in zip(labels, names):
        keep = np.ones(len(group), dtype=bool) if row == -1 else group == row
        episode_values = []
        for episode_id in np.unique(episode[keep]):
            selected = keep & (episode == episode_id)
            episode_values.append(float(np.mean(delta[selected])))
        episode_values = np.asarray(episode_values)
        rng = np.random.default_rng(SEED + 10 + row)
        indices = rng.integers(
            len(episode_values), size=(30000, len(episode_values))
        )
        boot = episode_values[indices].mean(axis=1)
        result.append(
            {
                "name": name,
                "mean": float(episode_values.mean()),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "episodes": int(len(episode_values)),
            }
        )
    return {
        "fixed_mean_ll": float(fixed_ll.mean()),
        "hetero_mean_ll": float(hetero_ll.mean()),
        "rows": result,
        "split": "two-fold, disjoint episodes",
        "uniform_model": "one variance shared across Q1/Q3/Q5",
        "heterogeneous_model": "one variance per action-scale regime",
    }


def draw(
    specialists: dict[str, np.ndarray],
    flow: dict[str, np.ndarray],
    likelihood: dict,
    output: Path,
) -> dict:
    configure()
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.15, 2.28),
        gridspec_kw={"width_ratios": [1.06, 1.05, 1.06, 0.93]},
    )
    axa, axb, axc, axd = axes

    # (a) Held-out action scale versus residual scale.
    for group in (0, 2, 4):
        keep = specialists["group"] == group
        axa.scatter(
            specialists["action_scale"][keep],
            specialists["residual_scale"][keep],
            s=5.0,
            color=GROUP_COLORS[group],
            alpha=0.13,
            linewidths=0,
            rasterized=True,
            label=GROUP_LABELS[group],
        )
        axa.scatter(
            [np.median(specialists["action_scale"][keep])],
            [np.sqrt(np.mean(specialists["residual_scale"][keep] ** 2))],
            s=27,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    rho_a, ci_a = clustered_spearman_ci(
        specialists["action_scale"],
        specialists["residual_scale"],
        specialists["episode"],
    )
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(0.085, 0.72)
    axa.set_ylim(0.035, 0.82)
    axa.set_xticks([0.1, 0.2, 0.4, 0.7], ["0.1", "0.2", "0.4", "0.7"])
    axa.set_yticks([0.05, 0.1, 0.2, 0.4, 0.8], ["0.05", "0.1", "0.2", "0.4", "0.8"])
    axa.xaxis.set_minor_locator(NullLocator())
    axa.yaxis.set_minor_locator(NullLocator())
    axa.set_xlabel(r"Demonstration-action RMS $\Vert a\Vert_{\rm RMS}$")
    axa.set_ylabel("Held-out residual RMS")
    axa.set_title("a  Residual scale follows label scale", loc="left", fontweight="bold")
    axa.text(
        0.04,
        0.96,
        rf"$\rho={rho_a:.2f}$; 95\% CI $[{ci_a[0]:.2f},{ci_a[1]:.2f}]$",
        transform=axa.transAxes,
        ha="left",
        va="top",
        color="#9B2F22",
        fontweight="bold",
    )
    axa.legend(frameon=False, loc="lower right", labelspacing=0.18, handletextpad=0.25)

    # (b) Direct visualization of the three fitted Gaussian radii.
    centered = {}
    sigma = {}
    for group in (0, 2, 4):
        keep = specialists["group"] == group
        residual = specialists["residual"][keep]
        residual = residual - residual.mean(axis=0, keepdims=True)
        centered[group] = residual.reshape(-1)
        sigma[group] = float(np.sqrt(np.mean(centered[group] ** 2)))
    extent = max(np.quantile(np.abs(centered[group]), 0.995) for group in (0, 2, 4))
    grid = np.linspace(-extent, extent, 700)
    for group in (4, 2, 0):
        density = np.exp(-0.5 * (grid / sigma[group]) ** 2) / (
            np.sqrt(2 * np.pi) * sigma[group]
        )
        axb.fill_between(grid, density, color=GROUP_COLORS[group], alpha=0.10, linewidth=0)
        axb.plot(
            grid,
            density,
            color=GROUP_COLORS[group],
            lw=1.8,
            label=rf"{GROUP_LABELS[group].split(':')[0]}: $\sigma={sigma[group]:.2f}$",
        )
    axb.axvline(0, color="0.55", lw=0.65, ls=(0, (3, 2)))
    axb.set_xlim(-extent, extent)
    axb.set_ylim(bottom=0)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.set_xlabel("Centered continuous residual")
    axb.set_ylabel("Density")
    axb.set_title("b  Different regimes, different radii", loc="left", fontweight="bold")
    axb.text(
        0.97,
        0.96,
        rf"Q5 / Q1 $={sigma[4] / sigma[0]:.2f}\times$",
        transform=axb.transAxes,
        ha="right",
        va="top",
        color="#9B2F22",
        fontweight="bold",
    )
    axb.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.82), labelspacing=0.22)

    # (c) Flow sampling spread versus executed continuous-action scale.
    uid = flow["uid"].astype(str)
    env = np.asarray([int(value.rsplit(":", 1)[1]) for value in uid])
    cluster = flow["task"].astype(int) * 10 + env
    flow_x = task_relative(flow["label_scale"], flow["task"])
    flow_y = task_relative(flow["flow_spread"], flow["task"])
    rho_c, ci_c = flow_cluster_ci(flow_x, flow_y, flow["task"], cluster)
    centers, curve, curve_ci = quantile_summary(flow_x, flow_y, cluster)
    for task_id in (0, 1):
        keep = flow["task"] == task_id
        axc.scatter(
            flow_x[keep],
            flow_y[keep],
            s=5.2,
            color=TASK_COLORS[task_id],
            alpha=0.12,
            linewidths=0,
            rasterized=True,
            label=("Close drawer" if task_id == 0 else "Eggplant to basket"),
        )
    axc.fill_between(centers, curve_ci[0], curve_ci[1], color=ORANGE, alpha=0.13, linewidth=0)
    axc.plot(centers, curve, "o-", color=ORANGE, lw=1.7, ms=3.0, mec="white", mew=0.45)
    axc.axhline(1, color="0.55", lw=0.65, ls=(0, (3, 2)))
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlim(0.32, 2.05)
    axc.set_ylim(0.32, 2.65)
    axc.set_xticks([0.4, 0.7, 1, 1.5, 2], ["0.4", "0.7", "1", "1.5", "2"])
    axc.set_yticks([0.4, 0.7, 1, 1.5, 2.5], ["0.4", "0.7", "1", "1.5", "2.5"])
    axc.xaxis.set_minor_locator(NullLocator())
    axc.yaxis.set_minor_locator(NullLocator())
    axc.set_xlabel("Task-normalized executed-action RMS")
    axc.set_ylabel("Task-normalized Flow spread")
    axc.set_title("c  Flow spread follows motion scale", loc="left", fontweight="bold")
    axc.text(
        0.04,
        0.96,
        rf"$\rho={rho_c:.2f}$; 95\% CI $[{ci_c[0]:.2f},{ci_c[1]:.2f}]$",
        transform=axc.transAxes,
        ha="left",
        va="top",
        color=ORANGE,
        fontweight="bold",
    )
    axc.legend(
        frameon=False,
        loc="lower right",
        labelspacing=0.12,
        handletextpad=0.15,
        markerscale=1.5,
        borderaxespad=0.15,
    )

    # (d) Minimal forest plot of held-out log-likelihood gain.
    names = [row["name"] for row in likelihood["rows"]]
    ypos = np.arange(len(names))[::-1]
    for y, row in zip(ypos, likelihood["rows"]):
        mean = row["mean"]
        lo, hi = row["ci95"]
        color = ORANGE if row["name"] == "Overall" else GROUP_COLORS[{"Q1": 0, "Q3": 2, "Q5": 4}[row["name"]]]
        marker = "D" if row["name"] == "Overall" else "o"
        axd.plot([lo, hi], [y, y], color=color, lw=2.1, solid_capstyle="round")
        axd.plot(mean, y, marker=marker, color=color, ms=4.4, mec="white", mew=0.55)
        label = (
            rf"+{mean:.3f}  [{lo:.3f}, {hi:.3f}]"
            if row["name"] == "Overall"
            else f"+{mean:.3f}"
        )
        axd.text(
            hi + 0.009,
            y,
            label,
            va="center",
            color=color,
            fontsize=6.3,
            fontweight=("bold" if row["name"] == "Overall" else "normal"),
        )
    axd.axvline(0, color="0.45", lw=0.8, ls=(0, (3, 2)))
    axd.set_xlim(-0.025, 0.315)
    axd.set_ylim(-0.65, len(names) - 0.35)
    axd.set_yticks(ypos, names)
    axd.tick_params(axis="y", length=0, pad=2)
    axd.set_xticks([0, 0.1, 0.2], ["0", "+0.1", "+0.2"])
    axd.set_xlabel(r"$\Delta$ held-out log likelihood" + "\n" + "(hetero. $-$ uniform; nat/dim)")
    axd.set_title("d  Heterogeneous Gaussian fits better", loc="left", fontweight="bold", fontsize=7.7)

    for ax in axes:
        ax.grid(axis="y", color="0.92", linewidth=0.55, zorder=0)
        ax.tick_params(length=2.4, pad=2)
    fig.text(
        0.5,
        0.995,
        "GR00T N1.7 / WidowX  ·  continuous 6-DoF actions  ·  gripper excluded",
        ha="center",
        va="top",
        fontsize=7.6,
        color="0.32",
    )
    fig.subplots_adjust(left=0.064, right=0.995, bottom=0.255, top=0.82, wspace=0.42)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=340)
    plt.close(fig)

    return {
        "mse_specialists": {
            "heldout_states": int(len(specialists["group"])),
            "unique_episodes": int(len(np.unique(specialists["episode"]))),
            "groups": ["Q1", "Q3", "Q5"],
            "residual_vs_label_spearman": rho_a,
            "episode_bootstrap_ci95": ci_a,
            "fitted_sigma": {f"Q{group + 1}": sigma[group] for group in (0, 2, 4)},
            "q5_q1_sigma": sigma[4] / sigma[0],
        },
        "flow_rollouts": {
            "states": int(len(flow_x)),
            "rollout_streams": int(len(np.unique(cluster))),
            "samples_per_state": 8,
            "within_task_spearman": rho_c,
            "stream_bootstrap_ci95": ci_c,
        },
        "gaussian_goodness_of_fit": likelihood,
        "channels": "six continuous end-effector channels; gripper excluded throughout",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("analysis/paper/action_magnitude_specialists/results"),
    )
    parser.add_argument(
        "--flow",
        type=Path,
        default=Path("analysis/paper/action_magnitude_specialists/widowx_flow_scale.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/paper/action_magnitude_specialists/fig_widowx_specialist_legacy"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    specialists = load_specialists(args.results)
    with np.load(args.flow, allow_pickle=False) as z:
        flow = {key: z[key] for key in z.files}
    likelihood = likelihood_audit(specialists)
    summary = draw(specialists, flow, likelihood, args.output)
    args.output.with_name(args.output.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
