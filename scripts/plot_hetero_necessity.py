#!/usr/bin/env python3
"""Build the ICLR motivation figure for input-dependent residual scale.

Panels (a-c) are model-free: action chunks are compared with the mean chunk of
nearest states from other episodes of the same task.  Panel (d) is the causal
control: the Student-t tail is held fixed while its scale is either global or
input-dependent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/paper/hetero_necessity"
RAW = OUT / "raw"
CROSSFIT = OUT / "crossfit"
COLORS = {
    "RoboCasa-GR1": "#4C78A8",
    "LIBERO": "#009E73",
    "BridgeData V2": "#E69F00",
    "Tool-Hang": "#CC79A7",
    "Transport-MH": "#7B61A8",
}
SOURCES = {
    "RoboCasa-GR1": "gr1.npz",
    "LIBERO": "pi05-libero.npz",
    "BridgeData V2": "bridge.npz",
    "Tool-Hang": "toolhang-ph.npz",
    "Transport-MH": "transport-mh.npz",
}


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def centered_logs(z: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove task-wide scale before measuring within-task persistence."""
    eps = 1e-12
    x = np.log(z["first_energy"].astype(float) + eps)
    y = np.log(z["second_energy"].astype(float) + eps)
    full = np.log(z["full_energy"].astype(float) + eps)
    task = z["task"]
    for task_id in np.unique(task):
        keep = task == task_id
        x[keep] -= np.median(x[keep])
        y[keep] -= np.median(y[keep])
        full[keep] -= np.median(full[keep])
    return x, y, full


def reliability(x: np.ndarray, y: np.ndarray) -> float:
    r = float(np.corrcoef(x, y)[0, 1])
    return 2 * r / (1 + r)


def quintile_ratio(z: dict) -> float:
    """Q5/Q1 held-out-half RMS; groups use only first-half residual scale."""
    task = z["task"]
    ratios = []
    for task_id in np.unique(task):
        keep = task == task_id
        if keep.sum() < 50:
            continue
        first = z["first_energy"][keep]
        second = z["second_energy"][keep]
        group = np.digitize(first, np.quantile(first, [.2, .4, .6, .8]))
        lo = np.sqrt(np.mean(second[group == 0]))
        hi = np.sqrt(np.mean(second[group == 4]))
        ratios.append(float(hi / lo))
    return float(np.median(ratios))


def episode_bootstrap(z: dict, seed: int = 20260907, draws: int = 500) -> tuple[float, list[float]]:
    """Cluster bootstrap episodes within each task for split-half reliability."""
    rng = np.random.default_rng(seed)
    task = z["task"]
    episode = z["episode"]
    x0 = np.log(z["first_energy"].astype(float) + 1e-12)
    y0 = np.log(z["second_energy"].astype(float) + 1e-12)
    task_groups = []
    for task_id in np.unique(task):
        task_idx = np.flatnonzero(task == task_id)
        episodes = np.unique(episode[task_idx])
        by_episode = [task_idx[episode[task_idx] == ep] for ep in episodes]
        if by_episode:
            task_groups.append(by_episode)

    def one(indices_by_task: list[np.ndarray]) -> float:
        xs, ys = [], []
        for idx in indices_by_task:
            x = x0[idx].copy()
            y = y0[idx].copy()
            x -= np.median(x)
            y -= np.median(y)
            xs.append(x)
            ys.append(y)
        return reliability(np.concatenate(xs), np.concatenate(ys))

    observed = one([np.concatenate(group) for group in task_groups])
    boot = []
    for _ in range(draws):
        sampled = []
        for group in task_groups:
            picks = rng.integers(0, len(group), size=len(group))
            sampled.append(np.concatenate([group[pick] for pick in picks]))
        boot.append(one(sampled))
    return observed, np.quantile(boot, [.025, .975]).tolist()


def summarize() -> tuple[dict, dict]:
    data = {name: load(RAW / filename) for name, filename in SOURCES.items()}
    summary = {}
    for i, (name, z) in enumerate(data.items()):
        x, y, full = centered_logs(z)
        rel, ci = episode_bootstrap(z, seed=20260907 + i)
        metadata = json.loads(str(z["metadata"]))
        summary[name] = {
            "source": str((RAW / SOURCES[name]).resolve()),
            "sha256": hashlib.sha256((RAW / SOURCES[name]).read_bytes()).hexdigest(),
            "chunks": int(len(x)),
            "tasks": int(len(np.unique(z["task"]))),
            "episodes": int(len(np.unique(z["episode"]))),
            "continuous_dimensions": int(metadata["continuous_dimensions"]),
            "q5_q1_heldout_half_rms": quintile_ratio(z),
            "split_half_reliability": rel,
            "split_half_reliability_ci95_episode_bootstrap": ci,
            "within_task_sd_log_residual_energy": float(np.std(full)),
        }
    return data, summary


def crossfit_curves() -> dict:
    """Per-task curves; query episodes never contribute to their predictors."""
    result = {}
    for name, filename in SOURCES.items():
        z = load(CROSSFIT / filename)
        pred = z["predicted_energy"].astype(float)
        observed = z["observed_energy"].astype(float)
        task = z["task"]
        curves, ratios, correlations = [], [], []
        for task_id in np.unique(task):
            keep = task == task_id
            if keep.sum() < 50:
                continue
            p, o = pred[keep], observed[keep]
            group = np.digitize(p, np.quantile(p, [.2, .4, .6, .8]))
            rms = np.array([np.sqrt(np.mean(o[group == q])) for q in range(5)])
            curves.append(rms / np.sqrt(np.mean(o)))
            ratios.append(float(rms[-1] / rms[0]))
            correlations.append(float(np.corrcoef(np.log(p + 1e-12), np.log(o + 1e-12))[0, 1]))
        metadata = json.loads(str(z["metadata"]))
        result[name] = {
            "curve": np.median(np.asarray(curves), axis=0),
            "q5_q1": float(np.median(ratios)),
            "log_energy_pearson": float(np.median(correlations)),
            "tasks_summarized": len(curves),
            "queries": int(len(pred)),
            "metadata": metadata,
            "source": str((CROSSFIT / filename).resolve()),
            "sha256": hashlib.sha256((CROSSFIT / filename).read_bytes()).hexdigest(),
        }
    return result


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw(data: dict, summary: dict, crossfit: dict) -> None:
    style()
    fig = plt.figure(figsize=(7.1, 2.15))
    ax_a = fig.add_axes([0.055, 0.23, 0.225, 0.61])
    ax_b = fig.add_axes([0.385, 0.23, 0.135, 0.61])
    ax_c = fig.add_axes([0.575, 0.23, 0.14, 0.61])
    ax_d = fig.add_axes([0.775, 0.23, 0.215, 0.61])

    # (a) Cross-half density: group on the first four steps and display only
    # residual coordinates from the held-out last four steps.
    small = load(RAW / "toolhang-residuals-5k.npz")
    first = small["first_energy"].astype(float)
    residual = small["residual"].astype(float)
    group = np.digitize(first, np.quantile(first, [.2, .4, .6, .8]))
    palette = {0: "#0072B2", 2: "#7F7F7F", 4: "#D55E00"}
    labels = {0: "Q1: low", 2: "Q3: mid", 4: "Q5: high"}
    held = residual[:, residual.shape[1] // 2 :].reshape(len(residual), -1)
    pooled = np.concatenate([held[group == q].reshape(-1) for q in (0, 2, 4)])
    limit = float(np.quantile(np.abs(pooled), .985))
    grid = np.linspace(-limit, limit, 400)
    radii = {}
    for q in (0, 2, 4):
        values = held[group == q].reshape(-1)
        values = values - np.median(values)
        kde = gaussian_kde(values)
        density = kde(grid)
        radii[q] = float(np.sqrt(np.mean(values**2)))
        ax_a.fill_between(grid, density, color=palette[q], alpha=.12, lw=0)
        ax_a.plot(grid, density, color=palette[q], lw=1.5, label=labels[q])
    ax_a.set_xlim(-limit, limit)
    ax_a.set_ylim(bottom=0)
    ax_a.set_yticks([])
    ax_a.spines["left"].set_visible(False)
    ax_a.set_xlabel("Held-out residual")
    ax_a.set_ylabel("Density", labelpad=3)
    ax_a.set_title("a  Held-out steps preserve radius", loc="left", fontweight="bold",
                   fontsize=8.4, pad=6)
    ax_a.text(
        .98,
        .96,
        rf"Q5 / Q1 = {radii[4] / radii[0]:.1f}$\times$",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#9E341F",
    )
    ax_a.text(.02, .96, "group: first 4 steps\nplot: last 4 steps", transform=ax_a.transAxes,
              va="top", color="0.35", fontsize=6.8)
    ax_a.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.02, .78),
                handlelength=1.6, labelspacing=.25)

    names = list(SOURCES)
    short_names = ["GR1", "LIBERO", "Bridge", "Tool-Hang", "Transport-MH"]
    y = np.arange(len(names))[::-1]

    # (b) Held-out scale separation across datasets.
    ratios = [summary[name]["q5_q1_heldout_half_rms"] for name in names]
    for yi, name, value in zip(y, names, ratios):
        ax_b.plot([1, value], [yi, yi], color=COLORS[name], lw=2.4, solid_capstyle="round")
        ax_b.plot(value, yi, "o", color=COLORS[name], ms=5, mec="white", mew=.7)
        ax_b.text(value + .08, yi, f"{value:.1f}$\times$", va="center", fontsize=7,
                  color=COLORS[name])
    ax_b.axvline(1, color="0.55", lw=.8, ls=(0, (3, 2)))
    ax_b.set_xlim(.9, 5.8)
    ax_b.set_xticks([1, 3, 5])
    ax_b.set_yticks(y, short_names)
    ax_b.tick_params(axis="y", length=0, pad=3)
    ax_b.set_xlabel("Q5 / Q1 RMS")
    ax_b.set_title("b  Scale separation repeats", loc="left", fontweight="bold",
                   fontsize=8.4, pad=6)

    # (c) Strict episode-crossfit: state predicts scale on the held-out fold.
    q = np.arange(1, 6)
    for name in names:
        curve = crossfit[name]["curve"]
        ax_c.plot(q, curve, "o-", color=COLORS[name], lw=1.25, ms=3.1,
                  mec="white", mew=.45)
    ax_c.axhline(1, color="0.55", lw=.75, ls=(0, (3, 2)))
    ax_c.set_xlim(.75, 5.25)
    ax_c.set_ylim(.45, 1.55)
    ax_c.set_xticks([1, 3, 5], ["Q1", "Q3", "Q5"])
    ax_c.set_yticks([.5, 1, 1.5])
    ax_c.set_xlabel(r"Predicted $\sigma(o)$ quintile")
    ax_c.set_ylabel("Held-out RMS / mean", labelpad=2)
    ax_c.text(.04, .96, "disjoint episodes", transform=ax_c.transAxes,
              va="top", color="0.35", fontsize=6.8)
    ax_c.set_title("c  Input predicts scale", loc="left", fontweight="bold",
                   fontsize=8.4, pad=6)

    # (d) Late-five success: same Student-t tail, only the scale model changes.
    tasks = ["Tool-Hang", "Transport-MH", "Transport-PH"]
    hetero = [[75.4, 81.6], [45.5, 55.5, 49.0, 48.0], [62.5, 65.0, 62.0]]
    global_scale = [[44.0, 57.6], [38.0, 41.0], [68.0, 67.5]]
    xpos = np.arange(len(tasks))
    offsets = {"Global $\\sigma$": -.13, r"Input-dependent $\sigma(o)$": .13}
    methods = [("Global $\\sigma$", global_scale, "#8C8C8C"),
               (r"Input-dependent $\sigma(o)$", hetero, "#7B61A8")]
    rng = np.random.default_rng(7)
    means = {}
    for method, values, color in methods:
        means[method] = []
        for xi, seeds in zip(xpos, values):
            center = xi + offsets[method]
            jitter = rng.uniform(-.035, .035, len(seeds))
            ax_d.scatter(center + jitter, seeds, s=13, facecolor="white", edgecolor=color,
                         linewidth=.8, zorder=3)
            mean = float(np.mean(seeds))
            means[method].append(mean)
            ax_d.plot([center - .075, center + .075], [mean, mean], color=color,
                      lw=3.0, solid_capstyle="round", zorder=2)
    for xi, g, h in zip(xpos, means["Global $\\sigma$"], means[r"Input-dependent $\sigma(o)$"]):
        delta = h - g
        color = "#7B61A8" if delta > 0 else "0.42"
        ax_d.text(xi, max(g, h) + 4.0, f"{delta:+.0f}", ha="center", va="bottom",
                  color=color, fontsize=7.5, fontweight="bold")
    ax_d.set_xlim(-.45, 2.45)
    ax_d.set_ylim(25, 90)
    ax_d.set_yticks([40, 60, 80])
    ax_d.set_xticks(xpos, ["Tool-\nHang", "Trans.\nMH", "Trans.\nPH"])
    ax_d.set_ylabel("Success (%)", labelpad=2)
    ax_d.set_title("d  A global scale can fail", loc="left", fontweight="bold",
                   fontsize=8.4, pad=6)
    ax_d.legend([plt.Line2D([0], [0], color="#8C8C8C", lw=3),
                 plt.Line2D([0], [0], color="#7B61A8", lw=3)],
                [r"Learned global $\sigma$", r"$\sigma(o)$"], frameon=False, loc="lower left",
                bbox_to_anchor=(-.03, -.02), ncol=2, handlelength=1.3,
                columnspacing=.8, handletextpad=.35)

    fig.savefig(OUT / "fig_hetero_necessity.pdf", bbox_inches="tight", pad_inches=.025)
    fig.savefig(OUT / "fig_hetero_necessity.png", dpi=320, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)

    payload = {
        "data": summary,
        "panel_a_visualization_subset": {
            "source": str((RAW / "toolhang-residuals-5k.npz").resolve()),
            "chunks": int(len(first)),
            "second_half_rms": {f"Q{q + 1}": radii[q] for q in (0, 2, 4)},
            "q5_q1": radii[4] / radii[0],
        },
        "crossfit_input_scale": {
            name: {key: value for key, value in row.items() if key != "curve"}
            | {"normalized_rms_by_predicted_scale_quintile": row["curve"].tolist()}
            for name, row in crossfit.items()
        },
        "control_ablation_late5_success_percent": {
            task: {"global_sigma": g, "input_dependent_sigma": h,
                   "mean_delta": float(np.mean(h) - np.mean(g))}
            for task, g, h in zip(tasks, global_scale, hetero)
        },
        "scope": "Heteroscedastic scale is necessary on Tool-Hang and Transport-MH, but not universally (Transport-PH is the negative control).",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, summary = summarize()
    crossfit = crossfit_curves()
    draw(data, summary, crossfit)
    print(json.dumps({name: {"q5/q1": row["q5_q1_heldout_half_rms"],
                                  "reliability": row["split_half_reliability"],
                                  "ci": row["split_half_reliability_ci95_episode_bootstrap"]}
                      for name, row in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
