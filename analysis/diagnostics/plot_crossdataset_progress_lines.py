"""Keep the action-correlation data; replace progress summary dots with curves.

Input is the existing 69-task audit, without modifying any source file.
Highlights are sampled by task identity only with one fixed RNG seed.
"""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent / "crossdataset_residual"
SOURCE = Path(os.environ.get("CROSSDATASET_SUMMARY", str(ROOT / "summary.json")))
PREFIX = os.environ.get("CROSSDATASET_FIGURE_PREFIX", "all_tasks_progress_lines")
source_bytes = SOURCE.read_bytes()
data = json.loads(source_bytes)
EXPANDED = "fractal" in data["datasets"]
STACKS = ["gr1", "pi05", "bridge", "fractal"] if EXPANDED else ["gr1", "pi05", "widowx"]
NAMES = ["RoboCasa-GR1", "LIBERO", "Bridge", "Fractal"][:len(STACKS)]
ROW_COLORS = ["#7657A5", "#009E73", "#D97924", "#C99714"]
LINE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#9C609C", "#C99714"]
SEED = 20260907
rng = np.random.default_rng(SEED)
manifest = {"source": str(SOURCE), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "selection_seed": SEED, "selection": "Up to five tasks sampled uniformly without replacement in fixed dataset/task order; no residual-based selection.",
            "normalization": "Each stage RMS divided by sqrt(mean squared stage RMS over available stages), with equal stage weighting.",
            "scope": "General-MSE training-demonstration residuals. Descriptive stage curves, not policy-held-out validation or a statistical significance test.",
            "datasets": {}}

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42})
fig = plt.figure(figsize=(5.5, 5.1 if EXPANDED else 4.1))
gs = fig.add_gridspec(len(STACKS), 2, width_ratios=[1.05, 2.15], left=.26, right=.985,
                      bottom=.19, top=.86, wspace=.23, hspace=.34)
axs = []
x = np.arange(5, 100, 10)
for si, stack in enumerate(STACKS):
    rows = [r for r in data["per_task"] if r["dataset"] == stack]
    chosen = np.sort(rng.choice(len(rows), size=min(5, len(rows)), replace=False))
    chosen_set = set(chosen.tolist())
    palette = {int(j): LINE_COLORS[k] for k, j in enumerate(chosen)}
    if stack == "widowx":
        palette = {j: {"sweep into pile": "#0072B2", "open the drawer": "#D55E00", "close the drawer": "#009E73"}[rows[j]["task"]]
                   for j in chosen_set}
    if len(rows) == 1:
        palette[0] = ROW_COLORS[si]
    curves = []
    for row in rows:
        v = np.array([np.nan if a is None else a for a in row["stage_rms"]], dtype=float)
        denom = float(np.sqrt(np.nanmean(v ** 2)))
        curves.append(v / denom)
    manifest["datasets"][stack] = {"task_count": len(rows), "highlight_count": len(chosen),
                                  "tasks": [{"task": r["task"], "highlighted": j in chosen_set,
                                             "color": palette.get(j, "#9CA3AA"),
                                             "normalized_stage_rms": [float(v) if np.isfinite(v) else None for v in curves[j]]}
                                            for j, r in enumerate(rows)]}
    ax = fig.add_subplot(gs[si, 0])
    bx = fig.add_subplot(gs[si, 1])
    axs.append((ax, bx))
    ys = np.linspace(-.23, .23, len(rows)) if len(rows) > 1 else np.zeros(1)
    ax.scatter([r["label_residual_rho"] for r in rows], ys, color=ROW_COLORS[si],
               s=13, alpha=.82, linewidths=.35, edgecolors="white", zorder=3)
    ax.axvline(0, color=".65", lw=.7, ls=(0, (3, 3)))
    ax.set(xlim=(-1, 1), ylim=(-.5, .5), yticks=[], xticks=[-1, 0, 1])
    unit = "groups" if stack in ["bridge", "fractal"] else "tasks"
    ax.set_ylabel(NAMES[si] + f"\n({len(rows)} {unit})", rotation=0,
                  ha="right", va="center", labelpad=8, fontsize=8)
    ax.spines["left"].set_visible(False)
    for j, vals in enumerate(curves):
        if j not in chosen_set:
            bx.plot(x, vals, color="#929BA4", lw=.65, alpha=.22, zorder=1)
    for j in chosen:
        bx.plot(x, curves[j], color=palette[int(j)], lw=1.15, alpha=.95,
                marker="o", ms=2.2, markeredgecolor="white", markeredgewidth=.35, zorder=3)
    bx.axhline(1, color=".6", lw=.65, ls=(0, (3, 3)), zorder=0)
    bx.set(xlim=(0, 100), ylim=(0 if EXPANDED else .3, 2.05), xticks=[0, 50, 100], yticks=[.5, 1, 1.5, 2])
    bx.grid(axis="y", color=".92", lw=.5, zorder=0)
    if si < len(STACKS) - 1:
        ax.tick_params(labelbottom=False)
        bx.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r"Spearman $\rho$", labelpad=5)
        bx.set_xlabel("Episode progress (%)", labelpad=5)

fig.text(.26, .96, "a  Action magnitude", ha="left", fontsize=9, fontweight="bold")
fig.text(axs[0][1].get_position().x0, .96, "b  Task-stage profiles", ha="left", fontsize=9, fontweight="bold")
fig.text(.76, .90, "Residual RMS / task stage-scale RMS", ha="center", fontsize=6.8, color=".35")
legend = [Line2D([], [], color="#0072B2", lw=1.2, marker="o", ms=2.5, label="Up to 5 highlighted tasks"),
          Line2D([], [], color="#929BA4", lw=.8, alpha=.5, label="All remaining tasks")]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.62, .022),
           ncol=2, frameon=False, fontsize=6.2, handlelength=1.4, columnspacing=1)
for suffix in ["png", "pdf"]:
    fig.savefig(ROOT / f"{PREFIX}.{suffix}", dpi=300)
plt.close(fig)
(ROOT / ("expanded_progress_lines_manifest.json" if EXPANDED else "progress_lines_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
assert SOURCE.read_bytes() == source_bytes, "Original correlation/statistics data must not change"
assert sum(v["task_count"] for v in manifest["datasets"].values()) == sum(data["datasets"][s]["tasks"] for s in STACKS)

caption = r"""\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{analysis/diagnostics/crossdataset_residual/all_tasks_progress_lines.pdf}
  \caption{\textbf{Action magnitude and task-stage structure of regression residuals.}
  General MSE policies are evaluated on 67 tasks across three large-policy settings:
  GR00T/GR1, $\pi_{0.5}$/LIBERO, and GR00T/WidowX.
  (a) Each point is one task's Spearman correlation between demonstration-action
  RMS and residual RMS in the same continuous channels and executed window.
  (b) Each line shows residual RMS in ten episode-progress bins, normalized by
  the root mean square of the task's available bin scales. Each bin aggregates
  residual energy with equal episode weighting. The horizontal reference is one.
  All tasks in these three settings are shown; up to five per setting are highlighted using a fixed random
  seed, with other tasks in light gray. Colors distinguish highlighted tasks within
  each setting. Missing bins remain missing, and no smoothing is applied.
  These are descriptive diagnostics on training demonstrations.}
  \label{fig:crossdataset-residual-profiles}
\end{figure}
"""
if EXPANDED:
    caption = caption.replace("all_tasks_progress_lines.pdf", PREFIX+".pdf").replace(
        "67 tasks across three large-policy settings:\n  GR00T/GR1, $\\pi_{0.5}$/LIBERO, and GR00T/WidowX.",
        "RoboCasa-GR1, LIBERO, Bridge, and Fractal. Bridge and Fractal panels show all nonempty instruction groups with at least 12 sampled episodes; rarer groups contribute to separately saved dataset-wide statistics.").replace(
        "All tasks in these three settings are shown;", "All qualifying tasks/groups in these four settings are shown;")
(ROOT / ("expanded_progress_lines_figure.tex" if EXPANDED else "progress_lines_figure.tex")).write_text(caption)
note = """# Progress-line figure

Mini-outline: action-size association across tasks -> task-specific nonmonotonic stage profiles.

- Original summary.json, per_task.csv, all_tasks.png/pdf and all raw arrays remain unchanged.
- New outputs: all_tasks_progress_lines.png/pdf; progress_lines_manifest.json records every line, selected task, color and source checksum.
- All 67 tasks from the three large-policy settings included; GR1 and LIBERO highlight five tasks each, sampled uniformly with fixed seed 20260907. No task selected by amplitude, shape, rho or significance.
- Tool-Hang and Transport removed from the displayed figure at the user's request. Their original data and the 69-task audit remain unchanged; the previous five-row PNG/PDF and selection manifest are archived with a _5rows suffix.
- Mean residual energy is computed within each episode/progress bin, then averaged across episodes, and square-rooted (existing audit). Normalize each task's ten-bin curve by its root-mean-square bin scale, excluding missing bins.
- All three progress panels use the same y range [0.3, 2.05], containing every valid value. Gray alpha 0.22, color alpha 0.95. No smoothing/interpolation or fabricated terminal bins.
- Dataset episodes/valid windows are inherited from the original audit. The figure is not a semantic contact annotation or policy-held-out experiment.

Five-dimension self-review:
1. Contribution: illustrates existing fitting structure; no claim that visualization proves HT causality.
2. Clarity: fixed y scale and unit reference; per-task normalization and highlighted/background lines explained in caption.
3. Experimental strength: all tasks within the displayed datasets retained, weak/flat curves visible, no outcome-based highlight selection.
4. Completeness: 24 GR1, 40 LIBERO, 3 WidowX displayed; RoboMimic data retained separately. No Fractal or small-policy generality claim from this figure.
5. Method soundness: source checksum preserved, missing values retained; no smoothing, no monotonicity assumption, no inferential claim from visual differences.

Claim-evidence: Stage-resolved residual profiles can vary nonmonotonically | all per-task measured bin RMS curves | descriptive support, not proof of generalization to unseen demonstrations.
"""
if EXPANDED:
    note = "# Expanded dataset figure\n\nDataset labels: RoboCasa-GR1, LIBERO, Bridge, Fractal. See expanded_summary.json for sampling coverage and all_instruction_groups.json for every sampled group, including those too sparse for multi-episode task profiles.\n\n" + "\n".join(
        line for line in note.splitlines() if not any(token in line for token in ["67 tasks", "three", "24 GR1, 40 LIBERO, 3 WidowX"]))
    note += "\n\nAll four progress panels use the same y range [0, 2.05], including every measured value (minimum 0.053 in Fractal). No curves are clipped.\n"
(ROOT / ("expanded_progress_lines_notes.md" if EXPANDED else "progress_lines_notes.md")).write_text(note)
print(ROOT / f"{PREFIX}.png")
