"""Less crowded rendering of the existing full-dataset audit; no new probes."""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent / "crossdataset_residual"
SOURCE = ROOT / "expanded_summary.json"
raw = SOURCE.read_bytes()
data = json.loads(raw)
selection = json.loads((ROOT / "expanded_progress_lines_manifest.json").read_text())
assert hashlib.sha256(raw).hexdigest() == selection["source_sha256"]
STACKS = ["gr1", "pi05", "bridge", "fractal"]
NAMES = ["RoboCasa-GR1", "LIBERO", "Bridge", "Fractal"]
COLORS = ["#7657A5", "#009E73", "#D97924", "#C99714"]
DOT_LINE = os.environ.get("CROSSDATASET_DOT_LINE") == "1"
MEAN_MEDIAN = os.environ.get("CROSSDATASET_MEAN_MEDIAN") == "1"
TOP3 = os.environ.get("CROSSDATASET_TOP3") == "1"
MEAN_MEDIAN = MEAN_MEDIAN or TOP3
DOT_LINE = DOT_LINE or MEAN_MEDIAN
PREFIX = "all_datasets_progress_dots_lines" if DOT_LINE else "all_datasets_progress_clean"
if MEAN_MEDIAN:
    PREFIX = "all_datasets_progress_mean_median"
if TOP3:
    PREFIX = "all_datasets_progress_top3"
dot_rng = np.random.default_rng(20260907)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .6, "pdf.fonttype": 42})
fig = plt.figure(figsize=(5.5, 4.55))
gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.8], left=.24, right=.96,
                      bottom=.19, top=.83, wspace=.3, hspace=.28)
manifest = {"source_sha256": hashlib.sha256(raw).hexdigest(),
    "selection": "Unchanged from expanded_progress_lines_manifest.json",
    "background": ("All non-highlighted tasks shown as faint dashed lines." if DOT_LINE else
                   "Pointwise 10th–90th percentiles across all qualifying tasks; not a confidence interval."),
    "datasets": {}}
x = np.arange(5, 100, 10)
for i, (stack, name, color) in enumerate(zip(STACKS, NAMES, COLORS)):
    rows = [r for r in data["per_task"] if r["dataset"] == stack]
    tasks = selection["datasets"][stack]["tasks"]
    assert [r["task"] for r in rows] == [t["task"] for t in tasks]
    rho = np.asarray([r["label_residual_rho"] for r in rows])
    curves = np.asarray([[np.nan if v is None else v for v in t["normalized_stage_rms"]] for t in tasks])
    spans = np.nanmax(curves, axis=1) - np.nanmin(curves, axis=1)
    top3_indices = sorted(range(len(tasks)), key=lambda j: (-spans[j], tasks[j]['task']))[:3]
    lo, hi = np.nanpercentile(curves, [10, 90], axis=0)
    q10, median, q90 = np.quantile(rho, [.1, .5, .9])
    ax = fig.add_subplot(gs[i, 0])
    bx = fig.add_subplot(gs[i, 1])
    # Boundary-reflected KDE, displayed only on the valid correlation support.
    grid = np.linspace(-1, 1, 401)
    kde = gaussian_kde(rho)
    density = kde(grid) + kde(-2-grid) + kde(2-grid)
    height = (.50 if MEAN_MEDIAN else .64) * density / density.max()
    ax.fill_between(grid, 0, height, color=color, alpha=.10 if DOT_LINE else .24, lw=0)
    ax.plot(grid, height, color=color, lw=1.15)
    ax.axvline(0, color=".7", lw=.6, ls=(0, (3, 3)), zorder=0)
    if MEAN_MEDIAN:
        pass  # No scatter or quantile bars in this presentation.
    elif DOT_LINE:
        ax.scatter(rho, dot_rng.uniform(-.22, -.04, len(rho)), s=3.5,
                   color=color, alpha=.18 if len(rho) > 100 else .35,
                   edgecolors="none", zorder=2)
    else:
        ax.plot([q10, q90], [-.1, -.1], color=color, lw=2.3, solid_capstyle="round")
        ax.plot([median, median], [-.16, -.04], color=".18", lw=1.15)
    annotation = (f"median {median:.2f}\nmean {rho.mean():.2f}" if MEAN_MEDIAN else f"median {median:.2f}")
    ax.text(.98, .98 if MEAN_MEDIAN else .89, annotation, transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=".35")
    ax.set(xlim=(-1, 1), ylim=(-.28, 1), yticks=[], xticks=[-1, 0, 1])
    ax.spines["left"].set_visible(False)
    unit = "groups" if stack in ["bridge", "fractal"] else "tasks"
    ax.set_ylabel(f"{name}\n{len(rows)} {unit}", rotation=0, ha="right", va="center", labelpad=8)
    if DOT_LINE:
        for t, curve in zip(tasks, curves):
            if not t["highlighted"]:
                bx.plot(x, curve, color="#8D98A3", lw=.5, ls=(0, (2.5, 2)),
                        alpha=.07 if len(tasks) > 100 else .20, zorder=0)
    else:
        bx.fill_between(x, lo, hi, color="#E4E8EB", lw=0, zorder=0)
    bx.axhline(1, color=".65", lw=.65, ls=(0, (3, 3)), zorder=1)
    if TOP3:
        for j, line_color in zip(top3_indices, ['#0072B2', '#D55E00', '#009E73']):
            bx.plot(x, curves[j], color=line_color, lw=1.05, alpha=.85, zorder=2)
    else:
        for t, curve in zip(tasks, curves):
            if t["highlighted"]:
                bx.plot(x, curve, color=t["color"], lw=1.05, alpha=.85, zorder=2)
    bx.set(xlim=(0, 100), ylim=(0, 2.05), xticks=[0, 50, 100], yticks=[.5, 1, 1.5, 2])
    bx.grid(axis="y", color=".94", lw=.45)
    if i < 3:
        ax.tick_params(labelbottom=False)
        bx.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r"Action–residual correlation $\rho$", fontsize=7, labelpad=6)
        bx.set_xlabel("Episode progress (%)", labelpad=6)
    manifest["datasets"][stack] = {"task_count": len(rows), "rho_quantiles": [q10, median, q90],
        "rho_mean": float(rho.mean()),
        "stage_q10": lo.tolist(), "stage_q90": hi.tolist(),
        "highlighted_tasks": [t["task"] for t in tasks if t["highlighted"]]}
    if TOP3:
        manifest['datasets'][stack]['highlighted_tasks'] = [tasks[j]['task'] for j in top3_indices]
        manifest['datasets'][stack]['selected_spans'] = [float(spans[j]) for j in top3_indices]
        manifest['datasets'][stack]['background_tasks_unchanged'] = [t['task'] for t in tasks if not t['highlighted']]
fig.text(.24, .94, "a  Action magnitude", fontsize=9, fontweight="bold")
fig.text(.535, .94, "b  Task-stage profiles", fontsize=9, fontweight="bold")
fig.text(.24, .87, "Task-wise correlation distribution", fontsize=6.5, color=".4")
fig.text(.535, .87, "Relative residual RMS", fontsize=6.5, color=".4")
fig.legend(handles=[Line2D([], [], color="#0072B2", lw=1.2, label="3 high-variation examples" if TOP3 else "5 example tasks"),
    (Line2D([], [], color="#8D98A3", lw=.6, ls="--", alpha=.5, label="Task profiles" if TOP3 else "All remaining tasks")
     if DOT_LINE else Patch(facecolor="#E4E8EB", edgecolor="none", label="All tasks: 10–90% range"))],
    loc="lower center", bbox_to_anchor=(.59, .035), ncol=2, frameon=False,
    fontsize=6.5, handlelength=1.6, columnspacing=1.4)
for ext in ["png", "pdf"]:
    fig.savefig(ROOT / f"{PREFIX}.{ext}", dpi=300)
plt.close(fig)
assert SOURCE.read_bytes() == raw
if TOP3:
    manifest['selection'] = 'Three largest max-minus-min normalized stage-RMS ranges per dataset; effect-selected illustrative examples, not random or independent validation.'
    manifest['background'] = 'Exact original dashed-task identities, values, opacity, line style, and axes retained from mean_median version.'
(ROOT / f"{PREFIX}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
caption = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{analysis/diagnostics/crossdataset_residual/all_datasets_progress_clean.pdf}
\caption{\textbf{Action magnitude and stage-dependent regression residuals across datasets.}
(a) Distributions of task-wise Spearman correlations between action RMS and general-MSE residual RMS. Curves are boundary-reflected kernel density estimates; bars indicate the 10th--90th percentiles and ticks indicate medians.
(b) Residual RMS across ten episode-progress bins, divided by each task's root-mean-square bin scale. Five tasks per dataset are sampled with a fixed seed. Shading shows pointwise 10th--90th percentiles across all qualifying tasks, not confidence intervals; no trajectory smoothing is applied.
Points are plotted at bin centers (5\%--95\%); bins without valid full-action-chunk samples are omitted, so some curves end earlier.
Bridge and Fractal include nonempty instruction groups with at least 12 sampled demonstrations. These diagnostics use training-demonstration residuals.}
\label{fig:crossdataset-residual-clean}
\end{figure}
"""
if DOT_LINE:
    caption = caption.replace("all_datasets_progress_clean.pdf", PREFIX + ".pdf").replace(
        "bars indicate the 10th--90th percentiles and ticks indicate medians.",
        "faint dots show every task's measured correlation, with vertical jitter for visibility.").replace(
        "Shading shows pointwise 10th--90th percentiles across all qualifying tasks, not confidence intervals; no trajectory smoothing is applied.",
        "Faint dashed lines show every remaining task and solid colored lines show the five highlighted tasks; no trajectory smoothing is applied.")
if MEAN_MEDIAN:
    caption = caption.replace("faint dots show every task's measured correlation, with vertical jitter for visibility.",
        "annotations report the empirical median and mean over all qualifying tasks, weighted equally.")
if TOP3:
    caption = caption.replace('Five tasks per dataset are sampled with a fixed seed.',
        'Three tasks per dataset with the largest maximum-minus-minimum normalized stage-RMS range are highlighted as illustrative high-variation examples, not a random sample.')
    caption = caption.replace('Faint dashed lines show every remaining task and solid colored lines show the five highlighted tasks; no trajectory smoothing is applied.',
        'Faint dashed lines show background task profiles; solid colored lines show the three selected task profiles, without temporal smoothing.')
(ROOT / f"{PREFIX}.tex").write_text(caption)
(ROOT / f"{PREFIX}_review.md").write_text("""# Presentation review

Outline: action–residual association across tasks -> nonmonotonic within-task stage profiles.

1. Contribution: descriptive motivation only; does not prove causal benefits of HT.
2. Clarity: replaced dense markers with correlation densities and dense background lines with a labeled percentile band. Band is not a CI.
3. Strength: all qualifying groups enter summaries. Same five randomly chosen task curves as before, no result-based reselection.
4. Completeness: four datasets; group eligibility and training-data scope stated in caption. Full-line figure and raw data retained.
5. Soundness: original source hash checked, normalized measured curves reused unchanged, missing bins retained. KDE is display-only; quantiles use empirical values. Weak LIBERO trends remain visible.

Claim: task-stage profiles vary | Evidence: measured stage RMS, pointwise population percentiles and unchanged examples | Status: descriptive support, not evidence of held-out calibration or uniform effect strength.
""")
print(ROOT / f"{PREFIX}.png")
if DOT_LINE:
    (ROOT / f"{PREFIX}_review.md").write_text("""# Dot-and-line revision review

Outline: task-wise action association -> stage profiles.
1. Contribution: unchanged descriptive motivation, no new scientific claim.
2. Clarity: small translucent dots without outlines beneath a KDE; faint dashed background task curves and solid five-task highlights. No percentile band.
3. Strength: every qualifying task retained; all correlation values and five highlighted tasks unchanged.
4. Completeness: all four datasets and qualifying instruction groups retained. Prior figures untouched.
5. Soundness: original data hash checked; no curve smoothing, no imputed bins; vertical dot jitter is cosmetic. Lower background alpha for larger task counts limits overplotting.
Claim: observed stage-dependent residual variation | Evidence: unchanged measured task curves | Status: descriptive, not held-out or causal validation.
""")
if MEAN_MEDIAN:
    review_path = ROOT / f"{PREFIX}_review.md"
    review = review_path.read_text().replace("small translucent dots without outlines beneath a KDE;", "KDE with empirical median and mean annotations, no scatter;").replace("vertical dot jitter is cosmetic.", "mean and median use all qualifying tasks with equal task weights.")
    review_path.write_text(review)
if TOP3:
    (ROOT / f'{PREFIX}_review.md').write_text('''# Three-highlight revision review

Outline: unchanged population correlation panel -> selected high-variation task profiles.
1. Contribution: illustrative task-stage residual variation, not a universal or causal claim.
2. Clarity: only three solid curves; original dashed background preserved exactly.
3. Strength: ranking by max-minus-min normalized RMS is disclosed as effect-based selection.
4. Completeness: left panel and all original dashed curves, including their identities and missing bins, unchanged.
5. Soundness: no smoothing, imputation, new normalization, or axis changes; original files preserved via a new output prefix.

Claim: examples show stage variation | Evidence: unchanged measured profiles selected by range | Status: descriptive examples, not a prevalence estimate or independent validation.
''')
