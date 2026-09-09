"""Audit all available general-MSE tasks; no policy-held-out claim.

Label magnitude and residual use identical continuous channels/executed steps.
Progress is tested descriptively by ranking stages on one episode half and
measuring their residual scale on the other half, then swapping folds.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_mse_stage_scale import load_npz, metric_arrays, task_matrices, replication

OUT = Path(__file__).resolve().parent / "crossdataset_residual"
OUT.mkdir(exist_ok=True)
SEED = 20260907
BOOT = 600
COLORS = ["#6D53A5", "#009E73", "#D55E00", "#0072B2", "#56B4E9"]
NAMES = {"gr1": "GR00T / GR1", "pi05": "pi0.5 / LIBERO", "widowx": "GR00T / WidowX",
         "tool_hang": "U-Net / Tool-Hang", "transport": "U-Net / Transport"}

def mean_available(x, axis=0):
    count = np.isfinite(x).sum(axis=axis)
    return np.divide(np.nansum(x, axis=axis), count,
                     out=np.full(np.shape(count), np.nan), where=count > 0)

def load(stack):
    if stack != "widowx":
        path = REPO / "analysis/paper/mse_scale" / f"{stack}.npz"
        z = load_npz(path)
        meta = json.loads(str(z["metadata"]))
        a = z["gt"][:, meta["executed_start"]:meta["executed_start"] + meta["executed_horizon"],
                    meta["continuous_channels"]].astype(float)
        residual = metric_arrays(z, "pred_final")
        energy = np.mean(residual ** 2, axis=1)
        paths = [path]
    else:
        paths = sorted((REPO / "analysis/paper/widowx_heterogeneous_scale/raw").glob("mse_rank*.npz"))
        parts = [load_npz(p) for p in paths]
        meta = json.loads(str(parts[0]["metadata"]))
        z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "metadata"}
        task_names = meta["task_names"]
        z["task"] = np.asarray(task_names)[z["task_id"].astype(int)]
        z["stage"] = np.minimum(9, (10 * z["progress"]).astype(int))
        z["split"] = np.empty(len(z["episode"]), dtype=int)
        rng = np.random.default_rng(SEED)
        for task in np.unique(z["task"]):
            eps = np.unique(z["episode"][z["task"] == task])
            first = rng.permutation(eps)[:len(eps) // 2]
            keep = z["task"] == task
            z["split"][keep] = np.isin(z["episode"][keep], first).astype(int)
        a = z["target"].astype(float)
        assert np.allclose(z["residual"], z["prediction"] - z["target"], atol=1e-6)
        energy = np.mean(z["residual"].astype(float) ** 2, axis=(1, 2))
    label = np.sqrt(np.mean(a ** 2, axis=(1, 2)))
    return z, label, energy, paths

rows, groups, sources = [], {}, {}
for si, stack in enumerate(NAMES):
    z, label, energy, paths = load(stack)
    sources[stack] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    rms = np.sqrt(energy)
    assert np.all(np.isfinite(label)) and np.all(np.isfinite(rms))
    task_rows = []
    mats = task_matrices(z, energy)
    reps = replication(mats)
    for ti, (task, matrix, split) in enumerate(mats):
        mask = z["task"] == task
        xx, yy, ep = label[mask], rms[mask], z["episode"][mask]
        eps = np.unique(ep)
        indices = [np.flatnonzero(ep == e) for e in eps]
        rng = np.random.default_rng(SEED + si * 1000 + ti)
        boot = []
        for _ in range(BOOT):
            idx = np.concatenate([indices[j] for j in rng.integers(len(eps), size=len(eps))])
            boot.append(float(spearmanr(xx[idx], yy[idx]).statistic))
        ci = np.nanquantile(boot, [.025, .975])
        fold = [r for r in reps if r["task"] == task]
        profile = np.sqrt(mean_available(matrix))
        radius_ratio = float(np.nanmax(profile) / np.nanmin(profile))
        replica = float(np.exp(np.mean([np.log(r["high_low_ratio"]) for r in fold])))
        row = dict(dataset=stack, task=str(task), n=int(mask.sum()), episodes=len(eps),
                   label_residual_rho=float(spearmanr(xx, yy).statistic),
                   rho_ci_low=float(ci[0]), rho_ci_high=float(ci[1]),
                   progress_monotone_rho=float(spearmanr(z["step"][mask] / (z["length"][mask] - 1), yy).statistic),
                   descriptive_progress_max_min=radius_ratio,
                   cross_episode_progress_high_low=replica,
                   both_progress_folds_above_one=all(r["high_low_ratio"] > 1 for r in fold),
                   progress_fold_ratios=[r["high_low_ratio"] for r in fold],
                   stage_rms=[float(x) if np.isfinite(x) else None for x in profile])
        rows.append(row)
        task_rows.append(row)
    groups[stack] = dict(tasks=len(task_rows), states=len(rms), episodes=sum(r["episodes"] for r in task_rows),
                         mean_task_rho=float(np.mean([r["label_residual_rho"] for r in task_rows])),
                         median_task_rho=float(np.median([r["label_residual_rho"] for r in task_rows])),
                         positive_tasks=sum(r["label_residual_rho"] > 0 for r in task_rows),
                         positive_ci_tasks=sum(r["rho_ci_low"] > 0 for r in task_rows),
                         median_progress_max_min=float(np.median([r["descriptive_progress_max_min"] for r in task_rows])),
                         median_progress_replication=float(np.median([r["cross_episode_progress_high_low"] for r in task_rows])),
                         positive_progress_replication_tasks=sum(r["cross_episode_progress_high_low"] > 1 for r in task_rows),
                         both_folds_positive_tasks=sum(r["both_progress_folds_above_one"] for r in task_rows))
    print(stack, json.dumps(groups[stack]), flush=True)

result = dict(scope="General MSE policies, training-demonstration diagnostics; not policy-held-out.",
              label_metric="RMS of normalized continuous demonstration action; NOT predicted movement magnitude.",
              grouping="Every recorded task is included. Equal task weight in summary rho.",
              bootstrap="600 whole-episode resamples per task; pointwise 95% CI, not multiplicity-adjusted.",
              progress="Rank decile RMS using one episode half; evaluate high/low thirds in the opposite half; geometric mean of both fold ratios. Not a monotonic-correlation test.",
              sources=sources, datasets=groups, per_task=rows)
(OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
with (OUT / "per_task.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                    "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.15), gridspec_kw={"width_ratios": [1, 1]}, layout="constrained")
for si, stack in enumerate(NAMES):
    rr = [r for r in rows if r["dataset"] == stack]
    # Fixed task-name order; vertical offsets avoid overplotting, not a statistical dimension.
    ys = np.full(len(rr), 4-si, dtype=float) + np.linspace(-.22, .22, len(rr)) if len(rr) > 1 else np.array([4-si])
    axes[0].scatter([r["label_residual_rho"] for r in rr], ys, s=13, color=COLORS[si], alpha=.7)
    axes[1].scatter([r["cross_episode_progress_high_low"] for r in rr], ys, s=13, color=COLORS[si], alpha=.7)
axes[0].axvline(0, color=".5", ls="--", lw=.8)
axes[1].axvline(1, color=".5", ls="--", lw=.8)
axes[0].set(yticks=np.arange(5)[::-1], yticklabels=[f"{NAMES[k]} ({groups[k]['tasks']})" for k in NAMES],
            xlabel="Action RMS vs. residual RMS\nSpearman correlation per task", xlim=(-1,1),
            title="a  Action magnitude")
axes[1].set(yticks=np.arange(5), yticklabels=[], xlabel="Other-half residual RMS\nHigh-scale / low-scale stages", title="b  Progress-pattern replication")
for ax in axes:
    ax.set_ylim(-.5,4.5)
    ax.grid(axis="x", alpha=.18)
fig.savefig(OUT / "all_tasks.png", dpi=300)
fig.savefig(OUT / "all_tasks.pdf")
plt.close(fig)

lines = ["# General-MSE residual structure across all available tasks", "",
         "## Main result", "", "All tasks are retained; this is a training-demonstration fit audit, not policy-held-out validation.",
         "Action-label magnitude is not the same quantity as predicted movement magnitude used in some older probes.",
         "", "| Setting | Tasks | Mean task rho | Positive rho tasks | Median other-half progress ratio | Both progress folds >1 |",
         "|---|---:|---:|---:|---:|---:|"]
for k,g in groups.items():
    lines.append(f"| {NAMES[k]} | {g['tasks']} | {g['mean_task_rho']:.3f} | {g['positive_tasks']}/{g['tasks']} | {g['median_progress_replication']:.2f} | {g['both_folds_positive_tasks']}/{g['tasks']} |")
lines += ["", "Each dot in all_tasks.png is one task. Panel a uses the RMS of the demonstrated label and prediction-minus-label residual in identical normalized continuous channels. Binary grippers are excluded; GR1 hand joints are retained.",
          "Panel b ranks 10 progress bins by energy in one episode half, then evaluates the top/bottom thirds on the other half and swaps folds. Values above 1 support repeated stage-scale ordering, not a common increasing/decreasing trajectory shape. Half splits are for diagnostic replication, not policy training.",
          "CSV contains all task names, episode-bootstrap pointwise CIs, both progress fold ratios, and all progress curves. Counts of CIs excluding zero are not multiplicity-adjusted discoveries. No significance claim is made from a ratio merely exceeding 1.",
          "", "## Five-dimension self-review", "",
          "- Contribution: describes cross-task fit structure; does not establish the cause of HT performance gains.",
          "- Clarity: separates demonstration-label magnitude, model-predicted movement, and nonmonotonic progress dependence.",
          "- Experimental strength: every available task retained, including negative and weak associations.",
          "- Completeness: 24 GR1 + 40 LIBERO + 3 WidowX + 2 RoboMimic tasks; no comparable Fractal general-MSE archive included.",
          "- Soundness: identical label/residual channels, complete executed windows, episode-disjoint progress replication. Conditional noise and contact causality not identified.",
          "", "## Claim-evidence map", "",
          "- Action magnitude is associated with fit residuals across settings | All per-task rho values in CSV | Use measured coverage; do not assert every task is strongly positive.",
          "- Scale varies with progress in a task-dependent way | Other-episode-half stage ranking | Report effect sizes and both-fold consistency, not a universal monotonic trend.",
          "- These effects generalize to unseen policy inputs | No policy-held-out data here | Not established.", ""]
(OUT / "README.md").write_text("\n".join(lines))
