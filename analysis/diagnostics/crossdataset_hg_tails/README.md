# Cross-dataset HG heavy-tail evidence

## Current two-dataset figure

`fig_crossdataset_hg_tails.png` / `.pdf`; caption: `fig_crossdataset_hg_tails.tex`.

- RoboCasa-GR1: all 24 task datasets, 24 random demonstrations per task,
  12 valid full-chunk starts per demonstration (576 episodes, 6,912 states).
  All 24 tasks have excess 3-sigma events; task-median ratio 7.80, range 6.34–8.99.
- Bridge/WidowX: the 40 most frequent exact instruction groups among episodes
  of length at least 24. The three original groups retain their 48 episodes;
  the 37 additional groups each use 24 randomly selected episodes. In total:
  1,032 episodes, 12,384 states. All 40 have excess events; median 4.77,
  range 3.08–6.16. This is not all Bridge instruction groups.

The earlier 24-GR1 / 3-Bridge figure is preserved as
`fig_crossdataset_hg_tails_bridge3.png` / `.pdf` / `.tex`. Its three Bridge
groups' original arrays and statistics remain unchanged. The new groups were
selected only by episode counts (alphabetical tie breaking), before probing
their residuals. The scan found 101 groups with at least 24 eligible episodes;
we selected the 40 most frequent, not the 40 strongest tail effects.

Left: every task's measured tail ratio with conditional episode-bootstrap interval.
Right: every task's measured tail curve. The highlighted GR1 tasks are nearest
the 10th, 50th and 90th percentiles of the ratio, not the three largest effects.
The expanded Bridge panel uses the same highlighting rule.
Selection and numeric results are saved in `crossdataset_figure_manifest.json`.
Student-t fits and Laplace comparisons remain in the statistical JSON files;
the overview itself displays only empirical curves and the Gaussian reference.

The new GR1 probe uses the frozen pure-HG checkpoint below, its own processor,
all 29 continuous action channels (including hand joints), and its exact learned
scalar sigma. Forward predictions and deployment predictions agreed exactly on
the audited first batch. No training settings or checkpoints were changed.

Raw GR1 arrays are retained on the cluster at
`/mnt/pfs/yuchen/hg_task_tails_20260908/gr1/hg_task00.npz` through `hg_task23.npz`.
The local `gr1_raw/` mirror is retrieved with SHA256 validation by
`../fetch_hg_raw_chunks.py`; `.partial` files are failed transfers, not inputs.
`gr1_hg_tail_summary.json` contains each cluster source path and its SHA256.
`gr1_raw/protocol.json` lists every selected episode and the checkpoint protocol.

Expanded Bridge raw arrays: `bridge40/raw/hg_task00.npz` through
`hg_task39.npz`. They are mirrored from
`/mnt/pfs/yuchen/bridge_hg40_20260908/raw/` with SHA256 validation.
`bridge40/protocol.json` records all instructions, eligible episode counts and
selected episodes. The cluster-computed statistics are archived in
`bridge40/cluster_summary.json`; the independently recomputed local statistics
and cross-fitted residuals are under `bridge40/stats/`.

Recompute GR1 statistics and redraw, from the repository root:

```sh
python analysis/diagnostics/analyze_gr1_hg_task_tails.py \
  --raw-dir analysis/diagnostics/crossdataset_hg_tails/gr1_raw \
  --output analysis/diagnostics/crossdataset_hg_tails/gr1_stats
python analysis/diagnostics/analyze_gr1_hg_task_tails.py \
  --dataset bridge --expected-tasks 40 \
  --raw-dir analysis/diagnostics/crossdataset_hg_tails/bridge40/raw \
  --output analysis/diagnostics/crossdataset_hg_tails/bridge40/stats
python analysis/diagnostics/plot_crossdataset_hg_tails.py
```

The figure reads the archived `gr1_hg_tail_summary.json` at this directory's root.
The first command writes an independently recomputed copy under `gr1_stats/`,
which can be compared with the archive before replacing it.
The plot now defaults to 40 Bridge groups; `--bridge-groups 3` reproduces the
earlier three-group selection (and writes the standard output filenames).

## Earlier Bridge-only preview

Figure: `fig_bridge_hg_task_tails.pdf` / `.png`.
Reproduce from the repository root with:

```sh
python analysis/diagnostics/plot_hg_task_tails_preview.py
```

## Suggested caption

**Heavier-than-Gaussian residual tails persist after HG scaling in three Bridge tasks.**
Continuous-action residuals from a pure HG checkpoint are divided by its predicted
input-dependent scale, then centered and standardized separately for each task
and action-chunk coordinate using five-fold episode-level cross-fitting.
(a) Frequency of absolute standardized residuals above three, relative to the
standard Gaussian reference. Bars show 95% episode-bootstrap intervals conditional
on the fitted calibration. (b) Empirical two-sided tail probabilities for all three
available tasks, with calibration-fitted Student-t curves and the Gaussian reference.
Colors identify the same tasks in both panels. The probe contains 144 demonstrations
(48 per task); binary gripper residuals are excluded.

## Scope and review

- All three tasks in this existing HG probe are displayed; the earlier preview is
  not a full-Bridge or cross-dataset result. The new GR1 probe described above adds
  the task/episode identities that were absent from the older GR1 residual dumps.
- Calibration and density fitting exclude the evaluated episode fold. The policy
  itself was trained on demonstrations; these are not established policy-held-out
  episodes. No final rescaling uses pooled evaluation residuals.
- The statistic is the absolute value of each scalar coordinate, **not** a vector
  L2 norm. Coordinates and chunk steps are pooled within task after calibration.
- The model sigma is preserved as predicted, including its original training loss's
  channel aggregation. Removing binary gripper residuals does not redefine sigma.
- Compared with the older pooled-task figure, task-specific centering and calibration
  change the numerical results. Both the protocol and source hash are in
  `bridge_hg_tail_summary.json`; cross-fitted observations are saved separately.
- Student-t improves average cross-fitted log likelihood over Gaussian for all three
  tasks, but its plotted far tails overestimate observed frequencies. This figure
  supports excess tails relative to Gaussian, not an exact Student-t law or Student-t
  superiority to every alternative. Laplace likelihoods are also saved in the JSON.
- A fitted scalar Student-t degree of freedom does not directly prescribe the joint
  action-chunk loss's degree of freedom.

Source probe:
`analysis/paper/widowx_heterogeneous_scale/raw/hg_rank0.npz`

Pure HG checkpoints verified for this preview:

- Bridge/WidowX: `/mnt/pfs/yuchen/groot/ft_wxpurehg/checkpoint-9000`
  (`ht_hg_steps=20000`).
- RoboCasa/GR1: `/mnt/pfs/yuchen/groot/ft_gr1purehg/checkpoint-22000`
  (`ht_hg_steps=60000`).

Both checkpoint steps precede their HG-to-HT switch. Do not substitute the later
`ft_gr1hgpre/checkpoint-22000`, whose configured switch is at 9000 steps.
