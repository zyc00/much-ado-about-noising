# GR1 convergence curve

Regenerate from the repository root:

```bash
python scripts/plot_convergence_curve.py
```

`results.csv` contains the task-level-log aggregation used by the plot. Every
plotted checkpoint was evaluated on all 24 RoboCasa-GR1 tasks with 20 episodes
per task. The runs use GR00T N1.7 and the same data and 60k-step training
schedule; only the objective differs.

The paper plot keeps the intended 2k, 16k, 32k, and 60k checkpoints; Flow has
no retained 32k checkpoint. The HT curve uses the measured $\nu=2$ checkpoints
at 2k, 16k, and 32k, followed by the paper-setting $\nu=4d$ endpoint at 60k.
This merge is stated explicitly in the figure caption because the early
$\nu=4d$ checkpoints were rotated. Dense evaluations from 50k--58k remain in
`results.csv` for provenance but are intentionally omitted from the paper
figure. Lines connect displayed values and are not smoothed.
