# Input-dependent-scale evidence

This directory contains the replacement for the rejected
action-magnitude/progress scatter plot:

- `fig_hetero_necessity.pdf`: vector, ICLR-width figure.
- `fig_hetero_necessity.png`: review image.
- `figure.tex`: inclusion snippet and caption.
- `summary.json`: exact statistics, source hashes, and ablation values.
- `raw/*.npz`: model-free residual-energy dumps copied from the cluster.

## Why the earlier probe was wrong

The earlier figure treated the RMS magnitude of one target action chunk as
"label scale." That quantity is action amplitude, not the conditional radius
of the label distribution. It also used normalized episode progress as a
semantic-phase proxy and Flow sampling spread as a substitute for the
heteroscedastic head's learned scale. Weak correlations with those proxies do
not test whether residual scale depends on the full policy input.

## Stronger measurement

For each 8-step chunk, we approximate its local conditional mean by averaging
the action chunks of the eight nearest proprioceptive states from other
episodes of the same task. Actions use the dataset-wide q01/q99 affine
normalization without clipping. We exclude binary gripper channels; GR1 hand
joint channels remain because they are continuous.

The residual chunk is split into two 4-step halves. Quintiles are defined only
from first-half residual energy, and scale is measured only on the second half.
Thus the displayed high/low separation is held out within the chunk. For
multi-task datasets, ratios are computed per task and summarized by their
median. Split-half reliability is computed on log energy after subtracting the
task-wise median; confidence intervals bootstrap whole episodes within tasks.

| Dataset | chunks | tasks | held-out Q5/Q1 RMS | split-half reliability |
| --- | ---: | ---: | ---: | ---: |
| RoboCasa-GR1 | 29,916 | 24 | 5.11x | 0.949 [0.945, 0.951] |
| LIBERO | 30,000 | 40 | 2.78x | 0.874 [0.867, 0.879] |
| BridgeData V2 | 16,057 | 124 | 1.30x | 0.561 [0.536, 0.579] |
| Tool-Hang | 29,780 | 1 | 2.87x | 0.862 [0.856, 0.869] |
| Transport-MH | 28,472 | 1 | 3.86x | 0.949 [0.947, 0.952] |

The cross-half test matters: a broad marginal histogram could be produced by
heavy tails alone, whereas persistent scale over disjoint action steps shows
that large and small residual radii are structured by the input/chunk.

We then apply a stricter two-fold episode split. For each query episode, both
the local action mean and local scale are computed only from nearest states in
the opposite fold. The scale estimate never sees the query action. Held-out
RMS increases monotonically from predicted Q1 to Q5 in all five datasets:

| Dataset | cross-fit queries | tasks | predicted-scale Q5/Q1 RMS |
| --- | ---: | ---: | ---: |
| RoboCasa-GR1 | 30,000 | 24 | 2.61x |
| LIBERO | 30,000 | 40 | 1.61x |
| BridgeData V2 | 14,282 | 32 | 1.30x |
| Tool-Hang | 30,000 | 1 | 2.50x |
| Transport-MH | 30,000 | 1 | 2.08x |

## Causal control

The last panel uses a matched RoboMimic component ablation. Both methods use
the same Student-t residual shape; the control replaces the per-input
`sigma(o)` with one learned global scale. Values are late-five success rates:

| Task | learned global sigma | input-dependent sigma(o) | difference |
| --- | ---: | ---: | ---: |
| Tool-Hang | 50.8 (two seeds) | 78.5 (two seeds) | +27.7 |
| Transport-MH | 39.5 (two seeds) | 49.5 (four seeds) | +10.0 |
| Transport-PH | 67.8 (two seeds) | 63.2 (three seeds) | -4.6 |

This supports the scoped claim: input-dependent scale is necessary where a
single scale materially misprices training examples; it is not guaranteed to
help on every task. The Transport-PH negative control is retained in the main
figure to make that scope explicit.

## Reproduction

The raw dumps were generated on `yuchen-vla-util` with
`scripts/dump_split_half_heterogeneity.py`. Regenerate the figure locally with:

```bash
.venv/bin/python scripts/plot_hetero_necessity.py
```

The plotting script records SHA-256 hashes of all raw inputs. The PDF uses
embedded vector fonts; the KDE in panel (a) is based on a deterministic
5,000-chunk visualization subset, while panel (b) uses the full dumps and
panel (c) uses the disjoint-episode cross-fit dumps above.
