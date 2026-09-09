# Multi-scale residual figure

## Deliverables

- [Vector figure](multiscale_rms.pdf) and [PNG](multiscale_rms.png).
- [ICLR-layout preview](preview.pdf) with the [LaTeX figure/caption](figure.tex).
- [Exact statistics and source hashes](summary.json).
- Generator: `scripts/plot_multiscale_rms.py`; tests:
  `scripts/test_multiscale_rms.py`.

The figure was generated from existing saved outputs; no policy was retrained
or evaluated on new inputs. Existing paper text and earlier figures were not
replaced. The research-paper-writing skill guided the two-panel evidence
structure, native 5.5-inch vector layout, grayscale readability, and bounded
interpretation of training-fit measurements.

## Mini-outline and paragraph roles

- **Observation:** residual RMS spans multiple scales within tasks across four
  trained MSE settings.
- **Learned-scale diagnostic:** a separate GR1 HT model predicts scales that
  track its own observed residual RMS.
- **Motivation:** input-dependent weighting is worth studying; these figures
  do not by themselves establish intrinsic demonstration noise or optimality
  of a particular likelihood.

Suggested evidence/motivation paragraph (also in `preview.tex`):

> Residual magnitudes span multiple scales across demonstration inputs.
> After removing differences in median scale between tasks, the 10th–90th
> percentile range of MSE residual RMS spans 4.6× on GR1, 2.1× on LIBERO,
> 1.9× on Tool-Hang, and 1.8× on Transport. In a separate GR1 HT probe, the
> learned input-dependent scale is strongly associated with measured residual
> RMS (Spearman ρ=0.91). These fitting diagnostics motivate an input-dependent
> scale in the regression objective.

## Panel (a): MSE distributions

Use the frozen final-policy predictions from `../mse_scale/{gr1,pi05,
tool_hang,transport}.npz`: 2,880 / 4,730 / 4,000 / 4,000 observations,
respectively. The original [protocol](../mse_scale/PROTOCOL.md) specifies
checkpoint selection, sampling, action preprocessing, and execution alignment.

For sample i in task t, compute

\[
R_i=\sqrt{\|f(o_i)-a_i\|_2^2/d_i},\qquad
\widetilde R_i=R_i/\operatorname{median}_{j\in t}R_j.
\]

Each task has total weight one divided by the number of tasks. We plot
empirical histograms on common equal-log-width bins, including all samples;
each strip is rescaled to the same peak height. No KDE, Gaussian fit, or
mixture model is used. Thick bars are empirical weighted P10–P90 intervals,
not confidence intervals; black ticks mark the median. The annotated ratio
is weighted P90 divided by weighted P10 of the plotted normalized samples.

| Setting | N | P10 | P90 | P90/P10 |
| --- | ---: | ---: | ---: | ---: |
| GR00T N1.7 / GR1 | 2,880 | 0.5536 | 2.5564 | 4.6181× |
| π0.5 / LIBERO | 4,730 | 0.6817 | 1.3987 | 2.0518× |
| Chi-UNet / Tool-Hang | 4,000 | 0.7580 | 1.4372 | 1.8960× |
| Chi-UNet / Transport-ph | 4,000 | 0.7570 | 1.3378 | 1.7673× |

These differ from the earlier raw, pooled ratios (for example 4.79× for GR1)
because the figure explicitly removes task-wide median-scale differences.
They are also not the same statistic as the median of per-task ratios or
the coarse-stage high/low comparison. No examples were selected by outcome.

## Panel (b): HT scale versus measured residuals

Read the existing cluster archive
`/mnt/pfs/yuchen/groot/resid_dump3_gr1_ht60k_all.npz`, copied locally to
`../mse_scale/ht_gr1_60k_raw.npz`. It contains 2,400 training inputs for
`/mnt/pfs/yuchen/groot/ft_ht3/checkpoint-60000`.
The launch supplied all 24 GR1 datasets, preserving the training
normalization rather than recomputing it from a subset. The archive does
not contain per-input task/episode IDs; it does not establish 24-task sample
coverage or permit episode-bootstrap intervals.

There is one predicted scalar sigma per 8×29 joint-action chunk; hands are
continuous joints and remain included. The effective multivariate Student-t
degrees of freedom is 464. The figure uses eval-mode outputs, not the
separately stored train-mode outputs.

- Sigma P10 / median / P90: 0.0123915 / 0.0293470 / 0.115849.
- Residual RMS P10 / median / P90: 0.0116150 / 0.0293064 / 0.116001.
- Sigma-to-RMS Spearman correlation: 0.907712.
- RMS / sigma P10 / median / P90: 0.67253 / 1.00094 / 1.53872.

Small dots show all inputs. Five equal-count groups are defined by predicted
sigma alone, not by target actions or observed residuals. Each connected
marker is at `(sqrt(mean(sigma²)), sqrt(mean(RMS²)))` within that group.
The dashed reference is `RMS=sigma*sqrt(464/462)`, accounting for Student-t
scale not being exactly its marginal standard deviation. It is a model
reference, not a fitted regression line or a claim of perfect calibration.
All quantities are in normalized action units.

## Claim–evidence check

| Claim | Evidence | Status |
| --- | --- | --- |
| Measured MSE residual magnitudes span multiple scales within tasks | Full empirical distributions and within-task normalization | Supported descriptively |
| Trained HT predicts scale correlated with its own fitting errors | Sigma/residual rank correlation and sigma-defined bins | Supported on the saved training probe |
| Input-dependent regression scale is a motivated design to test | Residual spread plus input-conditioned HT diagnostic | Motivated, not proven necessary or optimal |
| The underlying data are heteroscedastic | Training-fit residuals also contain model error | Not independently established by this figure |
| Residuals form discrete Gaussian clusters | No mixture identification or Gaussianity test | Not claimed |
| Precision phases have lower noise or HT generalizes calibrated uncertainty | No semantic phase comparison or policy-held-out probe here | Not claimed |

A broad marginal RMS distribution alone is not a heteroscedasticity test:
correlated coordinates, heavy tails, and mean-model errors can also broaden
it. The HT panel provides a useful input-conditioned fitting check, but it
is not independent of the loss that trained the model. The two panels use
different inputs and are not a matched policy-performance comparison.

## Five-dimension self-review

1. **Contribution:** Does the figure show the relevant phenomenon? Yes,
   scale variation hidden by progress averaging; it does not claim a new
   distribution-identification result.
2. **Clarity:** Are scales, normalization, histogram heights, bars, and
   policy identities explicit? Yes, in axes/caption and this protocol.
3. **Experimental strength:** Are weak cases omitted? No; all four MSE
   settings remain, including modest 1.8–2.1× spreads.
4. **Evaluation completeness:** Is there independent calibration evidence?
   No; held-out scale validation and phase/noise identification remain open.
5. **Method soundness:** Are groups circularly defined by measured errors?
   HT groups use sigma only; task normalization is descriptive, not a
   predictive validation. No unsupported confidence intervals are drawn.

## Validation and reproduction

The generator checks finite positive scales, complete executed targets,
histogram mass, and exact task-weight normalization. It reconstructs HT
RMS from saved signed residuals and sigma from saved softplus outputs, with
relative tolerance 2e-6. Tests cover normalization, task replication
invariance, invalid inputs, tensor reconstruction, and quintile coverage.

The preview compiled under the unmodified ICLR 2027 style with no overfull
boxes, unresolved references, or warnings. The one-page PDF and grayscale
render were visually inspected. Fonts are embedded; the figure remains
vector, including the point cloud.

```bash
.venv/bin/python scripts/plot_multiscale_rms.py
.venv/bin/python scripts/test_multiscale_rms.py
/home/jigu/.local/bin/tectonic --only-cached --keep-logs -Z search-path=/tmp/preliminaries_iclr_J53JYA analysis/paper/multiscale_rms/preview.tex
```

Substitute the official template location if the cached temporary directory
expires. `summary.json` records SHA-256 hashes of all source archives.
