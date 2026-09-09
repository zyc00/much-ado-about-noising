# Frozen-MSE residual-scale audit

## Bottom line

The completed probes test the **fitting residuals of existing MSE policies**.
They do not directly recover the noise distribution of human demonstrations.
Residual scale varies across some task-progress and local-state groups, but
the simple hypothesis that larger predicted movements have larger residuals
is not consistently supported. Do not use these results to claim discrete
Gaussian scales, Gaussianity, or a demonstrated cause of MSE's control gap.

The useful motivation is **task-dependent precision**: fine alignment and
free-space transit can impose different tolerances. This is a hypothesis to
test with phase-aligned data, not a consequence of action magnitude alone.
Signal-dependent motor noise and task-constrained variability provide related
human motor-control motivation, not direct robot-dataset evidence:
[Harris and Wolpert (1998)](https://pubmed.ncbi.nlm.nih.gov/9723616/) and
[Todorov and Jordan (2002)](https://www.nature.com/articles/nn963).

## Artifacts

- [ICLR-layout preview](preview.pdf): stage and movement-size figures, with an
  independent-episode local-state diagnostic and interpretation.
- [Stage figure](mse_stage_scale.pdf) / [PNG](mse_stage_scale.png) /
  [LaTeX caption](figure.tex).
- [Movement-size figure](mse_action_magnitude.pdf) /
  [PNG](mse_action_magnitude.png) / [LaTeX caption](figure_magnitude.tex).
- [Independent-episode stage ranking](mse_stage_scale_replication.pdf).
- [Independent-episode local-state ranking](mse_local_scale.pdf).
- [Primary numerical results](summary.json),
  [movement-size results](magnitude_summary.json), and
  [local-state results](local_scale_summary.json).
- [Protocol](PROTOCOL.md), raw predictions `*.npz`, original action inverse
  transforms `action_affine_*.json`, and inference logs in this directory.

The main figure is a diagnostic candidate, not yet a strong demonstration-noise
motivation figure for the paper. The paper-writing skill guided the compact
5.5-inch vector layout and the separation of hypotheses from measured findings.
Existing preliminaries and other manuscript files were not edited.

## Coverage and results

All four complete archives passed ZIP integrity, shape, finite-value,
unique-observation, deterministic-repeat, and executed-chunk validity checks.
There are **15,610 observations from 1,168 episodes** across 66 task settings.

| Setting | Tasks / episodes / observations | Stage range | Independent-half stage high/low | Local-state high/low | Large/small movement |
| --- | ---: | ---: | ---: | ---: | ---: |
| GR00T N1.7 / GR1 RoboCasa | 24 / 288 / 2,880 | 2.47× | 1.22× | 1.22× | 0.71× |
| π0.5 / LIBERO | 40 / 480 / 4,730 | 1.43× | 1.11× | 1.08× | 1.00× |
| Chi-UNet / Tool-Hang | 1 / 200 / 4,000 | 1.25× | 1.18× | 1.23× | 0.84× |
| Chi-UNet / Transport-ph | 1 / 200 / 4,000 | 1.21× | 1.06× | 1.09× | 1.03× |

Movement-size high/low 95% intervals, in table order, are [0.63, 0.77],
[0.97, 1.02], [0.82, 0.87], and [1.00, 1.06]. The exact Transport interval
is [1.00496, 1.05727]: a small positive contrast, not a large multi-scale effect.
Local-state high/low intervals are [1.12, 1.38], [1.06, 1.10], [1.20, 1.26],
and [1.06, 1.12], conditional on the fitted neighborhood estimator.

LIBERO has 480 observations in each of the first nine bins and 410 in the
last bin. Seventy sampled episodes are too short for a complete executed
chunk in their final 10%. The `turn on the stove` task has no eligible final-bin
observations. These cells remain missing (JSON `null`), never zero or padded;
stage statistics use available bins and checkpoint correlations use matched
nonmissing cells.

### Additional checks that constrain interpretation

- Late-checkpoint sample residual-energy rank correlations are 0.9997,
  0.9785, 0.9998, and 0.9997, respectively. Final/late RMS ratios are 0.9994,
  0.9845, 0.9959, and 0.9937. The measured patterns are stable late in training.
- Independent-half Gaussian **stage-scale NLL gains per coordinate over a
  global scale** are −0.16185, −0.00612, +0.00488, and +0.00192. This elementary
  stage-dependent variance fit does not consistently improve predictive
  likelihood; six calibration episodes per VLA task/half can yield noisy scale
  estimates. These negative results are retained, not interpreted as a
  successful heteroscedastic likelihood fit.
- Single-action stage ranges are 2.84×, 1.58×, 1.28×, and 1.28×. Thus the
  descriptive scale variation is not solely a consequence of chunk length.
- After centering each task/stage residual coordinate, stage spread ranges
  are 2.47×, 1.42×, 1.25×, and 1.21×. This is a coarse bias check, not isolation
  of conditional demonstration noise.
- Using physical residuals only in the same channels that define movement
  size gives large/small ratios 0.52×, 1.05×, 0.52×, and 0.94×. Changing from
  normalized all-continuous residuals to matching physical channels does not
  reveal a consistent large-movement/high-residual relationship either.

### What the contrasts mean

1. **Stage range:** compute continuous-channel residual RMS in each of ten
   equal episode-progress bins; take the largest/smallest bin ratio within
   each task, then the median across tasks. This selects extremes on the same
   data and must not be presented as a replicated effect size.
2. **Independent-half stage contrast:** rank stages by residual scale using
   one episode half and measure high/low group RMS on the other half. Swap
   halves and report the median of task/fold ratios. The policy itself was
   trained on both halves.
3. **Local-state contrast:** use proprioceptive nearest neighbors from the
   other episode half to predict residual energy. Group query observations
   by predicted scale without using their target actions or residuals. Report
   the high/low ratio of equally task-weighted RMS curves.
4. **Movement-size contrast:** undo the saved action normalization, then
   group predicted physical movement sizes into five within-task quantiles.
   Report the largest/smallest group ratio of equally task-weighted residual
   RMS curves. Motion size uses translations for LIBERO/RoboMimic and relative
   arm/hand joint offsets for GR1, not absolute waist positions or rot6d offsets.
   The primary residual still includes all continuous action coordinates.

Stage curves are normalized within task and aggregated by their median.
Local-state and movement-size curves use an equal-task arithmetic mean.
Their confidence intervals bootstrap whole episodes within each task, with
task identities and fitted group definitions fixed. Local-state intervals do
not include refitting uncertainty. These are different summaries and need not
give the same contrast.

## Measurement and implementation checks

- Frozen final and nearby late MSE checkpoints are evaluated on identical
  demonstration observations. No policy training was launched.
- Binary grippers are excluded; all 29 GR1 joint channels, including hands,
  remain in the residual metric.
- Only complete executed chunks enter the main metric: GR1 8 steps,
  LIBERO 10, RoboMimic 8 starting at action index 1. Single-step and
  restricted-coordinate checks are also in `summary.json`.
- Repeated deterministic inference on the first batch agrees exactly.
- GR1 first-batch manual MSE agrees exactly with model training-forward loss.
  LIBERO inference MSE and training-forward loss differ by about 2% on the
  first batch; these use different numerical execution paths. This check is
  not described as exact agreement or as proof of a diagnosed cause.
- Saved checkpoint normalization is retained for the VLAs; RoboMimic uses
  the original full-dataset normalizer. Movement magnitudes are converted
  back with the actual inverse action transforms, not normalized vector norms.
- The early 23k Tool-Hang checkpoint was excluded; late 299999/287999
  checkpoints were used instead. Full checkpoint paths are in archive metadata.
- Four unit tests cover complete-chunk sampling, channel/time alignment,
  missing terminal bins, and a synthetic homoscedastic null plus
  heteroscedastic positive control. All passed.
- Initial GR1 launches failed while resolving the offline processor cache;
  the successful v4 launch changes only processor loading to the local cache.
  Checkpoint key loading was checked. Final GR1, LIBERO, and RoboMimic jobs
  all terminated successfully; no live probe GPU jobs remain.
- The two-page preview compiled under the unmodified ICLR style. Both pages
  were visually inspected; labels fit and citations resolve, with no overfull
  or undefined-reference warnings. Tectonic emits a bibliography consistency
  rerun warning despite the resolved bibliography in the final PDF.

## Interpretation for the paper

For a fixed predictor, a conditional residual second moment decomposes as

\[
\mathbb E[(f(o)-a)^2\mid o]
= (f(o)-\mathbb E[a\mid o])^2 + \operatorname{Var}(a\mid o).
\]

Thus unequal MSE residual RMS can reflect conditional spread, prediction bias,
or both. Training-data residuals additionally depend on fitting/memorization.
Coarse within-stage centering removes only a stage-average residual, not all
observation-dependent model error. Late-checkpoint stability is reassuring
about the fit diagnostic but does not eliminate these distinctions.

Heteroscedasticity also does not make population MSE inherently biased for
the conditional mean: its minimizer remains that mean. The question for this
paper is whether a different loss weighting improves learning with finite
data, finite model capacity, and practical optimization.

The appropriate next data experiment would use policy-held-out or cross-fitted
mean predictions and phase labels defined without looking at residual size
(for example transit versus alignment/contact). Repeated or carefully matched
observations would further separate local action spread from mean-estimation
error. Locally standardized residual tails require their own analysis before
motivating Student-t. No new training or such tail analysis was performed here.

### Claim–evidence map

| Claim | Evidence | Status |
| --- | --- | --- |
| Existing MSE fitting residuals can have input-dependent scale | Stage profiles and independently grouped local-state diagnostics | Supported descriptively; strength varies by setting |
| Large movements consistently have larger residual scale | Physical predicted-movement quintiles | Not supported across the settings |
| Fine-control phases have tighter demonstration noise | Human motor-control motivation; no semantic phase/noise identification here | Needs direct evidence |
| Demonstrations consist of discrete Gaussian scale clusters | No Gaussianity or mixture identification test | Not established |
| Heteroscedastic Student-t explains and closes the control gap | No loss ablation or control evaluation in this probe | Must be established by the separate method experiments |

### Five-dimension self-review

- **Contribution:** Does this establish a new data mechanism? Not yet. It
  establishes a reproducible residual diagnostic and rules out an overly
  simple amplitude-based interpretation in these fits.
- **Clarity:** Are RMS, noise, stage, and action magnitude distinguished? Yes;
  progress is explicitly a proxy, and the figures remain separate.
- **Experimental strength:** Are null/reversed results retained? Yes. Raw
  task profiles, replication checks, and all four settings are reported.
- **Evaluation completeness:** Is demonstration-noise identification complete?
  No. Policy-held-out predictions and meaningful phase conditioning remain
  necessary for the stronger claim; tail and control experiments are separate.
- **Method soundness:** Are targets used to select their own scale groups?
  Not in independent-episode or magnitude diagnostics. Coarse-stage extremes
  are explicitly descriptive, not independent confirmation.

## Reproduction

From the repository root, after retrieving the complete archives:

```bash
.venv/bin/python scripts/analyze_mse_stage_scale.py analysis/paper/mse_scale
.venv/bin/python scripts/analyze_mse_local_scale.py analysis/paper/mse_scale
.venv/bin/python scripts/analyze_mse_action_magnitude.py analysis/paper/mse_scale
.venv/bin/python scripts/test_mse_scale_analysis.py
/home/jigu/.local/bin/tectonic --only-cached --keep-logs -Z search-path=/tmp/preliminaries_iclr_J53JYA analysis/paper/mse_scale/preview.tex
```

The last command uses the cached, unmodified ICLR 2027 template; substitute
the location of the official style files if that temporary directory expires.
Cluster outputs and launch scripts are preserved under
`/mnt/pfs/yuchen/mse_stage_scale_20260904`. Pod specifications are included for
provenance, not for automatic relaunch over existing output files.
`pi05_partial.npz` is an explicitly incomplete transfer from an earlier progress
check and is excluded from every final analysis script.
