# Section 1 editorial review

Superseded by `preliminaries.tex` and `preliminaries_review.md`. This note
records the earlier probe-based draft, not the current paper argument.

This is a working review note, not paper text. The revised section is
`section1_single_mode.tex`; the rendered excerpt is `section1_single_mode.pdf`.

## Mini-outline and paragraph roles

1. Opening: explain why averaging incompatible actions is the usual argument
   against MSE and identify the same-input sampling test.
2. Design/evidence: compare one Gaussian with two on GR1 and LIBERO continuous
   channels, using independent projections and held-out likelihood.
3. Intervention: test the mean directly on WidowX over the executed horizon.
4. Challenge: establish the published MSE–Flow gap with one consistent metric.
5. Transition: ask why learning a useful action remains difficult and introduce
   stage-dependent scale and heavy tails as the next objects of study.

The section's message is that the usefulness of a single action does not
explain the effectiveness of the loss used to learn it. The narrative connects
policy-output evidence to a question about learning from demonstrations; it
does not identify those two distributions as the same object.

## Evidence and reporting decisions

The table contains two density comparisons. WidowX supplies a third policy
setting through the mean intervention, rather than a near-zero density score
that would invite a stronger interpretation than its evidence supports.
The table uses normal body size, explicit channels, and one score direction:
positive means an advantage for the single Gaussian.

The original likelihood medians pooled reciprocal split directions. The
revision averages the two directions within each state before taking the
median, matching the stated unit of analysis. From
`gaussianity/action_basin_report.json`, take column 1 of
`models[*].independent_pc1.state_deltas`, negate it, and compute the median:

| Setting | State-median 1G minus 2G | Original direction-median 2G minus 1G |
|---|---:|---:|
| GR1 arm + waist | 0.3305781231 | -0.2791862136 |
| LIBERO arm | 0.3519626454 | -0.2662120032 |

These are reporting changes, not new experiments. Raw samples and analysis
outputs are unchanged. Normality rejection counts remain available in the
existing diagnostic reports. A normality test does not count action modes.

The former "zero confirmed modes" column was removed: a shared, verified
strategy-labeling procedure was not established for all three rows, and one
GR1 body candidate remains unresolved. Its zero could be misread as a measured
absence. The draft also no longer claims that every conditional action is
Gaussian or that a local WidowX path-envelope test proves full-episode mean
policy success.

Table 12 values were checked against the authors' paper,
[arXiv v3](https://arxiv.org/html/2512.01809v3).
The requested OpenReview PDF URL presented a browser challenge. The corresponding
arXiv table contains the cited values. The revised prose reports best-checkpoint
success averaged over three seeds: Sudeep-DiT regression/Flow is 0.12/0.40 on
Transport-mh and 0.52/0.86 on Tool-Hang. Last-five-checkpoint averages are not
mixed with best-checkpoint results.

## Claim–evidence map

- Claim: The median held-out score favors one Gaussian for the tested GR1
  and LIBERO continuous channels. | Evidence: state-aggregated density report
  above. | Status: supported as a model comparison.
- Claim: Mean arm actions preserve the sampled local motion at 40 WidowX
  states. | Evidence: `widowx_mm/widowx_arm_mean_summary.json`,
  `mean_outside_95pct_reference_paths = 0`, four-step execution, matched gripper
  commands. | Status: supported.
- Claim: MSE underperforms Flow on the cited manipulation tasks with the same
  backbone. | Evidence: Pan et al., Table 12. | Status: supported.
- Claim: MSE targets the conditional mean and is a natural objective for a
  Gaussian model. | Evidence: the squared-loss population minimizer and
  fixed-variance Gaussian likelihood. | Status: supported mathematically.
- Claim: Stage-dependent radii and heavy tails explain the observed learning
  gap. | Evidence: to be established by the dataset and mechanism sections.
  | Status: needs evidence; introduced here as an investigation, not a proven
  causal conclusion.

## Five-dimension self-review

- Contribution: pass for motivation. The section poses the question that the
  later mechanism analysis must answer; it does not present the probe as the
  entire contribution.
- Writing clarity: pass. Each paragraph has one role, the table has one score,
  and the closing question connects the evidence to the outline.
- Experimental strength: pass for reporting existing evidence. The density
  comparison is descriptive; the mean claim names its local intervention.
- Evaluation completeness: additional evidence is required for universal
  Gaussianity or an all-policy mean-performance claim. Neither is stated.
  Complete probe protocols and channel ablations belong with the appendix
  material when the full paper is assembled.
- Method soundness: pass for the transition. Non-Gaussianity does not make MSE
  inherently unable to estimate a conditional mean. The proposed explanation
  concerns fitting and gradient allocation and requires later causal tests.

Clarity, paragraph flow, and terminology were checked by reverse outlining.
The ICLR preview uses the unmodified official 2027 style, full-size table text,
and a caption above the table. T1 font encoding ensures the Times family is
used by Tectonic rather than silently substituting its default font. The
section fits on one page; references occupy a separate preview page. The
render was inspected at full size, with no overfull boxes, undefined
references, or font substitution warnings.
