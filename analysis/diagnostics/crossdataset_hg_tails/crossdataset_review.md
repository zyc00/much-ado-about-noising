# Figure evidence review

## Claim–evidence map

- Claim: Excess tails remain after the learned HG scalar scale is removed.
  Evidence: Per-task cross-fitted absolute-coordinate residual tails exceed the
  standard Gaussian 3-sigma frequency in 24/24 GR1 tasks and 40/40 probed Bridge groups.
  Status: supported within these measured checkpoints and demonstration probes.
- Claim: This is not solely an artifact of pooling tasks with different scales.
  Evidence: Coordinate centering and scaling are fitted separately within each
  task, excluding evaluated episode folds; tail ratios are reported per task.
  Status: supported for this pooling concern; not a guarantee of perfect conditional
  scale estimation by the HG model.
- Claim: The full residual distribution is exactly Student-t, or the fitted scalar
  degrees of freedom specify the joint training loss's degrees of freedom.
  Evidence: not established. Status: excluded from the figure and caption.

## Five-dimension self-review

1. Contribution: Does this add more than a single illustrative curve?
   Yes: all 24 GR1 tasks and all 40 frequency-selected Bridge groups remain visible.
2. Clarity: Can the reader identify what a point or curve represents?
   Yes: one task, with matched colors across columns. The caption explains sorting,
   highlighted-rank selection, the 0.27% reference, channels and normalization.
3. Experimental strength: Is this supported by actual policy outputs?
   Yes: pure-HG checkpoints, preserved IDs, exact deployment/forward agreement on
   the first GR1 batch, source hashes and episode-clustered uncertainty.
4. Evaluation completeness: Are coverage and held-out scope honest?
   Yes: 24 GR1 tasks and the 40 most frequent eligible Bridge groups; calibration holdout is explicitly
   distinguished from policy holdout. Neither input-level generalization nor all
   Bridge tasks is claimed. Bridge labels are exact instruction groups, not 40
   independent benchmark task definitions.
5. Method soundness: Are the reference distribution and calibration appropriate?
   The statistic pools scalar coordinates after within-task coordinate calibration,
   not multivariate norms; the Gaussian reference is therefore scalar. Calibration
   uses other episodes and no final evaluation-set RMS rescaling. Bootstrap intervals
   are conditional on the fitted calibration. Remaining HG misspecification and
   finite-sample calibration are not ruled out by this diagnostic.

## Presentation checks

- No artificial distributions, fabricated samples, or substituted MSE/HT rows.
- No fitted Student-t curves presented as empirical measurements.
- No tail-dependent task exclusion; example selection is documented and all other
  tasks remain plotted.
- Both rows use the same Gaussian reference and axis limits.
- Separate PNG preview, vector PDF and standalone LaTeX caption are provided.

## Completed verification

All 24 local GR1 raw arrays match the SHA256 hashes of the cluster originals.
Independent local recomputation matches the cluster tail ratios, bootstrap
intervals, empirical curves and Student-t likelihood gains to 1e-10 tolerance.
Both temporary GPU probe processes finished normally. The final PNG was visually
inspected after shortening column headings to remove a title overlap.

The right-panel range was subsequently extended to zero using exact counts from
the saved cross-fitted residuals. All 27 empirical curves have P(|Z|>0)=1;
their values on the original 0.5–6 grid are verified unchanged. No interpolation
or forced endpoint was used, and the Gaussian reference also starts at (0,1).

## Bridge expansion

The original three Bridge probes were retained exactly, then 37 instruction groups
were added using the same pure HG 9k checkpoint. Selection uses eligible episode
counts only, with alphabetical tie breaking; it does not use any residual statistic.
There are 48 episodes per original group and 24 per new group, with 12 full-chunk
positions each (1,032 episodes, 12,384 states total). All 40 measured ratios exceed
one (median 4.77, range 3.08–6.16); the smallest conditional bootstrap lower bound
is 2.09. The unchanged first-three-group statistics were checked against the earlier
archive. Figure and caption disclose group selection and unequal episode counts.
All 40 local raw-array hashes match the cluster archive, and independently
recomputed statistics match to 1e-10 tolerance. Both inference-path checks return
zero difference. The expanded PNG was visually inspected; zero empirical tail
probabilities are omitted on the log axis, as disclosed in the caption.
