# Progress-line figure

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
