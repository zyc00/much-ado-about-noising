# Expanded dataset figure

Dataset labels: RoboCasa-GR1, LIBERO, Bridge, Fractal. See expanded_summary.json for sampling coverage and all_instruction_groups.json for every sampled group, including those too sparse for multi-episode task profiles.

# Progress-line figure

Mini-outline: action-size association across tasks -> task-specific nonmonotonic stage profiles.

- Original summary.json, per_task.csv, all_tasks.png/pdf and all raw arrays remain unchanged.
- New outputs: all_tasks_progress_lines.png/pdf; progress_lines_manifest.json records every line, selected task, color and source checksum.
- Tool-Hang and Transport removed from the displayed figure at the user's request. Their original data and the 69-task audit remain unchanged; the previous five-row PNG/PDF and selection manifest are archived with a _5rows suffix.
- Mean residual energy is computed within each episode/progress bin, then averaged across episodes, and square-rooted (existing audit). Normalize each task's ten-bin curve by its root-mean-square bin scale, excluding missing bins.
- Dataset episodes/valid windows are inherited from the original audit. The figure is not a semantic contact annotation or policy-held-out experiment.

Five-dimension self-review:
1. Contribution: illustrates existing fitting structure; no claim that visualization proves HT causality.
2. Clarity: fixed y scale and unit reference; per-task normalization and highlighted/background lines explained in caption.
3. Experimental strength: all tasks within the displayed datasets retained, weak/flat curves visible, no outcome-based highlight selection.
5. Method soundness: source checksum preserved, missing values retained; no smoothing, no monotonicity assumption, no inferential claim from visual differences.

Claim-evidence: Stage-resolved residual profiles can vary nonmonotonically | all per-task measured bin RMS curves | descriptive support, not proof of generalization to unseen demonstrations.

All four progress panels use the same y range [0, 2.05], including every measured value (minimum 0.053 in Fractal). No curves are clipped.
