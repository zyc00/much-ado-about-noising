# 20k evaluation followed by fresh 7k-hold training

## Protocol amendment: user requested unseeded evaluation

The user subsequently requested the same unseeded protocol as the original
10k evaluation and GR00T reference. This supersedes the seeded evaluation
plan below, which is retained as history. Do not combine the two protocols.

- Old 20k checkpoints are re-evaluated in `eval-20000-unseeded` for both datasets.
- Fractal's unfinished seeded evaluation is stopped; all PFS logs/results remain.
- `yuchen-fr-unseed20k-0908` evaluates Fractal on GPUs 0–5 and WidowX on GPUs
  6–7 (sequential task queues), then starts fresh Fractal 7k-hold training.
- WidowX's existing fresh training is not interrupted.
- New runs' 18k evaluations also use unseeded mode.
- No `--seed` argument and `GR00T_EVAL_SEED` is removed from child environments.
  Five vector environments, max 300 steps, requested 100/50 episodes and
  1/4 executed action steps for Fractal/WidowX are retained from the 10k protocol.
- Summaries record requested and actual episode counts. Unseeded deterministic
  runs can repeat scenes across vector environments; these are reference-protocol
  scores, not evidence of 100 independent scenes or paired comparisons to seeded runs.

## Original plan (superseded for evaluation only)

Requested: skip the old 16k evaluation, evaluate the old 20k checkpoint;
Fractal must finish existing training first. Each dataset then starts its
own fresh run from `/mnt/pfs/yuchen/groot/model`, not a staircase checkpoint.

Schedule (optimizer update numbers, inclusive):

| Updates | Joint Student-t nu |
| --- | ---: |
| 1–7000 | 1024 |
| 7001–8000 | 512 |
| 8001–9000 | 256 |
| 9001–10000 | 128 |
| 10001–11000 | 64 |
| 11001–12000 | 32 |
| 12001–18000 | 14 |

Eight GPUs, global batch 1024, LR 1e-4, 1000-update LR warmup, weight
decay 1e-5, saves every 1000 updates. Original per-dataset state dropout,
sigma bias, full datasets, sampling, augmentations and joint-t formulation
are unchanged. The LR schedule spans the new 18k total duration.

Evaluation: seed 1234, five vector environments, maximum 300 steps;
WidowX 50 episodes/task and 4 executed actions, Fractal 100/task and one
executed action. Seeded results are stored separately from the interrupted
old unseeded 16k evaluation. This does not make unseeded published numbers
paired comparisons. All seven WidowX and six Fractal tasks are evaluated.

Cluster code: `/mnt/pfs/yuchen/groot/nu_hold7k_20260908`.
Old runs: `ft_wx_nustair_20260907`, `ft_fr_nustair_20260907`, under
`/mnt/pfs/yuchen/groot`. Old 20k results: `eval-20000-seed1234`.
Each old run's `hold7k_pipeline_status.json` records the pipeline phase.

New runs: `ft_wx_nu1024_hold7k_to14_20260908` and
`ft_fr_nu1024_hold7k_to14_20260908`. Actual nu transitions are recorded in
`nu_schedule.jsonl`. After 18k training, the same seeded evaluator runs again.
Any evaluation failure stops the chain before new training; errors are
recorded in the pipeline status instead of silently advancing.

Fractal's existing training process is untouched. Its future evaluator
entrypoint redirects to the new pipeline. The original evaluator remains
in the original run's audit directory and an additional backup alongside
the cluster entrypoint. WidowX uses a seven-GPU evaluation-only pod after
cancelling its old 16k evaluation; PFS checkpoints and partial results remain.
Its eight-GPU training pipeline is queued and reuses the completed seeded
20k summary after validating task identities, episode counts, seed and nu.
This allows the separately queued one-GPU matched-Flow diagnostic to finish
without interruption while the seven-task evaluation runs concurrently.
