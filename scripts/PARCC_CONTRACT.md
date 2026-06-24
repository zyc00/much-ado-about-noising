# PARCC artifact contract

The Ubuntu/Mac `parcc-*` clients are dumb relays: they fetch files **by path and
name**. If training writes artifacts elsewhere or with other names, `parcc-tail`,
`parcc-watch`, and `parcc-fetch-ckpt` will run successfully but pull nothing.
This file is the single source of truth for what training must produce.

## Paths (Mac `config.env`)

- `PARCC_CODE`  — code mirror; this is the sbatch working dir.
- `PARCC_DATA_ROOT` — datasets, passed to the job as `$2`.
- `PARCC_RUNS`  — run outputs, passed to the job as `$3`.

## What the job receives

`parcc-mac-submit` runs (from `PARCC_CODE`):

```
sbatch --parsable --job-name=$JOBNAME ... scripts/train.slurm \
    <CONFIG> <PARCC_DATA_ROOT> <PARCC_RUNS>
```

So inside `train.slurm`: `$1=config`, `$2=data_root`, `$3=runs_root`, and
`$SLURM_JOB_ID` is the job id.

## What the job MUST write

### 1. Logs — for `parcc-tail` / `parcc-watch`

A log file under `${PARCC_CODE}/logs/` whose name contains the job id.
`scripts/train.slurm` does this via `#SBATCH --output=logs/%x-%j.out`
(`%j` = job id). `parcc-mac-tail-job <JOBID>` waits for `logs/*<JOBID>*.out`.

### 2. Run artifacts — under `${PARCC_RUNS}/${SLURM_JOB_ID}/`

The fetch modes pull these exact globs (no arbitrary `epoch_*.pt`):

| file | written by |
|---|---|
| `config.yaml` | train.slurm copies `$CONFIG` |
| `git_commit.txt` | train.slurm `git rev-parse HEAD` |
| `manifest.json` | **training code** (see below) |
| `normalizer*` / `obs_rms*` | **training code** (save normalizer stats) |
| `best*.pt` / `best*.pth` / `best*.ckpt` | **training code** (best checkpoint) |
| `last*.pt` / `last*.pth` / `last*.ckpt` | **training code** (latest checkpoint) |
| `*.json *.jsonl *.csv *.png *.mp4` `events.out.tfevents.*` | training code / tensorboard |

`best`/`last` fetch the small set; `all-light` adds metrics/plots/videos but
still only `best*`/`last*` weights.

## TODO on the training side (this repo)

`examples/train_robomimic.py` currently logs to W&B. To satisfy the contract it
must additionally, into its run/output dir:

1. accept a run/output dir override (the `log.output_dir`/`hydra.run.dir` keys in
   `train.slurm` are placeholders — wire them to the real config schema);
2. save `best*.pt` and `last*.pt` checkpoints;
3. dump the dataset normalizer as `normalizer.pt` (or `obs_rms*`);
4. write a small `manifest.json` (config name, commit, metrics, ckpt filenames);
5. set `WANDB_MODE=offline` if compute nodes lack internet.

Until (2)–(4) exist, `parcc-fetch-ckpt ... best` returns an empty dir.
