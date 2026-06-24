#!/usr/bin/env bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# DRAFT Slurm script for PARCC. Adapt the "environment" block to Betty.
#
# Invoked by parcc-mac-submit as:
#   sbatch ... scripts/train.slurm <CONFIG> <PARCC_DATA_ROOT> <PARCC_RUNS>
#
# Resource flags (--job-name/--partition/--gpus/--time/--cpus-per-task/--mem)
# are passed on the sbatch command line by parcc-mac-submit, NOT here, so the
# Ubuntu/Mac client stays the single source of truth for resources.
#
# CONTRACT (see scripts/PARCC_CONTRACT.md) — the client only fetches what this
# job writes to the agreed locations:
#   * logs   -> ${PARCC_CODE}/logs/<jobname>-<jobid>.out   (handled by #SBATCH above)
#   * run    -> ${RUNS}/${SLURM_JOB_ID}/                   (you must write here)
set -euo pipefail

CONFIG="${1:?usage: train.slurm <config> <data_root> <runs_root>}"
DATA_ROOT="${2:?missing data_root}"
RUNS_ROOT="${3:?missing runs_root}"

JOB="${SLURM_JOB_ID:-manual}"
RUN_DIR="${RUNS_ROOT%/}/${JOB}"
mkdir -p "$RUN_DIR" logs

echo "[train.slurm] job=$JOB host=$(hostname) config=$CONFIG"
echo "[train.slurm] data_root=$DATA_ROOT"
echo "[train.slurm] run_dir=$RUN_DIR"

# --- environment (ADAPT TO BETTY) -----------------------------------------
# module load cuda || true
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate mip
# export WANDB_MODE=offline   # PARCC compute nodes are usually offline

# --- provenance the client fetches ----------------------------------------
git rev-parse HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || echo "unknown" > "$RUN_DIR/git_commit.txt"
cp -f "$CONFIG" "$RUN_DIR/config.yaml" 2>/dev/null || true

# --- training -------------------------------------------------------------
# Pass run_dir + data path through to Hydra. Adjust override keys to match
# this repo's config schema (e.g. log.output_dir, task.dataset_path).
python examples/train_robomimic.py \
    --config-name "$(basename "${CONFIG%.yaml}")" \
    hydra.run.dir="$RUN_DIR" \
    log.output_dir="$RUN_DIR" \
    task.dataset_path="$DATA_ROOT" \
    2>&1

echo "[train.slurm] done -> $RUN_DIR"
