#!/usr/bin/env bash
# Task-specific adapter only. This script does not schedule a job by itself.
# Usage: GROOT_ROOT=... bash train.sh tool_hang DATASET OUTPUT flow [extra CLI args]
set -euo pipefail
task=$1; dataset=$2; output=$3; objective=$4
shift 4
dataset=$(realpath "$dataset")
output=$(realpath -m "$output")
case "$task" in tool_hang|transport_ph) ;; *) echo "Unknown task: $task" >&2; exit 2;; esac
case "$objective" in flow|mse|hetero_t) ;; *) echo "Unknown loss: $objective" >&2; exit 2;; esac
adapter_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
groot_dir=${GROOT_ROOT:-/mnt/pfs/yuchen/groot/Isaac-GR00T}
test ! -e "$output" || { echo "Refusing existing output: $output" >&2; exit 2; }
test -f "$dataset/meta/provenance.json"
cd "$groot_dir"
export PYTHONPATH="$groot_dir:$adapter_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/mnt/pfs/yuchen/hf_home}
export WANDB_MODE=disabled GROOT_ROPE_CACHE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
gpus=${NUM_GPUS:-1}
base_model=${BASE_MODEL:-/mnt/pfs/yuchen/groot/model}
loss_args=()
if [[ "$objective" == hetero_t ]]; then
  default_nu=320
  [[ "$task" != transport_ph ]] || default_nu=640
  loss_args=(--ht-mvt --ht-df "${HT_NU:-$default_nu}")
fi
"$groot_dir/.venv/bin/python" - "$dataset" "$task" <<'PY'
import json, sys
from pathlib import Path
p=json.loads((Path(sys.argv[1])/'meta/provenance.json').read_text())
assert p['task']==sys.argv[2]
assert p['source_action_mode']=='absolute'
PY
"$groot_dir/.venv/bin/torchrun" --nproc_per_node="$gpus" --master_port="${MASTER_PORT:-29500}" \
  "$adapter_dir/launch.py" \
  --base-model-path "$base_model" --dataset-path "$dataset" \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path "$adapter_dir/${task}_config.py" \
  --output-dir "$output" --num-gpus "$gpus" --no-use-wandb \
  --loss-type "$objective" --no-use-percentiles --state-dropout-prob 0.0 \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-64}" --max-steps "${MAX_STEPS:-2000}" \
  --learning-rate 1e-4 --warmup-ratio 0.05 --weight-decay 1e-5 \
  --save-steps "${SAVE_STEPS:-250}" --save-total-limit 8 \
  --dataloader-num-workers 4 --shard-size 256 --episode-sampling-rate 1.0 \
  "${loss_args[@]}" "$@"
