#!/bin/bash
# Runs INSIDE krun pod (CPU). Scripted clean-demo collection shard.
# args: $1=n_demos $2=start_seed $3=output_basename(in data/)
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
python scripts/collect_tool_hang_demos.py --n_demos "$1" --start_seed "$2" --output "data/$3"
echo "COLLECT_DONE $3"
