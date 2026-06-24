#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
python scripts/collect_scriptB_full.py --n_demos "$1" --start_seed "$2" --output "data/$3"
echo "COLLECTB_DONE $3"
