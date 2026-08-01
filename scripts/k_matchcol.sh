#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export WAYPOINT=1 WP_SPACING=0.12 WP_EPS=0.02 SMOOTH_CAP=1.0
python scripts/collect_scriptB_full.py --n_demos "$1" --start_seed "$2" --output "data/$3"
echo "MATCHCOL_DONE $3"
