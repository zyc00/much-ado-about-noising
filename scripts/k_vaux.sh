#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
export DS=data/tool_hang_full2ins_wpmatch_2000.hdf5
SNAPS=$(ls logs/wpm_lam27/models/snap_*.pt 2>/dev/null | sort | awk 'NR%2==1' | paste -sd,)
export CKPTS="$SNAPS,logs/wpm_lam27/models/model_latest.pt"
python -u scripts/eval_vaux.py
echo "=== lam3 series ==="
SNAPS=$(ls logs/wpm_lam3/models/snap_*.pt 2>/dev/null | sort | awk 'NR%2==1' | paste -sd,)
export CKPTS="$SNAPS,logs/wpm_lam3/models/model_latest.pt"
python -u scripts/eval_vaux.py
