#!/bin/bash
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
DSP=data/tool_hang_human_lowdim_up.hdf5 \
CKS="logs/hheterot_s5/models/model_latest.pt,logs/snap_hheterot_1k/models/model_latest.pt,logs/snap_hheterot_2k/models/model_latest.pt" \
NEP=50 python -u scripts/eval_ensemble.py
