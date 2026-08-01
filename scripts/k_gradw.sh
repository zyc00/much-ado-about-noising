#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
export MODELS="MSE:logs/mse_traj_2k/models/model_latest.pt:regression,MIP:logs/mip_traj_2k/models/model_latest.pt:mip"
python -u scripts/eval_grad_weight.py
