#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
CK=logs/mip_traj_2k/models/model_latest.pt
export MODELS="MIP1:$CK:mip_step1,MIP2:$CK:mip,MSE:logs/mse_traj_2k/models/model_latest.pt:regression"
export DS=data/tool_hang_full2ins_2000.hdf5 DEMOS=10 PROBES=8
NORMDS=$DS python -u scripts/eval_kalign.py --models "$MODELS"
python -u scripts/eval_edge_jacobian.py
MUJOCO_GL=egl python scripts/eval_twofactor.py --ckpt "$CK" --loss mip_step1 --tag "mip_step1-final" --dataset data/tool_hang_full2ins_2000.hdf5
echo STEP1CMP-DONE
