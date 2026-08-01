#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export MODELS="MSE:logs/mse_traj_2k/models/model_latest.pt:regression,MIP:logs/mip_traj_2k/models/model_latest.pt:mip,PDS:logs/pds_k0.015_h16/models/model_latest.pt:regression"
export DS=data/tool_hang_full2ins_2000.hdf5 DEMOS=12 PROBES=10
python -u scripts/eval_edge_jacobian.py
