#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
python scripts/eval_handoff_conditional.py \
  --grasp_ckpt logs/full_regression_20kB/models/model_latest.pt --grasp_ds data/tool_hang_full2ins_20kB.hdf5 \
  --back_ckpt logs/pick2ins_regression_20kB/models/model_latest.pt --back_ds data/tool_hang_pick2ins_20kB.hdf5 \
  --loss regression --demos data/warmstart_demos.hdf5 --n 100 --tag genG_specI_20k 2>&1 | grep -aE "HANDOFF_COND|dev"
echo "COND DONE"
