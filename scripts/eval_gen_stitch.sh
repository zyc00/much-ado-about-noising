#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
# stage1 = 80% MSE generalist (grasp part), stage2 = MSE insert specialist
MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/full_regression_2000/models/model_latest.pt --grasp_ds data/tool_hang_full2ins_2000.hdf5 \
  --back_ckpt logs/pick2ins_regression_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
  --loss regression --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/GENSTITCH MSEgen+MSEins /"
echo "GEN STITCH DONE"
