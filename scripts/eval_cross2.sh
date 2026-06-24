#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
cross(){ MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/$1/models/model_latest.pt --grasp_ds data/$2 --grasp_loss $3 \
  --back_ckpt logs/$4/models/model_latest.pt --back_ds data/$5 --back_loss $6 \
  --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/CROSS $7 /"; }
# MIP grasp + MSE insert
cross grasp_mip_2000 tool_hang_init2grasp_2000.hdf5 mip pick2ins_regression_2000 tool_hang_pick2ins_2000.hdf5 regression "MIPgrasp+MSEinsert"
# MSE grasp + MIP insert
cross grasp_regression_2000 tool_hang_init2grasp_2000.hdf5 regression pick2ins_mip_2000 tool_hang_pick2ins_2000.hdf5 mip "MSEgrasp+MIPinsert"
echo "CROSS2 DONE"
