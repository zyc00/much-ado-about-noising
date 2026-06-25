#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
cross(){ MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/$1/models/model_latest.pt --grasp_ds data/$2 --grasp_loss $3 \
  --back_ckpt logs/$4/models/model_latest.pt --back_ds data/$5 --back_loss $6 \
  --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/CROSSGEN $7 /"; }
# generalist-MSE grasp + MIP insert
cross full_regression_2000 tool_hang_full2ins_2000.hdf5 regression pick2ins_mip_2000        tool_hang_pick2ins_2000.hdf5 mip        "genMSE+insMIP"
# generalist-MIP grasp + MSE insert
cross full_mip_2000        tool_hang_full2ins_2000.hdf5 mip        pick2ins_regression_2000 tool_hang_pick2ins_2000.hdf5 regression "genMIP+insMSE"
echo "CROSSGEN DONE"
