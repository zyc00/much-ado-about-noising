#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
g2g(){ MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/$1/models/model_latest.pt --grasp_ds data/$2 \
  --back_ckpt logs/$1/models/model_latest.pt --back_ds data/$2 --loss $3 \
  --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/GEN2GEN $4 /"; }
g2g full_regression_2000 tool_hang_full2ins_2000.hdf5 regression "MSE_gen->gen (expect ~80 = e2e)"
g2g full_mip_2000        tool_hang_full2ins_2000.hdf5 mip        "MIP_gen->gen (expect ~95 = e2e)"
echo "GEN2GEN DONE"
