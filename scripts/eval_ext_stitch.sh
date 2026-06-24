#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
st(){ MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py --grasp_ckpt logs/$1/models/model_latest.pt --grasp_ds data/$2 \
  --back_ckpt logs/pick2ins_regression_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
  --loss regression --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/STITCH $3 /"; }
st grasp_regression_2000 tool_hang_init2grasp_2000.hdf5    "ext20_baseline"
st grasp_reg_ext40_2000  tool_hang_init2graspP40_2000.hdf5 "ext40"
echo "EXT STITCH DONE"
