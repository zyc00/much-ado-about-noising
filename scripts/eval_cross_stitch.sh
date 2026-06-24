#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
GR=logs/grasp_regression_2000/models/model_latest.pt;  GM=logs/grasp_mip_2000/models/model_latest.pt
BR=logs/pick2ins_regression_2000/models/model_latest.pt; BM=logs/pick2ins_mip_2000/models/model_latest.pt
GDS=data/tool_hang_init2grasp_2000.hdf5; BDS=data/tool_hang_pick2ins_2000.hdf5
cross () {  # $1 grasp_ckpt $2 grasp_loss $3 back_ckpt $4 back_loss $5 tag
  MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
    --grasp_ckpt $1 --grasp_ds $GDS --grasp_loss $2 \
    --back_ckpt $3 --back_ds $BDS --back_loss $4 \
    --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
    | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/$5 /"
}
cross $GR regression $BM mip "[MSEgrasp+MIPinsert]"
cross $GM mip        $BR regression "[MIPgrasp+MSEinsert]"
echo "CROSS DONE"
