#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
# sanity: MIP insertion specialist from clean c1 (expect ~100)
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/pick2ins_mip_2000/models/model_latest.pt \
  --dataset data/tool_hang_pick2ins_2000.hdf5 --loss mip --demos data/warmstart_demos.hdf5 \
  --warm_to c1 --init_mode state0 --success assembled --n 100 2>&1 | grep -aE "WARMSTART" | sed 's/^/MIP_INS_from_c1 /'
# MIP no-dagger stitching: grasp_mip + pick2ins_mip (vs MSE 42%)
MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/grasp_mip_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
  --back_ckpt logs/pick2ins_mip_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
  --loss mip --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed 's/^/MIP_STITCH_nodagger /'
echo "MIP STITCH EVAL DONE"
