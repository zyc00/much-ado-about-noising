#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
# (1) IDENTITY: can grasp_mip_2000 complete the FULL task? (generalist can insert; specialist cannot)
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/grasp_mip_2000/models/model_latest.pt \
  --dataset data/tool_hang_init2grasp_2000.hdf5 --loss mip --demos data/warmstart_demos.hdf5 \
  --warm_to 0 --init_mode reset_settle --settle 10 --success assembled --n 50 2>&1 | grep -aE "WARMSTART" | sed "s/^/IDENTITY grasp_mip_FULLtask /"
# (1b) and grasp SR for reference
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/grasp_mip_2000/models/model_latest.pt \
  --dataset data/tool_hang_init2grasp_2000.hdf5 --loss mip --demos data/warmstart_demos.hdf5 \
  --warm_to 0 --init_mode reset_settle --settle 10 --success grasp --n 50 2>&1 | grep -aE "WARMSTART" | sed "s/^/IDENTITY grasp_mip_GRASP /"
# (2) MIP SPECIALIST two-stage: grasp_mip + pick2ins_mip
MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/grasp_mip_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
  --back_ckpt logs/pick2ins_mip_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
  --loss mip --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/MIPSPEC_STITCH grasp_mip+pick2ins_mip /"
echo "VERIFY DONE"
