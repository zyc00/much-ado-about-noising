#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python; D=data/warmstart_demos.hdf5
GD=data/tool_hang_init2grasp_2000.hdf5; BD=data/tool_hang_pick2ins_2000.hdf5
GC=logs/grasp_regression_2000/models/model_latest.pt; BC=logs/pick2ins_regression_2000/models/model_latest.pt
diag () {  # $1=stage1 $2=tag
  MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py --grasp_ckpt $GC --grasp_ds $GD \
    --back_ckpt $BC --back_ds $BD --loss regression --demos $D --stage1 $1 --n 100 2>&1 \
    | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/$2 /"
}
diag replay_c1    "[A: expert->c1 + pick2ins]"
diag replay_grasp "[B: expert->grasped(0.86) + pick2ins]"
diag policy       "[C: grasp-specialist + pick2ins]"
echo "STITCH DIAG DONE"
