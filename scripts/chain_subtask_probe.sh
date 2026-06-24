#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python; D=data/warmstart_demos.hdf5
probe () {  # $1=ckpt $2=loss $3=tag
  # grasp from init (reset+settle)
  MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt $1 --dataset data/tool_hang_full2ins_2000.hdf5 --loss $2 \
    --demos $D --warm_to 0 --init_mode reset_settle --settle 10 --success grasp --n 100 2>&1 | grep -aE "WARMSTART" | sed "s/^/$3 GRASP /"
  # insert from clean align-done (set_state replay to align_done, then policy inserts)
  MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt $1 --dataset data/tool_hang_full2ins_2000.hdf5 --loss $2 \
    --demos $D --warm_to align_done --init_mode state0 --success assembled --n 100 2>&1 | grep -aE "WARMSTART" | sed "s/^/$3 INSERT /"
}
probe logs/full_regression_2000/models/model_latest.pt regression "MSE(80%)"
probe logs/full_mip_2000/models/model_latest.pt        mip        "MIP(95%)"
echo "SUBTASK PROBE DONE"
