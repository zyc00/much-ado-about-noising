#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
CK=logs/full_mip_2000/models/model_latest.pt
DS=data/tool_hang_full2ins_2000.hdf5
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt $CK --dataset $DS --loss mip \
  --demos data/warmstart_demos.hdf5 --warm_to 0 --success grasp --init_mode reset_settle --settle 10 --n 100 2>&1 \
  | grep -aE "WARMSTART" | sed 's/^/MIP_GEN_grasp_from_init /'
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt $CK --dataset $DS --loss mip \
  --demos data/warmstart_demos.hdf5 --warm_to c1 --success assembled --init_mode state0 --n 100 2>&1 \
  | grep -aE "WARMSTART" | sed 's/^/MIP_GEN_c1_to_insertion /'
echo "MIP GEN SUBTASKS DONE"
