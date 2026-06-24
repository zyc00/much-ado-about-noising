#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python; D=data/warmstart_demos.hdf5
# 1. grasp specialist, reset+settle init, grasp SR from init
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/grasp_regression_2000/models/model_latest.pt \
  --dataset data/tool_hang_init2grasp_2000.hdf5 --loss regression --demos $D \
  --warm_to 0 --init_mode reset_settle --settle 10 --success grasp --n 100 2>&1 | grep -aE "WARMSTART" | sed 's/^/GRASP_SPEC(reset+settle) /'
# 2. insertion specialist pick2ins from clean c1 (set_state; settle N/A since starts at c1) - reconfirm
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/pick2ins_regression_2000/models/model_latest.pt \
  --dataset data/tool_hang_pick2ins_2000.hdf5 --loss regression --demos $D \
  --warm_to c1 --init_mode state0 --success assembled --n 100 2>&1 | grep -aE "WARMSTART" | sed 's/^/INS_SPEC(from c1) /'
# 3. MSE stitching MISMATCH (grasp -> pick2ins), reset+settle init
MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/grasp_regression_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
  --back_ckpt logs/pick2ins_regression_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
  --loss regression --demos $D --init_mode reset_settle --n 100 2>&1 | grep -aE "TWOSTAGE_FAITHFUL" | sed 's/^/STITCH_MISMATCH(reset+settle) /'
# 4. MSE stitching MATCHED (grasp -> handoffins), reset+settle init
MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/grasp_regression_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
  --back_ckpt logs/handoffins_regression_2000/models/model_latest.pt --back_ds data/tool_hang_handoffins_2000.hdf5 \
  --loss regression --demos $D --init_mode reset_settle --n 100 2>&1 | grep -aE "TWOSTAGE_FAITHFUL" | sed 's/^/STITCH_MATCHED(reset+settle) /'
echo "RESETTLE RECHECK DONE"
