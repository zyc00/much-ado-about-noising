#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
bd(){ MUJOCO_GL=egl $PY scripts/eval_grasp_breakdown.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss regression --tag "$3" --n 50 2>&1 | grep -aE "GRASPBD"; }
sr(){ MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss regression --demos data/warmstart_demos.hdf5 --warm_to 0 --success grasp --init_mode reset_settle --settle 10 --n 100 2>&1 | grep -aE "WARMSTART" | sed "s/^/GRASPSR $3 /"; }
# breakdown
bd grasp_reg_ext40_2000  tool_hang_init2graspP40_2000.hdf5 "ext40(c1+40)"
# grasp-from-init SR: baseline vs ext40
sr grasp_regression_2000 tool_hang_init2grasp_2000.hdf5    "ext20_baseline"
sr grasp_reg_ext40_2000  tool_hang_init2graspP40_2000.hdf5 "ext40"
echo "EXT LOCAL EVAL DONE"
