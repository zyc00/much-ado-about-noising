#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
MUJOCO_GL=egl $PY scripts/eval_grasp_pose_dist.py --ckpt logs/grasp_regression_2000/models/model_latest.pt --dataset data/tool_hang_init2grasp_2000.hdf5 --loss regression --demos data/warmstart_demos.hdf5 --n 50 --out_npz /tmp/claude-1001/-home-jigu-projects-much-ado-about-noising/469964b2-9364-47f5-8a44-7dda4fc11f85/scratchpad/grasp_spec_mse.npz 2>&1 | grep -aE "GRASPDIST|POS bias|saved"
MUJOCO_GL=egl $PY scripts/eval_grasp_pose_dist.py --ckpt logs/grasp_mip_2000/models/model_latest.pt --dataset data/tool_hang_init2grasp_2000.hdf5 --loss mip --demos data/warmstart_demos.hdf5 --n 50 --out_npz /tmp/claude-1001/-home-jigu-projects-much-ado-about-noising/469964b2-9364-47f5-8a44-7dda4fc11f85/scratchpad/grasp_spec_mip.npz 2>&1 | grep -aE "GRASPDIST|POS bias|saved"
echo "SPEC DONE"
