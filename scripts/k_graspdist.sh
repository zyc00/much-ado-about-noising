#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
python scripts/eval_grasp_pose_dist.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss $3 \
  --demos data/warmstart_demos.hdf5 --n 50 2>&1 | grep -aE "GRASPDIST|cm\)|POS bias" | sed "s/^/GD_$4 /"
echo "GDDONE $4"
