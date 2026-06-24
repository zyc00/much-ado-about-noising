#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
python scripts/eval_twostage_faithful.py --grasp_ckpt logs/$1/models/model_latest.pt --grasp_ds data/$2 \
  --back_ckpt logs/$3/models/model_latest.pt --back_ds data/$4 --loss $5 \
  --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed "s/^/SR_stitch $6 /"
echo "SR_STITCH DONE $6"
