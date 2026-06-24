#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
python scripts/eval_warmstart.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss $3 \
  --demos data/warmstart_demos.hdf5 --warm_to 0 --init_mode reset_settle --settle 10 --success assembled --n 100 2>&1 \
  | grep -aE "WARMSTART" | sed "s/^/SR_full $4 /"
echo "SR_FULL DONE $4"
