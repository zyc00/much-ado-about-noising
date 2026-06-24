#!/bin/bash
# Generic warmstart SR. args: $1=ckpt_exp $2=dataset $3=loss $4=warm_to $5=success $6=init_mode $7=tag
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
python scripts/eval_warmstart.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss $3 \
  --demos data/warmstart_demos.hdf5 --warm_to $4 --success $5 --init_mode $6 --settle 10 --n 100 2>&1 \
  | grep -aE "WARMSTART" | sed "s/^/SRsub $7 /"
echo "SRSUB DONE $7"
