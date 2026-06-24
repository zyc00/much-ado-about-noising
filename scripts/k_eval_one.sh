#!/bin/bash
# Runs INSIDE krun GPU pod. val + train per-phase loss for ONE model.
# args: $1=ckpt_exp $2=normalizer_dataset_basename $3=loss $4=tag
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
CK=logs/$1/models/model_latest.pt; DS=data/$2; L=$3; T=$4
python scripts/eval_val_loss.py   --ckpt $CK --dataset $DS --loss $L --demos data/warmstart_demos.hdf5 --n 50 --stride 2 --tag "$T" 2>&1 | grep -aE "VALLOSS"  | sed 's/^/V20K /'
python scripts/eval_train_loss.py --ckpt $CK --dataset $DS --loss $L --demos data/tool_hang_full2ins_20k_c.hdf5 --n 50 --stride 2 --tag "$T" 2>&1 | grep -aE "TRAINLOSS" | sed 's/^/T20K /'
echo "EVAL_ONE DONE $T"
