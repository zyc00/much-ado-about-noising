#!/bin/bash
# Chunk-length campaign arm: LOSS CHUNK SEED. Aligned human protocol
# (k_align_human.sh), task.horizon=CHUNK, act_steps=CHUNK. All 20k-grid
# snapshots kept unconditionally.
set -eu
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
L=$1; C=$2; S=$3
case $L in regression) A=l2;; regression_hetero_gauss) A=hg;; regression_hetero_t) A=ht;; esac
TAG=chunk_${A}_c${C}_s${S}
LOSS=$L SEED=$S EXTRA="task.horizon=$C task.act_steps=$C" \
  bash scripts/k_align_human.sh $TAG
echo "CHUNK_TRAIN_DONE $TAG"
