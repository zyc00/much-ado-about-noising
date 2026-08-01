#!/bin/bash
# MountainCarContinuous SAC arm: ARCH LOSS [TNOISE], 4 seeds sequential.
set -e
cd "$(dirname "$0")/.."
ARCH=$1; LOSS=$2; TN=${3:-0}
for SEED in 0 1 2 3; do
  python -u q_learning/sac_env.py "$ARCH" "$LOSS" "$SEED" --env mcc \
    --tnoise "$TN" --out q_learning/results_sac_env.jsonl
done
echo "QMCC_ARM_DONE $ARCH $LOSS $TN"
