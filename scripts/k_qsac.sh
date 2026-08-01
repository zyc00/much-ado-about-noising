#!/bin/bash
# SAC critic-objective arm: ARCH LOSS TNOISE, 4 seeds sequential.
set -e
cd "$(dirname "$0")/.."
ARCH=$1; LOSS=$2; TN=${3:-0}
for SEED in 0 1 2 3; do
  python -u q_learning/sac_pendulum.py "$ARCH" "$LOSS" "$SEED" \
    --tnoise "$TN" --out q_learning/results_sac.jsonl
done
echo "QSAC_ARM_DONE $ARCH $LOSS $TN"
