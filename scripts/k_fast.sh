#!/bin/bash
# FAST-ITERATION chain: full-schedule recipe stopped at 60k, snapshots at
# 20k+60k, canonical twofactor on BOTH + recovery probe on 60k.
# Usage: k_fast.sh <name> <loss> [ENV=V ...]
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=${DSET:-data/tool_hang_full2ins_mp_200.hdf5}
NAME=$1; LOSS=$2; shift 2
for kv in "$@"; do export "$kv"; done
export SNAP_AT=${SNAP_AT:-20000,60000} STOP_AT=${STOP_AT:-60000}
SCORE_STEPS=$(echo $SNAP_AT | tr "," " ")
echo "FAST train start $NAME $LOSS extra: $*"
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/$D network=chiunet ${OVR:-} \
  optimization.loss_type=$LOSS optimization.seed=1000 optimization.auto_resume=false \
  log.log_dir=logs/$NAME 2>&1 | grep -aE "Error|Traceback" -A6 | head -20
echo "FAST train done $NAME"
unset RECOVJIT RJ_NOCORR RJ_NORMAL 2>/dev/null || true
for ST in $SCORE_STEPS; do
  CK=logs/${NAME}/models/model_step${ST}.pt
  [ -f "$CK" ] || continue
  EV_OVERRIDES="${OVR:-}" python -u scripts/eval_twofactor.py --ckpt $CK --loss $LOSS --AS 8 \
    --dataset $D --tag f_${NAME}_${ST} --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/FAST $NAME $ST /"
done
RV_ARMS="${NAME}:${LOSS}:logs/${NAME}/models/model_step60000.pt" \
python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD"
echo "FAST all done $NAME"
