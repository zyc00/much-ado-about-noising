#!/bin/bash
# Iteration chain: train (full schedule, stop 60k) + N=240 eval (two blocks).
# Usage: k_iter.sh <name> <loss> [ENV=V ...]
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=${DSET:-data/tool_hang_full2ins_mp_200.hdf5}
NAME=$1; LOSS=$2; shift 2
for kv in "$@"; do export "$kv"; done
export SNAP_AT=${SNAP_AT:-300000} STOP_AT=${STOP_AT:-300000}
echo "ITER train start $NAME $LOSS extra: $*"
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/$D network=chiunet ${OVR:-} \
  optimization.loss_type=$LOSS optimization.seed=${TSEED:-1000} optimization.auto_resume=false \
  log.log_dir=logs/$NAME 2>&1 | grep -aE "Error|Traceback" -A6 | head -20
echo "ITER train done $NAME"
for ST in $(echo $SNAP_AT | tr "," " "); do
  CK=logs/${NAME}/models/model_step${ST}.pt
  [ -f "$CK" ] || continue
  for RANGE in "21000 21120 a" "21160 21280 b"; do
    set -- $RANGE
    EV_OVERRIDES="${OVR:-}" python -u scripts/eval_twofactor.py --ckpt $CK --loss $LOSS --AS 8 \
      --dataset $D --tag it_${NAME}_${ST}_$3 --seed_lo $1 --seed_hi $2 2>&1 \
      | grep -a "TWOFACTOR.*SR=" | sed "s/^/ITER ${NAME}@${ST} blk$3 /"
  done
done
echo "ITER all done $NAME"
