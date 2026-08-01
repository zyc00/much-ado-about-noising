#!/bin/bash
# Score completed hetero+condreg tuning arms. Usage: k_score_tune.sh <arm...>
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
for NAME in "$@"; do
  case $NAME in
    *htc*) LOSS=regression_hetero_t_cnd ;;
    *) LOSS=regression_hetero_gauss_cnd ;;
  esac
  CK=logs/mp200_${NAME}_s1000/models/snap_300000.pt
  [ -f logs/mp200_${NAME}_s1000/models/model_latest.pt ] && cp -n logs/mp200_${NAME}_s1000/models/model_latest.pt $CK
  python -u scripts/eval_twofactor.py --ckpt $CK --loss $LOSS --AS 8 \
    --dataset $D --tag tn_${NAME} --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/TN $NAME /"
done
echo "TUNE-SCORE done"
