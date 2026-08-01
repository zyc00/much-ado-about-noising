#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
CK=logs/mp200r_l3e2t05/models/model_step60000.pt
RK_ARMS="RM54:regression_hetero_t_rankmid:$CK,HTBASE:regression_hetero_t:logs/mp200_ht_s1000/models/snap_60000.pt" \
  python -u scripts/probe_rank.py 2>&1 | grep -aE "^RK"
for RANGE in "21000 21120 a" "21160 21280 b"; do
  set -- $RANGE
  python -u scripts/eval_twofactor.py --ckpt $CK --loss regression_hetero_t_rankmid \
    --AS 8 --dataset $D --tag rmv_$3 --seed_lo $1 --seed_hi $2 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/RMV block_$3 /"
done
echo "RMFINAL done"
