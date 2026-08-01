#!/bin/bash
# Eval-only N=240 promotions of existing wave-2 winners.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
run() { local tag=$1 ck=$2 loss=$3
  for RANGE in "21000 21120 a" "21160 21280 b"; do
    set -- $RANGE
    python -u scripts/eval_twofactor.py --ckpt $ck --loss $loss --AS 8 \
      --dataset $D --tag pr_${tag}_$3 --seed_lo $1 --seed_hi $2 2>&1 \
      | grep -a "TWOFACTOR.*SR=" | sed "s/^/PROMO $tag blk$3 /"
  done
}
run b_lr05 logs/mp200b_lr05/models/model_step60000.pt regression_hetero_t_jspec
run r_sm02 logs/mp200r_sm02/models/model_step60000.pt regression_hetero_t_rankmid
run r_t075 logs/mp200r_t075/models/model_step60000.pt regression_hetero_t_rankmid
run c_mid_t8 logs/mp200c_mid_l3e-3_t8/models/model_step60000.pt regression_hetero_t_jspec
run rg_cv2 logs/mp200rg_cv2/models/model_step60000.pt regression_hetero_gauss_jspec
echo "PROMO done"
