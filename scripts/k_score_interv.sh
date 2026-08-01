#!/bin/bash
# Score the six completed intervention arms with the canonical twofactor
# protocol (AS=8, 60 seeds, snap_300000). Prints "IV <cell> TWOFACTOR" lines.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5

run() { # name loss extra_env...
  local name=$1 loss=$2; shift 2
  echo "IV start $name"
  env "$@" python -u scripts/eval_twofactor.py \
    --ckpt logs/${name}/models/snap_300000.pt --loss $loss --AS 8 \
    --dataset $D --tag iv_${name} --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/IV /"
}

run mp200_cnd2a_s1000  regression_condreg
run mp200_cnd2b_s1000  regression_condreg
run mp200_jac1e4_s1000 regression_jacreg
run mp200_jac1e5_s1000 regression_jacreg
run mp200_b4096_s1000  regression
run mp200_wide_s1000   regression EV_OVERRIDES="network.model_dim=256 network.emb_dim=512"
echo "IV done"
