#!/bin/bash
# HT-in-Diffusion-Policy runner.  Usage: run_dp_ht.sh <seed> [extra hydra overrides...]
# Uses DP's official train.py / workspace / backbone / EMA / env-runner unchanged;
# only the policy target (hetero-t) differs.
set -eu
SEED=${1:-42}
CFG=${2:-train_hetero_t_transformer_lowdim_workspace}
shift 2 || shift || true

export DEBIAN_FRONTEND=noninteractive MUJOCO_GL=osmesa
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libglew-dev >/dev/null 2>&1 || true

V=/mnt/pfs/yuchen/dp_venv/bin/python
SRC=/mnt/pfs/yuchen/code/much-ado-about-noising/analysis/dp_ht
DP=/mnt/pfs/yuchen/dp

cp "$SRC"/hetero_t_*_lowdim_policy.py "$DP/diffusion_policy/policy/"
cp "$SRC"/train_hetero_t_*_lowdim_workspace.yaml "$DP/diffusion_policy/config/"
cp "$SRC/dp_ht_train.py" "$DP/"

cd "$DP"
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
[ -n "${HT_MINMAX_ACTION:-}" ] && export HT_MINMAX_ACTION
echo "DPHT start seed=$SEED extra=$*"
"$V" dp_ht_train.py \
  --config-name="$CFG" \
  training.seed="$SEED" \
  hydra.run.dir="$DP/runs/${RUNTAG:-ht}_s${SEED}" \
  "$@"
echo "DPHT done seed=$SEED"
