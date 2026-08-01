#!/bin/bash
# Table-12/13 HT arm: TASK NET SEED [LOSS] [DPATH]. In-train eval protocol
# (no snapshot re-eval): 300k steps, eval every 20k x 50 eps.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HOME=/mnt/pfs/yuchen/.hf
TASK=$1; NET=$2; SEED=$3; LOSS=${4:-regression_hetero_t}; DPATH=${5:-}; AR=${6:-false}; STAGE=${7:-false}; OVR=${8:-}; OVR2=${9:-}
if [ "$STAGE" = "true" ] && [ -n "$DPATH" ]; then
  mkdir -p /tmp/stage
  LDS=/tmp/stage/$(basename "$DPATH")
  if [ ! -f "$LDS" ]; then
    echo "T12 staging $DPATH -> $LDS"
    cp "$DPATH" "$LDS.part" && mv "$LDS.part" "$LDS"
  fi
  if [ -f "$LDS" ]; then DPATH="$LDS"; fi
fi
EXTRA=""
[ -n "$DPATH" ] && EXTRA="+task.dataset_path=$DPATH"
if [ -n "$DPATH" ] && [ ! -f "$DPATH" ]; then
  echo "T12 MISSING_DATASET $DPATH"
  exit 4
fi
SFX=""; [ -n "$OVR" ] && SFX="_$(echo $OVR $OVR2 | tr -dc a-z0-9)"
NAME=t12_${TASK}_${NET}_s${SEED}${SFX}
echo "T12 start $NAME loss=$LOSS extra=$EXTRA"
TRAINER=examples/train_robomimic.py
case $TASK in
  pusht*) TRAINER=examples/train_pusht.py;;
  kitchen*) TRAINER=examples/train_kitchen.py;;
esac
python -u $TRAINER task=$TASK network=$NET $EXTRA $OVR $OVR2 \
  optimization.loss_type=$LOSS optimization.seed=$SEED \
  optimization.auto_resume=$AR log.log_dir=logs/$NAME \
  log.wandb_mode=disabled 2>&1 | grep -aE "mean_success|Error|Traceback|Downloading dataset|Using local|Downloaded dataset" -A3 \
  | sed "s/^/T12EV $NAME /"
echo "T12 done $NAME"
