#!/bin/bash
# Table-12/13 HT arm: TASK NET SEED [LOSS] [DPATH]. In-train eval protocol
# (no snapshot re-eval): 300k steps, eval every 20k x 50 eps.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HOME=/mnt/pfs/yuchen/.hf
case $1 in kitchen*) export PYTHONPATH=/mnt/pfs/yuchen/mj316_slim:${PYTHONPATH:-};; esac
case $1 in *image*) export MIP_NUM_WORKERS=${MIP_NUM_WORKERS:-12} MIP_FAST_IMG=${MIP_FAST_IMG:-1};; esac
TASK=$1; NET=$2; SEED=$3; LOSS=${4:-regression_hetero_t}; DPATH=${5:-}; AR=${6:-false}; STAGE=${7:-false}; OVR=${8:-}; OVR2=${9:-}
if [ "$STAGE" = "true" ] && [ -n "$DPATH" ]; then
  # per-pod RAM staging when it fits (fastest, no cross-pod sharing),
  # else node-local disk; unique name per dataset either way
  SZ=$(stat -c%s "$DPATH" 2>/dev/null || echo 0)
  SHM_FREE=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -1)
  UNIQ=$(echo "$DPATH" | md5sum | cut -c1-10)_$(basename "$DPATH")
  if [ "${SHM_FREE:-0}" -gt $((SZ + 8589934592)) ]; then
    mkdir -p /dev/shm/stage; LDS=/dev/shm/stage/$UNIQ
  else
    mkdir -p /tmp/stage; LDS=/tmp/stage/$UNIQ
  fi
  if [ ! -f "$LDS" ]; then
    echo "T12 staging $DPATH -> $LDS"
    cp "$DPATH" "$LDS.part" && mv "$LDS.part" "$LDS"
  fi
  if [ -f "$LDS" ]; then DPATH="$LDS"; fi
fi
EXTRA=""
[ -n "$DPATH" ] && EXTRA="+task.dataset_path=$DPATH"
if [ -n "$DPATH" ] && [ ! -e "$DPATH" ]; then
  echo "T12 MISSING_DATASET $DPATH"
  exit 4
fi
SFX=""; [ -n "$OVR" ] && SFX="_$(echo $OVR $OVR2 | tr -dc a-z0-9)"
[ -n "${HT_SMIN:-}" ] && { SFX="${SFX}_smin$(echo $HT_SMIN | tr -dc 0-9)"; export HT_SMIN; }
# NOTE: map "-" to "m" BEFORE stripping, or -1 and +1 collide into one run dir.
[ -n "${HT_SBIAS:-}" ] && { SFX="${SFX}_sb$(echo $HT_SBIAS | sed 's/-/m/g' | tr -dc 0-9m)"; export HT_SBIAS; }
[ -n "${WELSCH_H:-}" ] && { SFX="${SFX}_wh$(echo $WELSCH_H | tr -dc 0-9)"; export WELSCH_H; }
[ -n "${WELSCH_BETA:-}" ] && { SFX="${SFX}_wb$(echo $WELSCH_BETA | tr -dc 0-9)"; export WELSCH_BETA; }
[ -n "${BARRON_A:-}" ] && { SFX="${SFX}_ba$(echo $BARRON_A | tr -dc 0-9m)"; export BARRON_A; }
[ -n "${BARRON_C:-}" ] && { SFX="${SFX}_bc$(echo $BARRON_C | tr -dc 0-9)"; export BARRON_C; }
[ -n "${BARRON_MIX_FRAC:-}" ] && { SFX="${SFX}_bm$(echo $BARRON_MIX_FRAC | tr -dc 0-9)"; export BARRON_MIX_FRAC; }
[ -n "${EVAL_SEED_BASE:-}" ] && { SFX="${SFX}_fx"; export EVAL_SEED_BASE; }
[ -n "${WD_ENCODER:-}" ] && { SFX="${SFX}_wde"; export WD_ENCODER; }
[ -n "${WD_NODECAY:-}" ] && { SFX="${SFX}_wdg"; export WD_NODECAY; }
[ -n "${NU_MODE:-}" ] && { SFX="${SFX}_num${NU_MODE}"; export NU_MODE; }
# ext4 caps a path component at 255 bytes; long override strings blow past it
# (Errno 36). Keep the head readable and hash the tail so names stay unique.
if [ ${#SFX} -gt 110 ]; then
  SFX="${SFX:0:102}$(printf '%s' "$SFX" | md5sum | cut -c1-6)"
fi
case $LOSS in
  regression_hetero_t) LTAG=ht;;
  regression_hetero_t_learnnu) LTAG=htlnu;;
  regression_hetero_t_learnnu_cond) LTAG=htlnuc;;
  regression_hetero_t_learnnu_cond2) LTAG=htlnuc2;;
  xm) LTAG=xm;;
  regression_hetero_gauss) LTAG=hg;;
  regression) LTAG=l2;;
  mip) LTAG=mip;;
  flow) LTAG=flow;;
  straight_flow) LTAG=sflow;;
  *) LTAG=$(echo $LOSS | tr -dc a-z);;
esac
# default (bare HT, no override) keeps the legacy name for harvest continuity
if [ "$LOSS" = "regression_hetero_t" ] && [ -z "$OVR" ] && [ -z "$OVR2" ] && [ -z "${HT_SMIN:-}" ]; then
  NAME=t12_${TASK}_${NET}_s${SEED}
else
  NAME=t12_${TASK}_${NET}_s${SEED}_${LTAG}${SFX}
fi
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
