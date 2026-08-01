#!/bin/bash
# args: <tag/logdir> <loss> <dataset> <final_step>
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
TAG=$1; LOSS=$2; DS=$3; STEP=${4:-299999}
until python -c "import json;x=[json.loads(q) for q in open('logs/$TAG/metrics.jsonl')];assert x[-1]['step']>=$STEP" 2>/dev/null; do sleep 300; done
sleep 180
NORMDS="$DS" MUJOCO_GL=egl python -u scripts/eval_kalign.py --models "$TAG:logs/$TAG/models/model_latest.pt:$LOSS"
MUJOCO_GL=egl python scripts/eval_twofactor.py --ckpt "logs/$TAG/models/model_latest.pt" --loss "$LOSS" --tag "$TAG-final" --dataset "$DS"
