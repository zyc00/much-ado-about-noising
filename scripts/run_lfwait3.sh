#!/bin/bash
# args: <tag> <loss> [step_target]
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
TAG=$1; LOSS=$2; STEP=${3:-299999}
DS=data/tool_hang_full2ins_2000.hdf5
until python -c "import json;x=[json.loads(q) for q in open('logs/$TAG/metrics.jsonl')];assert x[-1]['step']>=$STEP" 2>/dev/null; do sleep 300; done
sleep 180
MUJOCO_GL=egl python scripts/eval_twofactor.py --ckpt logs/$TAG/models/model_latest.pt --loss $LOSS --tag $TAG-final --dataset $DS
CKPT=logs/$TAG/models/model_latest.pt LOSS=$LOSS TAG=$TAG MUJOCO_GL=egl python -u scripts/probe_ampbins.py
CKPT=logs/$TAG/models/model_latest.pt LOSS=mip_step1 TAG=$TAG-step1 MUJOCO_GL=egl python -u scripts/probe_ampbins.py
