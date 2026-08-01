#!/bin/bash
# Vision (image) training: stage the image h5 from the PVC HF cache to node-local
# disk, then run the aligned protocol. args: <tag>; env: TASK, HFFILE, LOSS, NET,
# SEED, EXTRA, GSTEPS
set -eu
export MUJOCO_GL=egl HF_HOME=/mnt/pfs/yuchen/hf_home
TAG=$1; TASK=${TASK:?}; HFFILE=${HFFILE:-}; LOSS=${LOSS:-regression_hetero_t}
NET=${NET:-chiunet}; SEED=${SEED:-1000}; EXTRA=${EXTRA:-}; GSTEPS=${GSTEPS:-300001}
if [ -n "${PVCFILE:-}" ]; then P="$PVCFILE"; else
P=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='ChaoyiPan/mip-dataset', filename='$HFFILE', repo_type='dataset'))")
fi
echo "STAGING $P -> /tmp/img.hdf5"
cp -L "$P" /tmp/img.hdf5
echo "STAGED"
rm -rf logs/$TAG
( LAST=-1
  while true; do
    sleep 120
    step=$(python -c "import json;print(json.loads(open('logs/$TAG/metrics.jsonl').readlines()[-1])['step'])" 2>/dev/null) || continue
    [ -z "$step" ] && continue
    grid=$(( step / 20000 * 20000 ))
    [ "$grid" -lt 20000 ] && continue
    [ "$grid" = "$LAST" ] && continue
    cp logs/$TAG/models/model_latest.pt logs/$TAG/models/snap_${grid}.pt 2>/dev/null || continue
    echo "SNAP saved snap_${grid}.pt"
    LAST=$grid
    [ "$grid" -ge $((GSTEPS-20001)) ] && break
  done ) &
python -u examples/train_robomimic.py task=$TASK +task.dataset_path=/tmp/img.hdf5 network=$NET \
  optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=20000 optimization.gradient_steps=$GSTEPS log.save_freq=20000 \
  optimization.seed=$SEED $EXTRA \
  optimization.loss_type=$LOSS log.exp_name=$TAG log.log_dir=logs/$TAG 2>&1 | grep -aE "mean_success|Step [0-9]*999\]|Traceback|Error" | sed "s/^/[$TAG] /"
cp logs/$TAG/models/model_latest.pt logs/$TAG/models/snap_300000.pt 2>/dev/null || true
python - "$TAG" <<'PYEOF'
import json, sys
tag = sys.argv[1]
evs = []
for line in open(f"logs/{tag}/metrics.jsonl"):
    d = json.loads(line)
    for k in d:
        if k.startswith("mean_success"):
            evs.append((d["step"], d[k]))
if evs:
    best = max(v for _, v in evs)
    last5 = [v for _, v in evs[-5:]]
    print(f"SUMMARY {tag} best={best:.2f} last5={sum(last5)/len(last5):.3f} n_evals={len(evs)}")
PYEOF
echo "TRAIN-DONE $TAG"
