#!/bin/bash
# HT c=2 GR1 rollouts with videos + per-call predicted sigma, 5 tasks, 1 env, seed 1234. Server restarted per task so the sigma file is per task.
cd /mnt/pfs/yuchen/groot/Isaac-GR00T || exit 3
export WANDB_MODE=disabled HF_HOME=/mnt/pfs/yuchen/hf_home MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export HF_TOKEN=$(cat /mnt/pfs/yuchen/hf_home/token)
CKPT=/mnt/pfs/yuchen/groot/ft_gr1c2/checkpoint-60000
OUT=/mnt/pfs/yuchen/groot/sigma_videos; mkdir -p $OUT
TASKS="${TASKS:-PosttrainPnPNovelFromCuttingboardToPanSplitA PosttrainPnPNovelFromTrayToPotSplitA PosttrainPnPNovelFromPlacematToBasketSplitA PnPBottleToCabinetClose PnPCanToDrawerClose}"
NEP=${NEP:-4}; PORT=5799
for T in $TASKS; do
  D=$OUT/$T; rm -rf $D; mkdir -p $D/videos
  GROOT_SIGMA_DUMP=$D/sigma.jsonl GROOT_SIGMA_STEPS=${SIG_STEPS:-8} GROOT_SIGMA_DIMS=${SIG_DIMS:-29} \
    .venv/bin/python gr00t/eval/run_gr00t_server.py --model-path $CKPT --embodiment-tag ROBOCASA_GR1_TABLETOP --use-sim-policy-wrapper --port $PORT > $D/server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 60); do sleep 10; grep -q -i "listening\|ready\|serving\|Server started\|Waiting for" $D/server.log 2>/dev/null && break; done; sleep 30
  GR00T_TRAJ_DUMP=$D GR00T_TRAJ_TAG=traj \
    gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py --n-episodes $NEP --policy-client-host 127.0.0.1 --policy-client-port $PORT \
    --max-episode-steps 720 --env-name gr1_unified/${T}_GR1ArmsAndWaistFourierHands_Env --n-action-steps 8 --n-envs 1 --seed ${SEED:-1234} --video-dir $D/videos > $D/client.log 2>&1
  echo "TASK $T client exit $? ; videos: $(ls $D/videos | tr '\n' ' ') ; sigma rows: $(wc -l < $D/sigma.jsonl 2>/dev/null)"
  kill $SPID 2>/dev/null; sleep 15; pkill -f "run_gr00t_server.py --model-path $CKPT" 2>/dev/null; sleep 5
done
echo SIGVID_DONE
