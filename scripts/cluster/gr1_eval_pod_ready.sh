#!/usr/bin/env bash
# Robust GR00T/RoboCasa-GR1 evaluation shard.
# Usage: gr1_eval_pod_ready.sh CKPT TAG SHARD "ENV1 ENV2 ENV3"

set -euo pipefail

checkpoint="$1"
tag="$2"
shard="$3"
envs="$4"
root=/mnt/pfs/yuchen/groot
repo="${root}/Isaac-GR00T"
server_log="${root}/evalsrv_${tag}_${shard}.log"
eval_log="${root}/evalout_${tag}_${shard}.log"

cd "${repo}"
export HF_HOME=/mnt/pfs/yuchen/hf_home
export HF_TOKEN
HF_TOKEN="$(cat /mnt/pfs/yuchen/hf_home/token)"
export http_proxy=http://172.17.0.12:2080
export https_proxy=http://172.17.0.12:2080
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

: > "${server_log}"
: > "${eval_log}"
.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path "${checkpoint}" \
  --embodiment-tag ROBOCASA_GR1_TABLETOP \
  --use-sim-policy-wrapper \
  --port 5555 > "${server_log}" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

# A fixed sleep is unreliable when many workers load from PFS concurrently.
# Wait until the ZMQ server is actually listening, or fail the shard clearly.
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "server exited before becoming ready" | tee -a "${eval_log}"
    exit 3
  fi
  if timeout 1 bash -c '</dev/tcp/127.0.0.1/5555' 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "server readiness timeout" | tee -a "${eval_log}"
  exit 4
fi

sim_python=gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python
for env_name in ${envs}; do
  completed=0
  for attempt in 1 2; do
    echo "EVAL_BEGIN env=${env_name} attempt=${attempt}" >> "${eval_log}"
    if "${sim_python}" gr00t/eval/rollout_policy.py \
      --env-name "${env_name}" \
      --n-episodes 20 \
      --n-envs 5 \
      --max-episode-steps 720 \
      --n-action-steps 8 \
      --policy-client-host localhost \
      --policy-client-port 5555 \
      --model-path "" >> "${eval_log}" 2>&1; then
      completed=1
      break
    fi
    echo "EVAL_RETRY env=${env_name} attempt=${attempt}" >> "${eval_log}"
    sleep 10
  done
  if [[ "${completed}" -ne 1 ]]; then
    echo "EVAL_FAILED env=${env_name}" >> "${eval_log}"
    exit 5
  fi
done

expected_count="$(wc -w <<< "${envs}")"
success_count="$(grep -a -c 'success rate:' "${eval_log}" || true)"
if [[ "${success_count}" -ne "${expected_count}" ]]; then
  echo "expected ${expected_count} task results, found ${success_count}" >> "${eval_log}"
  exit 6
fi
echo "EVAL_${tag}_${shard}_COMPLETE"
