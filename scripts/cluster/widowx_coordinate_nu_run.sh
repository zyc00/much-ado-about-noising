#!/usr/bin/env bash
# Matched repeat of ft_wxouthomog, changing only the scalar-coordinate df.
set -euo pipefail
df=${1:?Pass 8 or 16}
case "$df" in 8|16) ;; *) exit 2 ;; esac
root=/mnt/pfs/yuchen/groot
run_dir="$root/ft_wxouthomog_nu${df}_20260907_r2"
cd "$root/Isaac-GR00T"
export PATH="$PWD/.venv/bin:$PATH"
export HF_HOME=/mnt/pfs/yuchen/hf_home WANDB_MODE=disabled
export GROOT_ROPE_CACHE=1 PYTHONUNBUFFERED=1
# Refuse to overwrite any previous output for this run.
mkdir "$run_dir"
mkdir "$run_dir/audit" "$run_dir/evaluation"
cp gr00t/model/gr00t_n1d7/gr00t_n1d7.py "$run_dir/audit/loss_source.py"
cp gr00t/configs/finetune_config.py "$run_dir/audit/finetune_config.py"
cp examples/finetune.sh "$run_dir/audit/finetune.sh"
cp "$root/coordinate_nu_20260907/run.sh" "$run_dir/audit/run.sh"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$run_dir/audit/git_commit.txt"
fi
sha256sum gr00t/model/gr00t_n1d7/gr00t_n1d7.py > "$run_dir/audit/loss_sha256.txt"
mkdir -p /dev/shm/wx
cp -r "$root/bridge_orig_lerobot/." /dev/shm/wx/
episode_count=$(wc -l < /dev/shm/wx/meta/episodes.jsonl)
test "$episode_count" -eq 53192
echo "STAGED $episode_count episodes; coordinate df=$df; shared sample sigma"
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=1024 SAVE_STEPS=1000 \
  bash examples/finetune.sh \
  --base-model-path "$root/model" --dataset-path /dev/shm/wx \
  --embodiment-tag SIMPLER_ENV_WIDOWX --output-dir "$run_dir" \
  -- --no-use-wandb --loss-type=hetero_t --ht-sbias=-0.4535 --ht-df="$df" \
  --no-ht-mvt --ht-sum-mode=outside --ht-sigma-mode=homog \
  --ht-sbias-vec -0.9204 -0.9642 -0.4865 -0.9896 -0.9103 -0.9974 0.5403 \
  --state-dropout-prob 0.8 --dataloader-num-workers 8 \
  2>&1 | tee "$run_dir/train.log"
python - "$run_dir" "$df" <<'PY'
import json, sys
from pathlib import Path
root, df = Path(sys.argv[1]), float(sys.argv[2])
for step in (16000, 20000):
    cfg = json.loads((root / f"checkpoint-{step}" / "config.json").read_text())
    assert cfg["ht_df"] == df, cfg
    assert cfg["ht_sum_mode"] == "outside", cfg
    assert cfg["ht_sigma_mode"] == "homog", cfg
    assert not cfg.get("ht_mvt", False), cfg
print("CHECKPOINT_LOSS_CONFIG_VERIFIED", df)
PY

# One task per GPU; unique ports and exact prior rollout settings.
export MUJOCO_GL=egl
tasks=(widowx_spoon_on_towel widowx_carrot_on_plate widowx_stack_cube
       widowx_put_eggplant_in_basket widowx_put_eggplant_in_sink
       widowx_open_drawer widowx_close_drawer)
eval_task() (
  set -euo pipefail
  checkpoint=$1
  task=$2
  gpu=$3
  export CUDA_VISIBLE_DEVICES="$gpu"
  port=$((5555 + gpu))
  log_prefix="$run_dir/evaluation/${checkpoint}_${task}"
  .venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path "$run_dir/checkpoint-$checkpoint" \
    --embodiment-tag SIMPLER_ENV_WIDOWX --use-sim-policy-wrapper --port "$port" \
    > "${log_prefix}_server.log" 2>&1 &
  server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
  ready=0
  for attempt in {1..120}; do
    kill -0 "$server_pid" || exit 1
    if grep -Eq 'Server is ready|Listening|serving' "${log_prefix}_server.log"; then
      ready=1
      break
    fi
    sleep 10
  done
  test "$ready" -eq 1
  sleep 20
  gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 50 --policy-client-host 127.0.0.1 --policy-client-port "$port" \
    --max-episode-steps 300 --env-name "simpler_env_widowx/$task" \
    --n-action-steps 4 --n-envs 5 > "${log_prefix}.log" 2>&1
  grep -q 'success rate:' "${log_prefix}.log"
)
for checkpoint in 16000 20000; do
  pids=()
  for gpu in "${!tasks[@]}"; do
    eval_task "$checkpoint" "${tasks[$gpu]}" "$gpu" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  test "$failed" -eq 0
done
python - "$run_dir" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for step in (16000, 20000):
    tasks = {}
    for path in sorted((root / "evaluation").glob(f"{step}_widowx_*.log")):
        if path.stem.endswith("_server"):
            continue
        matches = re.findall(r"success rate:\s*([0-9.]+)", path.read_text())
        assert matches, path
        tasks[path.stem.removeprefix(f"{step}_")] = float(matches[-1])
    assert len(tasks) == 7, tasks
    summary[str(step)] = {"tasks": tasks, "mean_success": sum(tasks.values()) / 7}
(root / "evaluation" / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
echo "TRAIN_AND_EVAL_COMPLETE coordinate_nu=$df"
