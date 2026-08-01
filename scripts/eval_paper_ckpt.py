"""Evaluate a paper checkpoint (HF mip-checkpoints) under OUR official harness."""
import os, subprocess, sys
from huggingface_hub import hf_hub_download
f = os.environ["CKF"]
p = hf_hub_download(repo_id="ChaoyiPan/mip-checkpoints", filename=f)
task = os.environ.get("TASK", "square_mh_state_delta_legacy")
loss = os.environ.get("LOSS", "regression")
eps = os.environ.get("EPS", "100")
cmd = ["python", "-u", "examples/train_robomimic.py", "mode=eval", f"task={task}", "network=chiunet",
       f"optimization.loss_type={loss}", f"optimization.model_path={p}",
       f"log.eval_episodes={eps}", "log.wandb_mode=disabled", "log.save_video=false",
       "optimization.auto_resume=false"]
r = subprocess.run(cmd, capture_output=True, text=True)
for line in (r.stdout + r.stderr).splitlines():
    if "mean_success_1 - " in line or "Error" in line:
        print(f"PAPERCKPT {os.path.basename(f)} | {line.split('mean_success_1 - ')[-1] if 'mean_success' in line else line}", flush=True)
print("PAPERCKPT-DONE", os.path.basename(f))
