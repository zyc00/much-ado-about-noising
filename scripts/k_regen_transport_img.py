"""Regenerate transport image datasets from robomimic raw demos (state replay)."""
import os
import subprocess
import sys

from huggingface_hub import hf_hub_download

split = os.environ.get("SPLIT", "mh")
raw = hf_hub_download(repo_id="amandlek/robomimic", repo_type="dataset",
                      filename=f"v1.5/transport/{split}/demo_v15.hdf5")
out = f"data/transport_{split}_image4.hdf5"
script = os.path.join(os.path.dirname(__import__("robomimic").__file__), "scripts", "dataset_states_to_obs.py")
cmd = [sys.executable, script, "--dataset", raw, "--output_name", os.path.abspath(out),
       "--done_mode", "2", "--camera_names", "shouldercamera0", "shouldercamera1",
       "robot0_eye_in_hand", "robot1_eye_in_hand",
       "--camera_height", "84", "--camera_width", "84"]
print("RUN", " ".join(cmd), flush=True)
r = subprocess.run(cmd)
print("REGEN-DONE" if r.returncode == 0 else f"REGEN-FAILED rc={r.returncode}")
