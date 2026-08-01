"""Download LIBERO benchmark datasets (HDF5) to the PVC."""
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
from huggingface_hub import snapshot_download

p = snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
                      local_dir="/mnt/pfs/yuchen/data/libero",
                      max_workers=4)
print("LIBERO_DONE", p, flush=True)
