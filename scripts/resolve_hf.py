"""Resolve a ChaoyiPan/mip-dataset file to its local HF-cache path."""
import sys

from huggingface_hub import hf_hub_download

print(hf_hub_download(repo_id="ChaoyiPan/mip-dataset", filename=sys.argv[1], repo_type="dataset"))
