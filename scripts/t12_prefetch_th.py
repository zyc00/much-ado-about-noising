"""Prefetch tool_hang ph low_dim (missed by the first prefetch)."""
from huggingface_hub import hf_hub_download

p = hf_hub_download(repo_id="ChaoyiPan/mip-dataset",
                    filename="robomimic/tool_hang/ph/low_dim.hdf5",
                    repo_type="dataset")
print("FETCHED tool_hang ph", p, flush=True)
print("FETCH_TH_DONE", flush=True)
