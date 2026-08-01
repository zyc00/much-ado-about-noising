"""Prefetch Table-12 robomimic datasets into the PVC HF cache."""
from huggingface_hub import hf_hub_download

for t in ["lift", "can", "square", "transport"]:
    for v in ["mh", "ph"]:
        p = hf_hub_download(repo_id="ChaoyiPan/mip-dataset",
                            filename=f"robomimic/{t}/{v}/low_dim.hdf5",
                            repo_type="dataset")
        print("FETCHED", t, v, p, flush=True)
print("FETCH_ALL_DONE", flush=True)
