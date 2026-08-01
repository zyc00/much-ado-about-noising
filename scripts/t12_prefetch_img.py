"""Prefetch Table-13 image datasets into the PVC HF cache (big files)."""
from huggingface_hub import hf_hub_download

FILES = [f"robomimic/{t}/{v}/image_abs.hdf5"
         for t in ["lift", "can", "square", "tool_hang", "transport"]
         for v in (["ph"] if t == "tool_hang" else ["mh", "ph"])]
FILES.append("pusht/pusht_cchi_v7_replay.zarr.zip")
for f in FILES:
    p = hf_hub_download(repo_id="ChaoyiPan/mip-dataset", filename=f,
                        repo_type="dataset")
    print("FETCHED", f, flush=True)
print("FETCH_IMG_ALL_DONE", flush=True)
