"""Serial fetch of ALL Table-12/13 datasets with timeouts + retries.
Order: tool_hang low_dim first (unblocks state wave-1), then state,
then image, then pusht. Prints FETCHED <file> per success."""
import os
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
from huggingface_hub import hf_hub_download

FILES = ["robomimic/tool_hang/ph/low_dim.hdf5"]
FILES += [f"robomimic/{t}/{v}/low_dim.hdf5"
          for t in ["can", "square", "transport"] for v in ["mh", "ph"]]
FILES += [f"robomimic/{t}/{v}/image_abs.hdf5"
          for t in ["lift", "can", "square", "tool_hang", "transport"]
          for v in (["ph"] if t == "tool_hang" else ["mh", "ph"])]
FILES.append("pusht/pusht_cchi_v7_replay.zarr.zip")
for f in FILES:
    for attempt in range(5):
        try:
            hf_hub_download(repo_id="ChaoyiPan/mip-dataset", filename=f,
                            repo_type="dataset")
            print("FETCHED", f, flush=True)
            break
        except Exception as e:
            print(f"RETRY {f} attempt {attempt}: {type(e).__name__} "
                  f"{str(e)[:120]}", flush=True)
            time.sleep(20)
    else:
        print("FAILED", f, flush=True)
print("FETCH_COMBINED_DONE", flush=True)
