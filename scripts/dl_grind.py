"""Serial grinder: stream files with aggressive Range-resume (defeats
mid-stream CDN/NAT resets). Usage: dl_grind.py state|image"""
import os
import sys
import time

import requests
from huggingface_hub import hf_hub_url

STATE = ["robomimic/tool_hang/ph/low_dim.hdf5"] + \
    [f"robomimic/{t}/{v}/low_dim.hdf5"
     for t in ["lift", "can", "square", "transport"] for v in ["mh", "ph"]] + \
    ["pusht/pusht_cchi_v7_replay.zarr.zip",
     "kitchen/kitchen_demos_multitask.zip"]
IMAGE = [f"robomimic/{t}/{v}/image_abs.hdf5"
         for t in ["square", "tool_hang", "can", "lift", "transport"]
         for v in (["ph"] if t == "tool_hang" else ["mh", "ph"])]
FILES = STATE if sys.argv[1] == "state" else IMAGE
for f in FILES:
    dest = "/mnt/pfs/yuchen/data/mip/" + f
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print("FETCHED", f, "(already)", flush=True)
        continue
    part = dest + ".part"
    url = hf_hub_url("ChaoyiPan/mip-dataset", f, repo_type="dataset")
    t0, done = time.time(), False
    for a in range(2000):
        try:
            pos = os.path.getsize(part) if os.path.exists(part) else 0
            hd = {"Range": f"bytes={pos}-"} if pos else {}
            with requests.get(url, stream=True, timeout=(15, 20),
                              headers=hd) as r:
                if pos and r.status_code == 200:
                    pos = 0
                r.raise_for_status()
                with open(part, "ab" if pos else "wb") as w:
                    for ch in r.iter_content(1 << 20):
                        w.write(ch)
            os.rename(part, dest)
            sz = os.path.getsize(dest)
            print(f"FETCHED {f} {sz/1e9:.2f}GB in {time.time()-t0:.0f}s "
                  f"({a+1} attempts)", flush=True)
            done = True
            break
        except Exception as e:
            if a % 25 == 0:
                pos2 = os.path.getsize(part) if os.path.exists(part) else 0
                print(f"GRIND {f} a{a} {pos2/1e9:.2f}GB "
                      f"{type(e).__name__}", flush=True)
            time.sleep(2)
    if not done:
        print("FAILED", f, flush=True)
print("GRIND_DONE", sys.argv[1], flush=True)
