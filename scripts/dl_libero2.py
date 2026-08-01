"""Lock-free LIBERO download: stream every repo file to PFS."""
import os
import time

import requests
from huggingface_hub import HfApi, hf_hub_url

PX = os.environ.get("DL_PROXY", "http://172.17.0.12:2080")
PROXIES = {"https": PX, "http": PX}

R = "yifengzhu-hf/LIBERO-datasets"
files = [f for f in HfApi().list_repo_files(R, repo_type="dataset")
         if not f.startswith(".")]
print(f"LIBERO {len(files)} files", flush=True)
for f in files:
    dest = "/mnt/pfs/yuchen/data/libero/" + f
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        continue
    part, url = dest + ".part", hf_hub_url(R, f, repo_type="dataset")
    for a in range(20):
        try:
            pos = os.path.getsize(part) if os.path.exists(part) else 0
            hd = {"Range": f"bytes={pos}-"} if pos else {}
            with requests.get(url, stream=True, timeout=(30, 120), proxies=PROXIES,
                              headers=hd) as r:
                if pos and r.status_code == 200:
                    pos = 0
                r.raise_for_status()
                with open(part, "ab" if pos else "wb") as w:
                    for ch in r.iter_content(1 << 22):
                        w.write(ch)
            os.rename(part, dest)
            print("FETCHED", f, flush=True)
            break
        except Exception as e:
            print(f"RETRY {f} {type(e).__name__}", flush=True)
            time.sleep(10)
    else:
        print("FAILED", f, flush=True)
print("LIBERO_ALL_DONE", flush=True)
