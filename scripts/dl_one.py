"""Lock-free download of ONE mip-dataset file straight to PFS.
Streams from the resolved CDN URL with Range-resume; no hf filelock
(PFS flock hangs). Usage: dl_one.py <filename> [repo_id]"""
import os
import sys
import time

import requests
from huggingface_hub import hf_hub_url

PX = os.environ.get("DL_PROXY", "http://172.17.0.12:2080")
PROXIES = {"https": PX, "http": PX} if PX else None

f = sys.argv[1]
repo = sys.argv[2] if len(sys.argv) > 2 else "ChaoyiPan/mip-dataset"
dest = "/mnt/pfs/yuchen/data/mip/" + f
os.makedirs(os.path.dirname(dest), exist_ok=True)
if os.path.exists(dest):
    print("FETCHED", f, "(already)", flush=True)
    sys.exit(0)
part = dest + ".part"
url = hf_hub_url(repo, f, repo_type="dataset")
for attempt in range(30):
    try:
        pos = os.path.getsize(part) if os.path.exists(part) else 0
        hd = {"Range": f"bytes={pos}-"} if pos else {}
        with requests.get(url, stream=True, timeout=(30, 120), proxies=PROXIES,
                          headers=hd, allow_redirects=True) as r:
            if pos and r.status_code == 200:
                pos = 0  # server ignored Range; restart
            r.raise_for_status()
            with open(part, "ab" if pos else "wb") as w:
                n = pos
                for ch in r.iter_content(1 << 22):
                    w.write(ch)
                    n += len(ch)
        os.rename(part, dest)
        print("FETCHED", f, f"{n/1e9:.2f}GB", flush=True)
        break
    except Exception as e:
        print(f"RETRY {f} a{attempt} {type(e).__name__} {str(e)[:100]}",
              flush=True)
        time.sleep(15)
else:
    print("FAILED", f, flush=True)
