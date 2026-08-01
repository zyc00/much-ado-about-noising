"""Does the proxy forward Range? Small range via proxy and direct."""
import requests
from huggingface_hub import hf_hub_url

u = hf_hub_url("ChaoyiPan/mip-dataset",
               "robomimic/lift/ph/image_abs.hdf5", repo_type="dataset")
for tag, px in [("proxy", {"https": "http://172.17.0.12:2080",
                           "http": "http://172.17.0.12:2080"}),
                ("direct", None)]:
    try:
        r = requests.get(u, headers={"Range": "bytes=0-1048575"},
                         timeout=(15, 30), proxies=px, stream=True)
        n = 0
        for ch in r.iter_content(1 << 16):
            n += len(ch)
            if n >= (4 << 20):
                break
        print(f"{tag}: status {r.status_code} got {n} bytes "
              f"(206=range honored)", flush=True)
    except Exception as e:
        print(f"{tag}: {type(e).__name__} {str(e)[:80]}", flush=True)
