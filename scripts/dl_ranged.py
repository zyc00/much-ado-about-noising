"""Parallel byte-range downloader tuned for a connection that drops
every few MB. 4MB chunks, 8 workers, offset writes, chunk-level resume
via sidecar. Usage: dl_ranged.py <repo_file> [more files...]"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from huggingface_hub import hf_hub_url

BASE = os.environ.get("DL_BASE", "")
PX = os.environ.get("DL_PROXY", "http://172.17.0.12:2080")
PROXIES = None if BASE else {"https": PX, "http": PX}
CHUNK = 1 << 20
lock = threading.Lock()


def fetch(url, dest, size, done, sidecar):
    n_ok = [sum(done)]
    t0 = time.time()

    def one(i):
        if done[i]:
            return
        lo, hi = i * CHUNK, min((i + 1) * CHUNK, size) - 1
        for a in range(200):
            try:
                with requests.get(url,
                                  headers={"Range": f"bytes={lo}-{hi}"},
                                  timeout=(15, 60), proxies=PROXIES,
                                  stream=True) as r:
                    if r.status_code not in (200, 206):
                        raise IOError(f"status {r.status_code}")
                    buf = b""
                    for ch in r.iter_content(1 << 16):
                        buf += ch
                        if len(buf) > hi - lo + 1:
                            break
                if len(buf) == hi - lo + 1:
                    with lock:
                        with open(dest, "r+b") as w:
                            w.seek(lo)
                            w.write(buf)
                        done[i] = True
                        n_ok[0] += 1
                        if n_ok[0] % 50 == 0:
                            json.dump(done, open(sidecar, "w"))
                            rate = n_ok[0] * CHUNK / max(time.time() - t0, 1)
                            print(f"PROG {os.path.basename(dest)} "
                                  f"{n_ok[0]}/{len(done)} chunks "
                                  f"{rate/1e6:.2f}MB/s", flush=True)
                    return
            except Exception:
                pass
            time.sleep(1 + a % 5)
        print(f"CHUNK_FAIL {lo}", flush=True)

    with ThreadPoolExecutor(max_workers=int(os.environ.get("DL_WORKERS", "8"))) as ex:
        list(ex.map(one, range(len(done))))
    return all(done)


for f in sys.argv[1:]:
    dest = "/mnt/pfs/yuchen/data/mip/" + f
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print("FETCHED", f, "(already)", flush=True)
        continue
    if BASE:
        url = BASE.rstrip("/") + "/" + f
    else:
        url = hf_hub_url("ChaoyiPan/mip-dataset", f, repo_type="dataset")
    size = None
    for a in range(60):
        try:
            h = requests.get(url, headers={"Range": "bytes=0-0"},
                             allow_redirects=True, timeout=(15, 20),
                             proxies=PROXIES)
            if h.status_code in (200, 206):
                cr = h.headers.get("Content-Range", "")
                if "/" in cr:
                    size = int(cr.rsplit("/", 1)[1])
                elif h.status_code == 200:
                    size = int(h.headers["Content-Length"])
            if size and size > (1 << 20):
                break
            size = None
        except Exception:
            pass
        time.sleep(3)
    if size is None:
        print("HEAD_FAIL", f, flush=True)
        continue
    part, sidecar = dest + ".rpart", dest + ".rmeta"
    nch = (size + CHUNK - 1) // CHUNK
    if os.path.exists(sidecar) and os.path.exists(part):
        done = json.load(open(sidecar))
        if len(done) != nch:
            done = [False] * nch
    else:
        done = [False] * nch
        with open(part, "wb") as w:
            w.truncate(size)
    print(f"START {f} {size/1e9:.2f}GB {nch} chunks "
          f"({sum(done)} done)", flush=True)
    if fetch(url, part, size, done, sidecar):
        os.rename(part, dest)
        os.remove(sidecar) if os.path.exists(sidecar) else None
        print("FETCHED", f, flush=True)
    else:
        json.dump(done, open(sidecar, "w"))
        print("INCOMPLETE", f, flush=True)
print("RANGED_ALL_DONE", flush=True)
