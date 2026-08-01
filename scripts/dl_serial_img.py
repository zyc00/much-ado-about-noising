"""Serial proxied download of image_abs files, smallest-first, resume."""
import os
import subprocess
import sys
import time

FILES = [f"robomimic/{t}/{v}/image_abs.hdf5" for t, v in [
    ("lift", "ph"), ("lift", "mh"), ("can", "ph"), ("square", "ph"),
    ("can", "mh"), ("square", "mh"), ("transport", "ph"),
    ("transport", "mh"), ("tool_hang", "ph")]]
t0 = time.time()
for f in FILES:
    print(f"SERIAL start {f} t+{time.time()-t0:.0f}s", flush=True)
    subprocess.run([sys.executable, "-u", "scripts/dl_one.py", f])
print("SERIAL_ALL_DONE", flush=True)
