"""Find the dm_control release pinning mujoco==3.3.7."""
import subprocess
import sys
import zipfile
import glob
import os
import shutil

r = subprocess.run([sys.executable, "-m", "pip", "index", "versions",
                    "dm_control"], capture_output=True, text=True)
line = [x for x in r.stdout.splitlines() if "Available" in x]
vers = line[0].split(":", 1)[1].split(",") if line else []
vers = [v.strip() for v in vers][:10]
print("candidates", vers)
for v in vers:
    d = "/tmp/dmc"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    r = subprocess.run([sys.executable, "-m", "pip", "download",
                        f"dm_control=={v}", "--no-deps", "-q", "-d", d],
                       capture_output=True, text=True)
    whl = glob.glob(d + "/*.whl")
    if not whl:
        print(v, "no wheel")
        continue
    with zipfile.ZipFile(whl[0]) as z:
        meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
        pins = [ln for ln in z.read(meta).decode().splitlines()
                if "mujoco" in ln.lower() and "Requires" in ln]
    print(v, pins)
    if any("3.3.7" in p for p in pins):
        print("MATCH", v)
        break
