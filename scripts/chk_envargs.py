import json
import sys

import h5py

for p in sys.argv[1:]:
    try:
        with h5py.File(p, "r") as f:
            ea = json.loads(f["data"].attrs["env_args"])
            print(p.split("/")[-3:], "cams:",
                  ea.get("env_kwargs", {}).get("camera_names"),
                  "| env:", ea.get("env_name"))
            print("   obs keys:", list(f["data/demo_0/obs"].keys())[:8])
    except Exception as e:
        print(p, "ERR", type(e).__name__, str(e)[:80])
