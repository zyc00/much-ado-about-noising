import glob
import json
import os
import time

now = time.time()
for d in sorted(glob.glob("logs/t12_transport*")):
    f = d + "/metrics.jsonl"
    try:
        age = now - os.path.getmtime(f)
        last = {}
        for line in open(f):
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                pass
        print(d.split("/")[-1], f"step {last.get('step', '?')}",
              f"sps {last.get('perf/steps_per_sec', 0):.2f}",
              f"log_age {age:.0f}s")
    except OSError as e:
        print(d.split("/")[-1], "NO METRICS", type(e).__name__)
