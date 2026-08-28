"""Vision run health: steps, IO share, staleness, cross-task collisions."""
import glob
import json
import os
import time

now = time.time()
rows = []
for d in sorted(glob.glob("logs/t12_*image*")):
    f = d + "/metrics.jsonl"
    try:
        age = now - os.path.getmtime(f)
        last = {}
        n = 0
        for line in open(f):
            try:
                last = json.loads(line)
                n += 1
            except json.JSONDecodeError:
                pass
        rows.append((d.split("/")[-1], last.get("step", -1),
                     last.get("perf/data_load_ms", 0),
                     last.get("perf/update_ms", 0), age, n))
    except OSError:
        rows.append((d.split("/")[-1], -1, 0, 0, -1, 0))
print(f"{'run':52s} {'step':>7s} {'io_ms':>7s} {'upd_ms':>7s} {'age_s':>7s}")
stuck = 0
for r in sorted(rows, key=lambda x: -x[4]):
    flag = ""
    if r[4] > 1800:
        flag = " STALE"
        stuck += 1
    elif r[2] > 2 * max(r[3], 1):
        flag = " IO-BOUND"
    print(f"{r[0][:52]:52s} {r[1]:7d} {r[2]:7.0f} {r[3]:7.0f} {r[4]:7.0f}{flag}")
print(f"total {len(rows)} runs, {stuck} stale (>30min no log)")
