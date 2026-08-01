import glob
import json
import sys

pat = sys.argv[1] if len(sys.argv) > 1 else "transport_mh"
for d in sorted(glob.glob(f"logs/t12_*{pat}*")):
    srs, steps = [], []
    try:
        for line in open(d + "/metrics.jsonl"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "mean_success_1" in r:
                srs.append(r["mean_success_1"])
                steps.append(r["step"])
            last = r.get("step", 0)
    except OSError:
        continue
    print(d.split("/")[-1], f"trainstep~{last}",
          " ".join(f"{int(s/1000)}k:{v:.2f}" for s, v in zip(steps, srs)))
