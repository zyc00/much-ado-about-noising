"""Print eval score histories from logs/<tag>/metrics.jsonl for given tags."""
import json
import sys

for tag in sys.argv[1:]:
    try:
        vals = []
        for line in open(f"logs/{tag}/metrics.jsonl"):
            if "mean_success" not in line:
                continue
            d = json.loads(line)
            v = d.get("mean_success_1")
            if v is not None:
                vals.append((d.get("step"), v))
        print(f"{tag}: " + " ".join(f"{s//1000}k:{v:.2f}" for s, v in vals))
    except FileNotFoundError:
        print(f"{tag}: no metrics")
