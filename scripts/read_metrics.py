import json
import sys

for d in ["mp200_l2_s1000", "mp200_ht_s1000", "mp200_mip_s1000",
          "hbase2", "hheterot_s1000", "hmip0"]:
    try:
        rows = [json.loads(x) for x in open(f"logs/{d}/metrics.jsonl")]
        tr = [r for r in rows if any("loss" in k for k in r)]
        last = tr[-1]
        kept = {k: v for k, v in last.items()
                if "loss" in k or "step" in k}
        print("MJ", d, kept, flush=True)
    except Exception as e:
        print("MJ", d, "ERR", repr(e), flush=True)
