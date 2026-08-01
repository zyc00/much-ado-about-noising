"""best / last-5 mean of in-run eval mean_success from metrics.jsonl."""
import json
import sys

for d in sys.argv[1:]:
    try:
        rows = [json.loads(x) for x in open(f"logs/{d}/metrics.jsonl")]
        ev = [(r.get("step", 0), v) for r in rows for k, v in r.items()
              if "mean_success" in k and "avg" not in k]
        if not ev:
            print(f"B5 {d} NO-EVALS")
            continue
        vals = [v for _, v in ev]
        last5 = vals[-5:]
        print(f"B5 {d} n_evals {len(vals)} best {max(vals):.2f} "
              f"last5 {sum(last5)/len(last5):.2f} series "
              + " ".join(f"{v:.2f}" for v in vals))
    except Exception as e:
        print(f"B5 {d} ERR {e!r}")
