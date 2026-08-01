"""Direct test of the claim 'FFN is not necessary, it's just weight
rebalancing': hard-example UP-weighting (pw*/pema) vs precision
DOWN-weighting control (ipw) on plain MLP, against the known mlp+mse /
lff5+mse references. Appends to results.jsonl."""
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = __file__.rsplit("/", 1)[0]
OUT = f"{HERE}/results.jsonl"
EXPS = ["sup_clean", "nfq", "nfq_tnoise"]
LOSSES = ["pw05", "pw1", "pw2", "pema", "ipw"]
SEEDS = range(8)


def run(cell):
    exp, loss, seed = cell
    r = subprocess.run([sys.executable, f"{HERE}/run_toy.py", exp, "mlp",
                        loss, str(seed), OUT], capture_output=True, text=True)
    print((r.stdout.strip() or r.stderr[-200:]), flush=True)


if __name__ == "__main__":
    cells = list(itertools.product(EXPS, LOSSES, SEEDS))
    with ProcessPoolExecutor(max_workers=6) as ex:
        list(ex.map(run, cells))
    agg, seen = defaultdict(list), set()
    for line in open(OUT):
        r = json.loads(line)
        k = (r["exp"], r["arch"], r["loss"], r["seed"])
        if k in seen:
            continue
        seen.add(k)
        agg[k[:3]].append(r["final_rmse"])
    print("\nREBAL exp loss mean se")
    for e in EXPS:
        for arch, ls in [("mlp", "mse"), ("lff5", "mse")] + \
                [("mlp", l) for l in LOSSES]:
            v = agg.get((e, arch, ls), [])
            if v:
                print(f"REBAL {e} {arch}+{ls} {np.mean(v):.4f} "
                      f"{np.std(v) / np.sqrt(len(v)):.4f}", flush=True)
