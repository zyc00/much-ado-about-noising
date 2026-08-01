"""Full sweep: 5 experiments x arms x 8 seeds, parallel workers.
Writes results.jsonl (one line per run) then prints a mean+-se table."""
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = __file__.rsplit("/", 1)[0]
OUT = f"{HERE}/results.jsonl"
EXPS = ["sup_clean", "sup_gnoise", "sup_tnoise", "nfq", "nfq_tnoise"]
ARMS = [("mlp", l) for l in ["mse", "huber", "ht", "hg", "mip"]] + \
       [("lff5", l) for l in ["mse", "huber", "ht", "hg", "mip"]] + \
       [("lff1", "mse")]
SEEDS = range(8)


def run(cell):
    exp, (arch, loss), seed = cell
    r = subprocess.run([sys.executable, f"{HERE}/run_toy.py", exp, arch,
                        loss, str(seed), OUT], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL {exp} {arch} {loss} s{seed}: {r.stderr[-300:]}",
              flush=True)
    else:
        print(r.stdout.strip(), flush=True)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    done = set()
    try:
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["exp"], r["arch"], r["loss"], r["seed"]))
    except FileNotFoundError:
        pass
    cells = [c for c in itertools.product(EXPS, ARMS, SEEDS)
             if (c[0], c[1][0], c[1][1], c[2]) not in done]
    print(f"{len(cells)} runs, {nw} workers", flush=True)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        list(ex.map(run, cells))

    agg = defaultdict(list)
    for line in open(OUT):
        r = json.loads(line)
        agg[(r["exp"], r["arch"], r["loss"])].append(r["final_rmse"])
    print("\nQTABLE exp arch loss final_rmse_mean se n")
    for (exp, arch, loss), v in sorted(agg.items()):
        v = np.array(v)
        print(f"QTABLE {exp} {arch} {loss} {v.mean():.4f} "
              f"{v.std() / np.sqrt(len(v)):.4f} {len(v)}", flush=True)
