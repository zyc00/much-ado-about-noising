import glob
import re

ds = sorted(glob.glob("logs/t12_*"))
print("dirs", len(ds))
NAME_RE = re.compile(r"t12_(?P<task>.+?)_(?P<net>chiunet|chitransformer|"
                     r"sudeepdit)_s(?P<seed>\d+)")
n = 0
for d in ds:
    m = NAME_RE.search(d)
    ok = False
    try:
        for line in open(d + "/metrics.jsonl"):
            if "mean_success_1" in line:
                ok = True
                break
    except OSError:
        pass
    n += ok
    print(d.split("/")[-1], "re" if m else "NO_RE", "ev" if ok else "no_ev")
print("with_evals", n)
