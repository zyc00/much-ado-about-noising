"""Map every completed cell -> its checkpoint file + the eval it came from.
Proves pod deletion loses nothing: all state lives on the PVC."""
import glob
import json
import os

rows = []
for d in sorted(glob.glob("logs/t12_*")):
    mf, ck = d + "/metrics.jsonl", d + "/models/model_best.pt"
    if not os.path.exists(mf):
        continue
    srs = []
    for line in open(mf):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = ([x for x in r if x.startswith("p4_")] if "kitchen" in d
             else [x for x in r if x.startswith("mean_success_")])
        if k:
            srs.append((r["step"], float(r[sorted(k)[0]])))
    if not srs:
        continue
    bstep, bsr = max(srs, key=lambda t: t[1])
    have = os.path.exists(ck)
    sz = os.path.getsize(ck) / 1e6 if have else 0
    rows.append((d.split("/")[-1], len(srs), bsr, bstep, have, sz))
print(f"cells with metrics: {len(rows)}; with model_best.pt: "
      f"{sum(1 for r in rows if r[4])}")
tot = sum(r[5] for r in rows)
print(f"total checkpoint size: {tot/1000:.1f} GB")
for r in sorted(rows, key=lambda x: -x[2])[:8]:
    print(f"  {r[0][:62]:62s} best {r[2]:.2f} @step {r[3]:6d} "
          f"ckpt {'OK' if r[4] else 'MISSING'} {r[5]:.0f}MB")
