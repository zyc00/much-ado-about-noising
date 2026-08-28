"""Harvest the dp_harness campaign: best / last-5 per (task, net, config)."""
import glob
import json
import re
from collections import defaultdict

import numpy as np

TARGET = {"best": 1.00, "last5": 0.864}   # DP-T tool-hang published


def decode(sfx):
    """Turn the flattened override suffix back into readable config tags."""
    s = sfx.replace("optimizationdpharnesslogdpharness", "")
    out = []
    if "tasknumenvs22" in s:
        out.append("DPproto[160x22]")
        s = s.replace("tasknumenvs22", "")
    if "taskobssteps8" in s:
        out.append("obs8")
        s = s.replace("taskobssteps8", "")
    m = re.search(r"networkconddropoutrate0(\d)", s)
    if m:
        out.append(f"cd0.{m.group(1)}")
        s = re.sub(r"networkconddropoutrate0\d", "", s)
    m = re.search(r"networkattndropout0(\d)", s)
    if m:
        out.append(f"ad0.{m.group(1)}")
        s = re.sub(r"networkattndropout0\d", "", s)
    if "optimizationemapower00" in s:
        out.append("emaConst")
        s = s.replace("optimizationemapower00", "")
    m = re.search(r"optimizationemamax0(\d+)", s)
    if m:
        out.append(f"emaMax0.{m.group(1)}")
        s = re.sub(r"optimizationemamax0\d+", "", s)
    m = re.search(r"optimizationbatchsize(\d+)", s)
    if m:
        out.append(f"bs{m.group(1)}")
        s = re.sub(r"optimizationbatchsize\d+", "", s)
    m = re.search(r"optimizationlr(\d)e(\d)", s)
    if m:
        out.append(f"lr{m.group(1)}e-{m.group(2)}")
        s = re.sub(r"optimizationlr\de\d", "", s)
    m = re.search(r"optimizationweightdecay1e(\d)", s)
    if m:
        out.append(f"wd1e-{m.group(1)}")
    return ",".join(out) if out else "base(cd0.2,wd1e-3,emaProg)"


g = defaultdict(list)
for d in sorted(glob.glob("logs/t12_*dp_harness*")):
    try:
        rows = [json.loads(l) for l in open(d + "/metrics.jsonl") if l.strip()]
    except OSError:
        continue
    ev = [x["mean_success_1"] for x in rows if "mean_success_1" in x]
    st = [x["step"] for x in rows if "loss" in x]
    if not ev or not st:
        continue
    b = d.split("/")[-1].replace("t12_", "")
    m = re.match(r"(.+?)_(chiunet|chitransformer|sudeepdit)_dp_harness_s(\d)_ht_(.*)", b) \
        or re.match(r"(.+?)_(sudeepdit)_s(\d)_ht_(.*)", b)
    if not m:
        continue
    g[(m.group(1), m.group(2), decode(m.group(4)))].append(
        (max(ev), float(np.mean(ev[-5:])), max(st), len(ev))
    )

print(f"{'task':26s} {'net':14s} {'config':26s} {'n':>2s} {'step':>6s} "
      f"{'best':>5s} {'last5':>6s}  vs 1.00/0.864")
for (t, n, c), v in sorted(g.items(), key=lambda kv: -np.mean([x[1] for x in kv[1]])):
    b = np.mean([x[0] for x in v])
    l5 = np.mean([x[1] for x in v])
    st = min(x[2] for x in v)
    flag = ""
    if b >= TARGET["best"]:
        flag += " BEST=1.0"
    if l5 >= TARGET["last5"]:
        flag += " L5>=0.864"
    print(f"{t[:26]:26s} {n:14s} {c[:26]:26s} {len(v):2d} {st // 1000:5d}k "
          f"{b:5.2f} {l5:6.2f}{flag}")
