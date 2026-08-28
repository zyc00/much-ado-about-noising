"""Harvest HT-in-DP runs and compare against DP's published tool-hang logs."""
import glob
import json
import os
import re

import numpy as np

DP_ROOT = "/mnt/pfs/yuchen/dp/runs"
# DP published (tool_hang ph low_dim, 3 seeds, their own logs)
DP_REF = {
    "transformer": {"best": 1.000, "last5": 0.864, "last15": 0.866},
    "cnn": {"best": 0.864, "last5": 0.548, "last15": 0.524},
}


def series(path):
    try:
        txt = open(path).read()
    except OSError:
        return []
    return [float(m) for m in re.findall(r'"test/mean_score": ([0-9.]+)', txt)]


def epochs(path):
    try:
        txt = open(path).read()
    except OSError:
        return 0
    e = re.findall(r'"epoch": ([0-9]+)', txt)
    return int(e[-1]) if e else 0


groups = {}
for d in sorted(glob.glob(f"{DP_ROOT}/*_s[0-9]")):
    tag = os.path.basename(d).rsplit("_s", 1)[0]
    v = series(f"{d}/logs.json.txt")
    if v:
        groups.setdefault(tag, []).append((v, epochs(f"{d}/logs.json.txt")))

print(f"{'arm':22s} {'n':>2s} {'ep':>6s} {'best':>6s} {'last5':>6s} {'last15':>7s}  verdict")
for tag, runs in sorted(groups.items()):
    n = len(runs)
    best = np.mean([max(v) for v, _ in runs])
    l5 = np.mean([np.mean(v[-5:]) for v, _ in runs])
    l15 = np.mean([np.mean(v[-15:]) for v, _ in runs])
    ep = int(np.mean([e for _, e in runs]))
    arch = "cnn" if ("unet" in tag or tag.startswith("sw_u_")) else "transformer"
    ref = DP_REF[arch]
    flags = []
    if best >= ref["best"]:
        flags.append("BEST>=DP")
    if l5 >= ref["last5"]:
        flags.append("L5>=DP")
    if l15 >= ref["last15"]:
        flags.append("L15>=DP")
    print(f"{tag:22s} {n:2d} {ep:6d} {best:6.3f} {l5:6.3f} {l15:7.3f}  "
          f"vs {arch[:2]} ({ref['best']:.2f}/{ref['last5']:.2f}/{ref['last15']:.2f}) "
          f"{' '.join(flags) if flags else ''}")
