"""Where does the flat-loss drift live? Per-input-column first-layer weight drift
||W1[:,c](t+1) - W1[:,c](t)|| across snapshots, grouped: DISTRACTOR (base/tool blocks,
flags) vs TASK (frame-rel, eef, grip) vs FRAMEPOSE (frame world pose). Also normalized by
column weight norm. Models: mse_timeline, mip_timeline."""
import os, sys, glob, re
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch

D53 = {"DISTRACTOR": list(range(0, 14)) + list(range(28, 42)) + [42, 43],
       "TASK": list(range(14, 21)) + list(range(44, 53)),
       "FRAMEPOSE": list(range(21, 28))}
GROUPS = {k: v + [d + 53 for d in v] for k, v in D53.items()}

def first_layer(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    # find EMA encoder first linear weight (in_features 106)
    cand = []
    for key in sd:
        if isinstance(sd[key], dict):
            for k2, v in sd[key].items():
                if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[1] == 106:
                    cand.append((f"{key}.{k2}", v))
    ema = [c for c in cand if "ema" in c[0].lower()]
    pick = (ema or cand)[0]
    return pick[0], pick[1].float().numpy()

for tag, pat in [("MSE", "logs/mse_timeline/models/snap_*.pt"),
                 ("MIP", "logs/mip_timeline/models/snap_*.pt")]:
    snaps = sorted(glob.glob(pat), key=lambda p: int(re.search(r"snap_(\d+)", p).group(1)))
    prev = None; prev_step = None; name_printed = False
    for ck in snaps:
        step = int(re.search(r"snap_(\d+)", ck).group(1))
        key, W = first_layer(ck)
        if not name_printed:
            print(f"WDRIFT {tag}: using {key} shape={W.shape}", flush=True)
            name_printed = True
        if prev is not None:
            dW = W - prev
            line = f"WDRIFT {tag} {prev_step}->{step}:"
            for g, dims in GROUPS.items():
                drift = np.linalg.norm(dW[:, dims]) / (np.linalg.norm(prev[:, dims]) + 1e-9)
                line += f" {g}={drift:.4f}"
            print(line, flush=True)
        prev = W; prev_step = step
print("WDRIFT-DONE")
