"""Table-12 HT campaign: harvest in-train eval series from krun logs,
build the best/last-5 table vs MIP references, and upload qualifying
best checkpoints (cell 3-seed-mean best >= MIP best) to HuggingFace.

Run ON A POD (PVC access): parses /mnt/pfs/yuchen/.krun-logs/yuchen-t12*.log,
checkpoints at logs/t12_*/models/model_best.pt.
  python scripts/t12_harvest_upload.py            # harvest + table only
  python scripts/t12_harvest_upload.py --upload   # also upload qualifiers
Env: HF_TOKEN required for --upload. Repo private by default (flip public
at release).
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

# MIP reference row (best/last5), from paper Table 12
MIP_REF = {
    ("lift_mh", "sudeepdit"): (1.00, 0.99),
    ("lift_ph", "sudeepdit"): (1.00, 1.00),
    ("can_mh", "sudeepdit"): (0.98, 0.95),
    ("can_ph", "sudeepdit"): (1.00, 1.00),
    ("square_mh", "sudeepdit"): (0.82, 0.81),  # best: released ckpt filename mean (77/87/82); last5: paper (no artifact)
    ("square_ph", "sudeepdit"): (0.98, 0.94),
    ("transport_mh", "sudeepdit"): (0.44, 0.38),
    ("transport_ph", "sudeepdit"): (0.68, 0.55),  # released abs (paper delta unreproducible: no artifact)
    ("tool_hang_ph", "sudeepdit"): (0.50, 0.37),  # released-artifacts recomputation (PART CDXLVI)
    ("lift_mh", "chitransformer"): (1.00, 1.00),
    ("lift_ph", "chitransformer"): (1.00, 1.00),
    ("can_mh", "chitransformer"): (0.96, 0.95),
    ("can_ph", "chitransformer"): (1.00, 1.00),
    ("square_mh", "chitransformer"): (0.80, 0.73),  # best: released ckpt filename mean (77/87/77); last5: paper
    ("square_ph", "chitransformer"): (0.96, 0.89),
    ("transport_mh", "chitransformer"): (0.42, 0.37),
    ("transport_ph", "chitransformer"): (0.69, 0.58),  # released abs (paper delta unreproducible: no artifact)
    ("tool_hang_ph", "chitransformer"): (0.74, 0.62),  # released-artifacts recomputation (PART CDXLVI)
    ("lift_mh", "chiunet"): (1.00, 1.00),
    ("lift_ph", "chiunet"): (1.00, 1.00),
    ("can_mh", "chiunet"): (1.00, 0.98),
    ("can_ph", "chiunet"): (1.00, 0.99),
    ("square_mh", "chiunet"): (0.89, 0.81),  # best: released ckpt filename mean (87/87/92; paper 0.92 = max seed); last5: paper
    ("square_ph", "chiunet"): (1.00, 0.94),
    ("transport_mh", "chiunet"): (0.62, 0.46),
    ("transport_ph", "chiunet"): (0.81, 0.66),  # released abs
    ("tool_hang_ph", "chiunet"): (0.57, 0.43),  # released-artifacts recomputation (PART CDXLVI)
    ("kitchen_state", "sudeepdit"): (1.0, 0.97),
    ("kitchen_state", "chitransformer"): (0.98, 0.96),
    ("kitchen_state", "chiunet"): (1.0, 0.96),
}
LOGDIR = "/mnt/pfs/yuchen/.krun-logs"
NAME_RE = re.compile(
    r"t12_(?P<task>.+?)_(?P<net>chiunet|chitransformer|"
    r"sudeepdit)_s(?P<seed>\d+)(?P<sfx>_.+)?$")

# Table 13 (image) MIP reference row, keyed by "<task>_<v>_img"
MIP_REF_IMG = {}
# best: released ckpt filename means (mean over seeds; PART CDXLVI method)
# where artifacts exist; transport columns (idx 6,7) keep paper values
# (no released image artifacts for transport).
_T13 = {"sudeepdit": [1.00, 1.00, 0.99, 1.00, 0.83, 0.94, 0.50, 0.90,
                      0.49, 0.93],
        "chitransformer": [1.00, 1.00, 0.99, 0.99, 0.82, 0.92, 0.18,
                           0.86, 0.53, 0.92],
        "chiunet": [1.00, 1.00, 0.98, 1.00, 0.84, 0.90, 0.52, 0.96,
                    0.62, 0.94]}
_T13L5 = {"sudeepdit": [0.99, 1.00, 0.96, 0.98, 0.83, 0.92, 0.31, 0.84,
                        0.66, 0.87],
          "chitransformer": [0.98, 1.00, 0.91, 0.98, 0.21, 0.04, 0.06,
                             0.69, 0.48, 0.83],
          "chiunet": [1.00, 1.00, 0.95, 0.98, 0.84, 0.91, 0.37, 0.91,
                      0.50, 0.78]}
_COLS = ["lift_mh", "lift_ph", "can_mh", "can_ph", "square_mh",
         "square_ph", "transport_mh", "transport_ph", "tool_hang_ph",
         "pusht"]
for _n in _T13:
    for _c, _b, _l in zip(_COLS, _T13[_n], _T13L5[_n]):
        MIP_REF_IMG[(_c + "_img", _n)] = (_b, _l)


def ref_key(task):
    """Map a run's task-config name to the reference-table key."""
    task = re.sub(r"_(ht|hg|l2|mip|flow|sflow)?_?(optimization|task)[a-z0-9]+$", "", task)
    task = re.sub(r"_(mip|flow|sflow|l2|hg)$", "", task)
    if task.endswith("_state_delta_legacy"):
        return task[: -len("_state_delta_legacy")]
    if task.endswith("_state_abs"):
        return task[: -len("_state_abs")]
    for suf in ("_image_dl", "_image"):
        if task.endswith(suf):
            return task[: -len(suf)] + "_img"
    if task.startswith("pusht"):
        return "pusht_img"
    if task.startswith("kitchen"):
        return "kitchen_state"
    return task
SR_RE = re.compile(r"mean_success_1 - ([0-9.]+)")


def harvest():
    import json
    runs = {}
    for d in sorted(glob.glob("logs/t12_*")):
        m = NAME_RE.search(d)
        if not m:
            continue
        m2 = NAME_RE.search(d.split("/")[-1])
        sfx = (m2.group("sfx") or "") if m2 else ""
        srs = []
        try:
            for line in open(d + "/metrics.jsonl"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys = ([k for k in r if k.startswith("p4_")]
                        if "kitchen" in d else
                        [k for k in r if k.startswith("mean_success_")])
                if keys:
                    srs.append(float(r[sorted(keys)[0]]))
        except FileNotFoundError:
            continue
        if srs:
            runs[(m["task"] + sfx, m["net"], int(m["seed"]))] = srs
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo", default="yuchen0187/mip-ht-checkpoints")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--min_evals", type=int, default=15,
                    help="require full series before judging a run")
    cf = ap.parse_args()
    runs = harvest()
    cells = defaultdict(dict)
    for (task, net, seed), srs in runs.items():
        cells[(task, net)][seed] = srs
    qualify = []
    print(f"{'cell':38s} {'n':>2s} {'HT best/last5':>14s} "
          f"{'MIP ref':>12s} verdict")
    for (task, net), by_seed in sorted(cells.items()):
        full = {s: v for s, v in by_seed.items() if len(v) >= cf.min_evals}
        tag = f"{task} {net}"
        if not full:
            print(f"{tag:38s}  0  (in progress: "
                  f"{[len(v) for v in by_seed.values()]} evals)")
            continue
        best = float(np.mean([max(v) for v in full.values()]))
        l5 = float(np.mean([np.mean(v[-5:]) for v in full.values()]))
        rk = ref_key(task)
        ref = MIP_REF.get((rk, net)) or MIP_REF_IMG.get((rk, net))
        ok = ref and best >= ref[0]
        print(f"{tag:38s} {len(full):2d}  {best:.2f}/{l5:.2f}"
              f"{'':6s}{ref[0]:.2f}/{ref[1]:.2f}  "
              f"{'>=MIP UPLOAD' if ok else 'below'}" if ref else f"{tag} no-ref")
        if ok:
            qualify.append((task, net, sorted(full)))
    if not cf.upload:
        return
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(cf.repo, repo_type="model", exist_ok=True,
                private=not cf.public)
    for task, net, seeds in qualify:
        for s in seeds:
            p = f"logs/t12_{task}_{net}_s{s}/models/model_best.pt"
            if not os.path.exists(p):
                print(f"MISSING {p}")
                continue
            dest = f"{task}/{net}/seed{s}/model_best.pt"
            print(f"UPLOAD {p} -> {cf.repo}/{dest}", flush=True)
            api.upload_file(path_or_fileobj=p, path_in_repo=dest,
                            repo_id=cf.repo, repo_type="model")
    print("T12_UPLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
