"""Assemble final Table 12 (state) and Table 13 (vision) from the T12/T13
campaigns: per (column, backbone) pick our best completed HT config
(3-seed mean best SR, tie-break last-5), print ours vs MIP-artifact refs.

Run ON A POD: python scripts/final_tables.py
"""
import re

import numpy as np

from t12_harvest_upload import MIP_REF, MIP_REF_IMG, harvest, ref_key

T12_COLS = ["lift_ph", "lift_mh", "can_ph", "can_mh", "square_ph",
            "square_mh", "transport_ph", "transport_mh", "tool_hang_ph",
            "kitchen_state"]
T13_COLS = ["lift_ph_img", "lift_mh_img", "can_ph_img", "can_mh_img",
            "square_ph_img", "square_mh_img", "transport_ph_img",
            "transport_mh_img", "tool_hang_ph_img", "pusht_img"]
NETS = ["chiunet", "chitransformer", "sudeepdit"]
NLAB = {"chiunet": "CNN (chiunet)", "chitransformer": "Tf (chitransf)",
        "sudeepdit": "DiT (sudeep)"}


def is_ours(task_cfg):
    """HT-family runs only; exclude other losses and the dp_harness
    protocol arms (different eval cadence/episode count)."""
    if re.search(r"_(mip|flow|sflow|l2|hg|htlnu[c2]*|xm)(_|$)", task_cfg):
        return False
    if "dpharness" in task_cfg or "smin" in task_cfg:
        return False
    return True


def collect(runs):
    by = {}
    for (task_cfg, net, seed), srs in runs.items():
        if not is_ours(task_cfg):
            continue
        if len(srs) < 14:          # completed 300k runs only (15 evals)
            continue
        by.setdefault((task_cfg, net), {})[seed] = (
            max(srs), float(np.mean(srs[-5:])))
    cells = {}
    for (task_cfg, net), seeds in by.items():
        if len(seeds) < 3:
            continue
        vals = list(seeds.values())
        b = float(np.mean([v[0] for v in vals]))
        l5 = float(np.mean([v[1] for v in vals]))
        key = (ref_key(task_cfg), net)
        if key not in cells or (b, l5) > cells[key][:2]:
            cells[key] = (b, l5, task_cfg, len(vals))
    return cells


def show(title, cols, cells, ref):
    print(f"\n## {title}\n")
    short = [c.replace("_img", "").replace("_state", "").replace("_ph", "-ph")
              .replace("_mh", "-mh") for c in cols]
    hdr = "| method | " + " | ".join(short) + " |"
    print(hdr)
    print("|" + "---|" * (len(cols) + 1))
    for net in NETS:
        for label, src in (("MIP", "ref"), ("HT (ours)", "ours")):
            row = []
            for c in cols:
                if src == "ref":
                    v = ref.get((c, net))
                else:
                    v = cells.get((c, net))
                row.append(f"{v[0]:.2f}/{v[1]:.2f}" if v else "--")
            print(f"| {NLAB[net]} {label} | " + " | ".join(row) + " |")
    # column-SOTA: our best backbone vs their best backbone
    wins = []
    for c in cols:
        ours = max((cells[(c, n)][:2] for n in NETS if (c, n) in cells),
                   default=None)
        thrs = max((ref[(c, n)] for n in NETS if (c, n) in ref),
                   default=None)
        if ours and thrs:
            wb = ours[0] >= thrs[0] - 1e-9
            wl = ours[1] >= thrs[1] - 1e-9
            wins.append((c, ours, thrs, wb, wl))
    print("\ncolumn best-of-backbones (ours vs MIP, best/last5):")
    nb = nl = 0
    for c, o, t, wb, wl in wins:
        nb += wb
        nl += wl
        print(f"  {c:18s} ours {o[0]:.2f}/{o[1]:.2f}  MIP {t[0]:.2f}/{t[1]:.2f}"
              f"  {'BEST' if wb else '    '} {'L5' if wl else ''}")
    print(f"  => SOTA-or-tie: best {nb}/{len(wins)}, last5 {nl}/{len(wins)}")


runs = harvest()
cells = collect(runs)
show("Table 12 -- state", T12_COLS, cells, MIP_REF)
show("Table 13 -- vision", T13_COLS, cells, MIP_REF_IMG)
