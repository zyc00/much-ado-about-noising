#!/usr/bin/env python3
"""Re-test the executed four-step prefix in the uniform WidowX sample."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_widowx_mode_offsets import density_geometry  # noqa: E402
from probe_widowx_multimodality import (  # noqa: E402
    GROUP_COLUMNS,
    bh_qvalues,
    empirical_p,
    fit_gmm2,
    group_view,
)


def main() -> None:
    metadata = json.loads((HERE / "rollout_random_v1.json").read_text())
    with np.load(HERE / "rollout_random_v1.npz") as archive:
        discovery = archive["discovery"][:, :, :4].astype(np.float64)
        confirmation = archive["confirmation"][:, :, :4].astype(np.float64)
        uids = archive["deep_uid"].astype(str)
        null = archive["null_delta_bic"].astype(np.float64)

    rows = []
    for index, uid in enumerate(uids):
        tests = {}
        for group, columns in GROUP_COLUMNS.items():
            gd = group_view(discovery[index], columns)
            gc = group_view(confirmation[index], columns)
            center = gd.mean(axis=0, keepdims=True)
            _, singular, vh = np.linalg.svd(gd - center, full_matrices=False)
            axis = vh[0] if len(singular) and singular[0] >= 1e-12 else np.zeros(gd.shape[1])
            projected = (gc - center) @ axis
            fit = fit_gmm2(projected)
            labels = fit.pop("labels")
            n_modes, valley = density_geometry(fit, projected)
            if labels.all() or (~labels.astype(bool)).all():
                delta = np.zeros_like(confirmation[index, 0])
            else:
                delta = (
                    confirmation[index][labels == 1].mean(axis=0)
                    - confirmation[index][labels == 0].mean(axis=0)
                )
            energy = np.sum(delta * delta, axis=0)
            total = max(float(energy.sum()), 1e-18)
            tests[group] = {
                "delta_bic": fit["delta_bic"],
                "sep": fit["sep"],
                "min_weight": fit["min_weight"],
                "p": empirical_p(fit["delta_bic"], null),
                "density_modes": int(n_modes),
                "valley_depth": float(valley),
                "energy_translation": float(energy[:3].sum() / total),
                "energy_rotation": float(energy[3:6].sum() / total),
                "energy_gripper": float(energy[6] / total),
            }
        rows.append({"uid": uid, "tests": tests})

    summary = {}
    for group in GROUP_COLUMNS:
        qvalues = bh_qvalues(np.asarray([row["tests"][group]["p"] for row in rows]))
        detected, resolved = [], []
        for row, qvalue in zip(rows, qvalues):
            test = row["tests"][group]
            test["q"] = float(qvalue)
            test["detected"] = bool(
                qvalue <= 0.05
                and test["delta_bic"] >= 10.0
                and test["sep"] >= 2.0
                and test["min_weight"] >= 0.2
            )
            test["resolved"] = bool(
                test["detected"]
                and test["density_modes"] >= 2
                and test["valley_depth"] >= 0.25
            )
            if test["detected"]:
                detected.append(row["uid"])
            if test["resolved"]:
                resolved.append(row["uid"])
        summary[group] = {
            "n_detected": len(detected),
            "n_resolved": len(resolved),
            "detected_uids": detected,
            "resolved_uids": resolved,
        }

    result = {
        "source": metadata["source"],
        "checkpoint": metadata["checkpoint"],
        "n_states": len(rows),
        "predicted_chunk_steps": 8,
        "tested_prefix_steps": 4,
        "reason": "evaluation executes four of the eight predicted actions before replanning",
        "axis": "PC1 fit on prefix of fresh discovery samples only",
        "test": "1G versus 2G BIC on independent prefix confirmation projections",
        "multiple_testing": "BH q<=.05 separately by action group",
        "summary": summary,
        "rows": rows,
    }
    output = HERE / "rollout_random_prefix4_v1.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
