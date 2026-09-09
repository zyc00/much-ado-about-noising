#!/usr/bin/env python3
"""Aggregate task-level RoboCasa-GR1 convergence evaluations.

The benchmark metric is the unweighted mean of 24 task success rates.  The
script intentionally refuses incomplete checkpoints in ``--strict`` mode.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


SUCCESS_RE = re.compile(r"success rate:\s+([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class CheckpointSpec:
    method: str
    steps_k: int
    patterns: tuple[str, ...]


SPECS = (
    CheckpointSpec("MSE", 2, ("evalout_mse2k_*.log",)),
    CheckpointSpec("MSE", 16, ("evalout_mse16k_*.log",)),
    CheckpointSpec("MSE", 32, ("evalout_mse32k_*.log",)),
    CheckpointSpec("MSE", 52, ("evalout_mse-52000_*.log",)),
    CheckpointSpec("MSE", 54, ("evalout_mse-54000_*.log",)),
    CheckpointSpec("MSE", 56, ("evalout_mse-56000_*.log",)),
    CheckpointSpec("MSE", 58, ("evalout_mse-58000_*.log",)),
    CheckpointSpec("MSE", 60, ("evalout_mse60k_*.log",)),
    CheckpointSpec("Flow", 2, ("evalout_flow2k_*.log",)),
    CheckpointSpec("Flow", 16, ("evalout_flow16k_*.log",)),
    CheckpointSpec("Flow", 50, ("evalout_flow50k_*.log",)),
    CheckpointSpec("Flow", 52, ("evalout_flow-52000_*.log",)),
    CheckpointSpec("Flow", 54, ("evalout_flow-54000_*.log",)),
    CheckpointSpec("Flow", 56, ("evalout_flow-56000_*.log",)),
    CheckpointSpec("Flow", 58, ("evalout_flow-58000_*.log",)),
    CheckpointSpec("Flow", 60, ("evalout_flow60k_*.log",)),
    CheckpointSpec("HT (nu=2)", 2, ("evalout_ht2k_*.log",)),
    CheckpointSpec("HT (nu=2)", 16, ("evalout_ht16k_*.log",)),
    CheckpointSpec("HT (nu=2)", 32, ("evalout_ht32k_*.log",)),
    CheckpointSpec("HT (nu=2)", 52, ("evalout_ht52k_*.log",)),
    CheckpointSpec("HT (nu=2)", 60, ("evalout_ht60k_*.log",)),
    CheckpointSpec(
        "HT (nu=4d)",
        52,
        (
            "evalout_gr1c2nu4d-52000_[0-7].log",
            "evalout_gr1c2nu4d-52000-retry_[0-7].log",
        ),
    ),
    CheckpointSpec(
        "HT (nu=4d)",
        54,
        (
            "evalout_gr1c2nu4d-54000_[0-7].log",
            "evalout_gr1c2nu4d-54000-retry_[0-7].log",
        ),
    ),
    CheckpointSpec(
        "HT (nu=4d)",
        56,
        (
            "evalout_gr1c2nu4d-56000_[0-7].log",
            "evalout_gr1c2nu4d-56000-retry_[0-7].log",
        ),
    ),
    CheckpointSpec("HT (nu=4d)", 58, ("evalout_gr1c2nu4d-58000_[0-7].log",)),
    CheckpointSpec("HT (nu=4d)", 60, ("evalout_gr1c260k_[0-7].log",)),
)


def values_for(log_root: Path, patterns: tuple[str, ...]) -> tuple[list[float], list[Path]]:
    files = sorted({path for pattern in patterns for path in log_root.glob(pattern)})
    values: list[float] = []
    for path in files:
        values.extend(float(value) for value in SUCCESS_RE.findall(path.read_text(errors="replace")))
    return values, files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = []
    incomplete = []
    for spec in SPECS:
        values, files = values_for(args.log_root, spec.patterns)
        if len(values) != 24:
            incomplete.append((spec.method, spec.steps_k, len(values)))
            if not values:
                continue
        rows.append(
            {
                "method": spec.method,
                "steps_k": spec.steps_k,
                "success_pct": 100.0 * sum(values) / len(values),
                "n_tasks": len(values),
                "n_logs": len(files),
                "patterns": ";".join(spec.patterns),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['method']:12s} {row['steps_k']:>2d}k: "
            f"{row['success_pct']:5.2f}% ({row['n_tasks']}/24 tasks)"
        )
    if incomplete:
        print("Incomplete checkpoints:")
        for method, steps_k, count in incomplete:
            print(f"  {method} {steps_k}k: {count}/24 tasks")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
