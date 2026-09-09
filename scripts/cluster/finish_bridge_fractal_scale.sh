#!/usr/bin/env bash
# Finish the current figure after BOTH complete dataset probes; never plot a partial archive.
set -euo pipefail
probe_root=/mnt/pfs/yuchen/crossdataset_general_mse_20260907
project_root=/home/jigu/projects/much-ado-about-noising
cd "$project_root"
for attempt in $(seq 1 480); do
  if kubectl exec yuchen-vla-util -- bash -c 'test -f /mnt/pfs/yuchen/crossdataset_general_mse_20260907/bridge/rank00_done.json && test -f /mnt/pfs/yuchen/crossdataset_general_mse_20260907/bridge/rank01_done.json && test -f /mnt/pfs/yuchen/crossdataset_general_mse_20260907/fractal/rank00_done.json && test -f /mnt/pfs/yuchen/crossdataset_general_mse_20260907/fractal/rank01_done.json'; then
    kubectl exec yuchen-vla-util -- /mnt/pfs/yuchen/code/much-ado-about-noising/.venv/bin/python \
      "$probe_root/code/summarize_bridge_fractal_scale.py" --root "$probe_root" \
      --base "$probe_root/base_summary.json" --output "$probe_root/expanded_summary.json"
    kubectl cp "yuchen-vla-util:$probe_root/expanded_summary.json" \
      analysis/diagnostics/crossdataset_residual/expanded_summary.json
    CROSSDATASET_SUMMARY=analysis/diagnostics/crossdataset_residual/expanded_summary.json \
      CROSSDATASET_FIGURE_PREFIX=all_datasets_progress_lines \
      .venv/bin/python analysis/diagnostics/plot_crossdataset_progress_lines.py
    echo EXPANDED_DATASET_FIGURE_READY
    exit 0
  fi
  phase=$(kubectl get pod yuchen-bridge-fractal-scale-0907 -o jsonpath='{.status.phase}')
  if [[ "$phase" == Failed ]]; then
    kubectl logs yuchen-bridge-fractal-scale-0907 --tail=50
    echo 'Probe failed; existing figure preserved.' >&2
    exit 1
  fi
  sleep 30
done
echo 'Probe wait limit reached; no partial figure generated.' >&2
exit 2
