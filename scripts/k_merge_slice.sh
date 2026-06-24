#!/bin/bash
# Runs INSIDE krun pod (CPU ok). Merge 20 collection shards -> clean_20k_c,
# then slice into full2ins / init2grasp / pick2ins (cluster-generated 20k data).
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python
SH=""; for k in $(seq 0 19); do SH="$SH data/clean_shard_$k.hdf5"; done
$PY scripts/merge_hdf5.py --out data/tool_hang_clean_20k_c.hdf5 $SH
echo "=== MERGED clean_20k_c ==="
$PY scripts/slice_segments.py --segment full       --src data/tool_hang_clean_20k_c.hdf5 --out data/tool_hang_full2ins_20k_c.hdf5   --n 20000
$PY scripts/slice_segments.py --segment init2grasp --src data/tool_hang_clean_20k_c.hdf5 --out data/tool_hang_init2grasp_20k_c.hdf5 --n 20000
$PY scripts/slice_segments.py --segment pick2ins   --src data/tool_hang_clean_20k_c.hdf5 --out data/tool_hang_pick2ins_20k_c.hdf5   --n 20000
echo "MERGE_SLICE DONE"
