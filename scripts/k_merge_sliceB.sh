#!/bin/bash
# Merge 20 script-B shards -> clean_20kB, then slice into the 3 segments.
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python
SH=""; for k in $(seq 0 19); do SH="$SH data/clean_shardB_$k.hdf5"; done
$PY scripts/merge_hdf5.py --out data/tool_hang_clean_20kB.hdf5 $SH
echo "=== MERGED clean_20kB ==="
SRC=data/tool_hang_clean_20kB.hdf5
$PY scripts/slice_segments.py --segment full       --src $SRC --out data/tool_hang_full2ins_20kB.hdf5   --n 20000
$PY scripts/slice_segments.py --segment init2grasp --src $SRC --out data/tool_hang_init2grasp_20kB.hdf5 --n 20000
$PY scripts/slice_segments.py --segment pick2ins   --src $SRC --out data/tool_hang_pick2ins_20kB.hdf5   --n 20000
echo "MERGE_SLICEB DONE"
