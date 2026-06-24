#!/bin/bash
# Slice the already-merged clean_20k_c into the 3 segments (merge already done).
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python; SRC=data/tool_hang_clean_20k_c.hdf5
$PY scripts/slice_segments.py --segment full       --src $SRC --out data/tool_hang_full2ins_20k_c.hdf5   --n 20000
$PY scripts/slice_segments.py --segment init2grasp --src $SRC --out data/tool_hang_init2grasp_20k_c.hdf5 --n 20000
$PY scripts/slice_segments.py --segment pick2ins   --src $SRC --out data/tool_hang_pick2ins_20k_c.hdf5   --n 20000
echo "SLICE_ONLY DONE"
