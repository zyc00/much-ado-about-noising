#!/bin/bash
set -eu
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
until [ $(ls data/wpmatch_shard_*.hdf5 2>/dev/null | wc -l) -ge 8 ]; do sleep 120; done
sleep 60
SH=""; for k in $(seq 0 7); do SH="$SH data/wpmatch_shard_$k.hdf5"; done
python scripts/merge_hdf5.py --out data/tool_hang_wpmatch_clean_2000.hdf5 $SH
python scripts/slice_segments.py --segment full --src data/tool_hang_wpmatch_clean_2000.hdf5 --out data/tool_hang_full2ins_wpmatch_2000.hdf5 --n 2000
echo MATCHSLICE-DONE
