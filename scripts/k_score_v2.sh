#!/bin/bash
# Score mpv2 arms (own-dataset anchors/normalizer) + the dcw arm. "V2" lines.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
DV2=data/tool_hang_full2ins_mpv2_200.hdf5
DV1=data/tool_hang_full2ins_mp_200.hdf5

echo "V2 start l2v2"
python -u scripts/eval_twofactor.py --ckpt logs/mpv2_l2_s1000/models/snap_300000.pt \
  --loss regression --AS 8 --dataset $DV2 --tag v2_l2 --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/V2 /"
echo "V2 start dcw"
python -u scripts/eval_twofactor.py --ckpt logs/mp200_dcw_s1000/models/snap_300000.pt \
  --loss regression_dcw --AS 8 --dataset $DV1 --tag v2_dcw --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/V2 /"
# mip-v2 last (waits for snapshot)
for i in $(seq 1 60); do
  [ -f logs/mpv2_mip_s1000/models/snap_300000.pt ] && break
  sleep 120
done
echo "V2 start mipv2"
python -u scripts/eval_twofactor.py --ckpt logs/mpv2_mip_s1000/models/snap_300000.pt \
  --loss mip --AS 8 --dataset $DV2 --tag v2_mip --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/V2 /"
echo "V2 done"
