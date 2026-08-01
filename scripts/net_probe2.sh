#!/bin/bash
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
U=$(python -c "from huggingface_hub import hf_hub_url; print(hf_hub_url('ChaoyiPan/mip-dataset','robomimic/lift/ph/image_abs.hdf5',repo_type='dataset'))")
PX="http://172.17.0.12:2080"
for TAG in direct proxy; do
  if [ $TAG = proxy ]; then EX="-x $PX"; else EX=""; fi
  S=$(timeout 20 curl -skL $EX -o /dev/null -w "%{speed_download}" "$U" 2>/dev/null)
  echo "HF_$TAG ${S%.*} B/s"
done
S=$(timeout 20 curl -sk -o /dev/null -w "%{speed_download}" "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/dataset_info.json" 2>/dev/null)
echo "BERKELEY_direct ${S%.*} B/s"
