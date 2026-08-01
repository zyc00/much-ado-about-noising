#!/bin/bash
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
U=$(python -c "from huggingface_hub import hf_hub_url; print(hf_hub_url('ChaoyiPan/mip-dataset','robomimic/lift/ph/image_abs.hdf5',repo_type='dataset'))")
for PORT in 208 2080; do
  timeout 5 bash -c "echo > /dev/tcp/172.17.0.12/$PORT" 2>/dev/null && R=TCP_OK || R=TCP_FAIL
  S=$(timeout 25 curl -skL -x "http://172.17.0.12:$PORT" -o /dev/null -w "%{speed_download}" "$U" 2>/dev/null)
  echo "PORT $PORT $R speed ${S%.*} B/s"
done
