#!/bin/bash
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
U=$(python -c "from huggingface_hub import hf_hub_url; print(hf_hub_url('ChaoyiPan/mip-dataset','robomimic/lift/mh/image_abs.hdf5',repo_type='dataset'))")
PX="http://172.17.0.12:2080"
O=$(curl -skL -x "$PX" -m 35 -o /dev/null -w "code=%{http_code} bytes=%{size_download} speed=%{speed_download}" "$U" 2>&1)
echo "CDN30s: $O"
