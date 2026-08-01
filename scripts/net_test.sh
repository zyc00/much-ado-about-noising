#!/bin/bash
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
U=$(python -c "from huggingface_hub import hf_hub_url; print(hf_hub_url('ChaoyiPan/mip-dataset','robomimic/lift/mh/image_abs.hdf5',repo_type='dataset'))")
PX="https://172.17.0.12:2080"
echo "test proxy $PX"
S=$(timeout 40 env HTTPS_PROXY=$PX https_proxy=$PX HTTP_PROXY=$PX http_proxy=$PX curl -skL -o /dev/null -w "%{speed_download}" "$U" 2>/dev/null)
echo "USERPROXY ${S:-fail} B/s"
S2=$(timeout 40 curl -skL -x "$PX" --proxy-insecure -o /dev/null -w "%{speed_download}" "$U" 2>/dev/null)
echo "USERPROXY_X ${S2:-fail} B/s"
