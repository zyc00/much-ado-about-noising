#!/bin/bash
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
U=$(python -c "from huggingface_hub import hf_hub_url; print(hf_hub_url('ChaoyiPan/mip-dataset','robomimic/lift/mh/image_abs.hdf5',repo_type='dataset'))")
echo "reach:"; timeout 5 bash -c "echo > /dev/tcp/172.17.0.12/2080" && echo TCP_OK || echo TCP_FAIL
for PX in "http://172.17.0.12:2080" "https://172.17.0.12:2080"; do
  S=$(timeout 40 curl -skL -x "$PX" --proxy-insecure -o /dev/null -w "%{speed_download}" "$U" 2>/dev/null)
  echo "PX $PX -> ${S:-fail} B/s"
done
S=$(timeout 30 curl -skL -x "http://172.17.0.12:2080" -o /dev/null -w "%{http_code}" "https://huggingface.co" 2>/dev/null)
echo "hf.co via http-proxy: code ${S:-fail}"
