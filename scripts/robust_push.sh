#!/bin/bash
# Resume-push a single file to PVC, retrying through kubectl-exec stalls.
export KUBECONFIG=/home/jigu/.kube/config
SHIM="$HOME/.local/bin/kexec-rsh"; DEST=yuchen-dev:/mnt/pfs/yuchen/code/much-ado-about-noising
SRC="$1"; SUB="$2"
for i in $(seq 1 80); do
  rsync -a --partial --append-verify --timeout=90 --info=progress2 --blocking-io \
    --rsh="$SHIM" "$SRC" "$DEST/$SUB/" && { echo "PUSH_DONE $SRC"; exit 0; }
  echo "[retry $i] stalled, resuming..."; sleep 4
done
echo "PUSH_GAVEUP $SRC"
