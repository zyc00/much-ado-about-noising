#!/bin/bash
for SVC in https://api.ipify.org https://ident.me; do
  D=$(timeout 20 curl -s --noproxy '*' "$SVC" 2>/dev/null)
  [ -n "$D" ] && { echo "DIRECT_EGRESS $D"; break; }
done
for SVC in https://api.ipify.org https://ident.me; do
  P=$(timeout 20 curl -s -x http://172.17.0.12:2080 "$SVC" 2>/dev/null)
  [ -n "$P" ] && { echo "PROXY_EGRESS $P"; break; }
done
echo PROBE_EXIT
