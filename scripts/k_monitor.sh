#!/bin/bash
# Poll the dagger-mip k8s pipeline until the next milestone or failure, then exit.
export KUBECONFIG=/home/jigu/.kube/config
for i in $(seq 1 200); do
  st=$(kubectl get pod dagger-mip -o jsonpath='{.status.phase}' 2>/dev/null)
  log=$(kubectl logs --tail=400 dagger-mip 2>/dev/null | tr '\r' '\n')
  done_m=$(echo "$log" | grep -aoE "grasp_mip READY|handoffins_mip COLLECTED|handoffins_mip TRAINED|MIP_STITCH_dagger.*%|DAGGER MIP DONE" | tail -1)
  if [ "$st" = "Failed" ] || [ "$st" = "Succeeded" ]; then
    echo "POD_$st"; echo "$log" | grep -aiE "MIP_STITCH_dagger|error|Traceback|SSLError|FAILED" | tail -6; exit 0
  fi
  if [ -n "$done_m" ]; then
    step=$(echo "$log" | grep -aoE "\[Step [0-9]+\]" | tail -1)
    echo "MILESTONE: $done_m | last $step | pod=$st"; exit 0
  fi
  sleep 120
done
echo "MONITOR_TIMEOUT"
