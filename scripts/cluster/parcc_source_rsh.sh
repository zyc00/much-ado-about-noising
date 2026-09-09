#!/usr/bin/env bash
# rsync remote-shell adapter: no keys/credentials copied to either cluster.
set -euo pipefail
test "${1:?missing host}" = pfs-source
shift
exec kubectl exec -i yuchen-vla-util -- "$@"
