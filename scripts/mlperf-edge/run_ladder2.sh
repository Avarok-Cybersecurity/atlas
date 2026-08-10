#!/bin/bash
set -u
cd /workspace/.wt-decode-fold
for spec in "3 k4new" "5 k6new" "7 k8new"; do
  set -- $spec
  echo "===== LADDER leg nd=$1 ($2) $(date +%H:%M) ====="
  ND=$1 TAG=$2 bash run_agentic_kab.sh 2>&1 | grep -aE 'KAB_RESULT|SERVE_DIED|no result|serve up' | tail -4
done
echo LADDER2_DONE
