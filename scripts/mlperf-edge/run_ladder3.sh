#!/bin/bash
set -u
cd /workspace/.wt-decode-fold
for spec in "4 k5wyn" "5 k6wyn"; do
  set -- $spec
  echo "===== leg nd=$1 ($2) $(date +%H:%M) ====="
  ND=$1 TAG=$2 bash run_agentic_kab.sh 2>&1 | grep -aE 'KAB_RESULT|SERVE_DIED|no result|serve up' | tail -3
done
echo LADDER3_DONE
