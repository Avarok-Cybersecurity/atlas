#!/bin/bash
# K ladder on the chain-widened binary: K=3 control, then 4, 6, 8.
set -u
cd /workspace/.wt-decode-fold
for spec in "2 k3new" "3 k4new" "5 k6new" "7 k8new"; do
  set -- $spec
  echo "===== LADDER leg nd=$1 ($2) $(date +%H:%M) ====="
  ND=$1 TAG=$2 bash run_agentic_kab.sh 2>&1 | grep -aE 'KAB_RESULT|SERVE_DIED|no result|serve up|summary: ' | tail -5
done
echo "===== LADDER COMPLETE ====="
for t in k3new k4new k6new k8new; do
  [ -f "kn_ab_$t.json" ] && python3 -c "import json;d=json.load(open('kn_ab_$t.json'));print(f\"  {d['tag']:6} nd={d['nd']}: TPOT {d['tpot_med_ms']:6.2f}ms  wall {d['wall_s']:7.1f}s  TTFT {d['ttft_med_ms']:6.0f}ms  tps {d['tps']}\")"
done
echo LADDER_DONE
