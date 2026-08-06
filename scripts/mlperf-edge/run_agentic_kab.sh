#!/bin/bash
# Agentic-workload K A/B: serve at --num-drafts ND, run the gate-C subset
# (3 trajectories, ~174 turns, warm-dominant), extract wall/TTFT/TPOT.
# Usage: ND=3 TAG=k4 [BIN=...] bash run_agentic_kab.sh
set -u
ND="${ND:?}"; TAG="${TAG:?}"
BIN="${BIN:-/workspace/.wt-decode-fold/target/release/spark}"
CN="atlas-knab"
cd /workspace/endpoints-fresh
TS=$(date +%H%M%S); RD="results/kab_${TAG}_${TS}"
python3 - "$RD" <<'PY'
import sys, yaml
c = yaml.safe_load(open("/workspace/.wt-decode-fold/kab_template.yaml"))
c["report_dir"] = sys.argv[1]
yaml.safe_dump(c, open("/workspace/.wt-decode-fold/kab.yaml", "w"), sort_keys=False)
PY
sudo docker rm -f "$CN" >/dev/null 2>&1; sleep 3
sudo docker run -d --name "$CN" --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$BIN:/usr/local/bin/spark:ro" \
  atlas-gb10:followups serve centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --host 0.0.0.0 --port 8888 --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
  --speculative --num-drafts "$ND" --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
for i in $(seq 1 120); do curl -sf -m4 http://localhost:8888/v1/models 2>/dev/null | grep -q Qwen && break
  sudo docker ps --format '{{.Names}}' | grep -q "$CN" || { echo "[$TAG] SERVE_DIED"; exit 1; }; sleep 5; done
echo "[$TAG] serve up (nd=$ND); running gate-C subset..."
./.venv/bin/inference-endpoint benchmark from-config -c /workspace/.wt-decode-fold/kab.yaml --mode perf -v > "/workspace/kab_${TAG}.log" 2>&1
rc=$?
f=$(find "$RD" -name result_summary.json 2>/dev/null | head -1)
[ -z "$f" ] && f=$(find "$RD/run" -name result_summary.json 2>/dev/null | head -1)
if [ -n "$f" ]; then
  python3 -c "
import json;r=json.load(open('$f'))
out={'tag':'$TAG','nd':$ND,'wall_s':round(r['duration_ns']/1e9,1),'tpot_med_ms':round(r['tpot']['median']/1e6,2),'ttft_med_ms':round(r['ttft']['median']/1e6,1),'tps':round(r['tps'],2),'n':r['n_samples_completed']}
print('KAB_RESULT',json.dumps(out))
import pathlib;pathlib.Path('/workspace/.wt-decode-fold/kn_ab_${TAG}.json').write_text(json.dumps(out))
"
else echo "[$TAG] no result_summary (rc=$rc)"; tail -5 "/workspace/kab_${TAG}.log"; fi
# accept stats from serve
sudo docker logs "$CN" 2>&1 | grep -aE 'summary: |accepted=' | tail -3
sudo docker rm -f "$CN" >/dev/null 2>&1
