#!/bin/bash
# Same-box TTFT A/B for the snapshot fold. Serves one binary at a time (GB10 unified
# memory cannot host two serves at util 0.70), runs Gate-C2 smoke + the warm multi-turn
# probe, tears down, then the other. Everything else is the frozen c2final golden config.
#
# Usage: ab_ttft_probe.sh <tag> <binary> [turns]
set -u
TAG="$1"; BIN="$2"; TURNS="${3:-14}"
OUT="/workspace/.wt-golden/ab_${TAG}"
mkdir -p "$OUT"

sudo docker rm -f atlas-ab-probe >/dev/null 2>&1; sleep 3
sudo docker run -d --name atlas-ab-probe --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$BIN:/usr/local/bin/spark:ro" \
  atlas-gb10:followups serve centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --host 0.0.0.0 --port 8888 --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
  --speculative --num-drafts 3 --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1

for i in $(seq 1 150); do curl -sf -m4 http://localhost:8888/v1/models 2>/dev/null | grep -q Qwen && break
  sudo docker ps --format '{{.Names}}' | grep -q atlas-ab-probe || { echo "SERVE_DIED tag=$TAG"; exit 1; }; sleep 5; done
echo "=== [$TAG] serve up (bin=$BIN) ==="

# --- Gate C2: coherence + tool call (must pass BEFORE any timing is believed) ---
echo "--- [$TAG] gate C2 coherence ---"
curl -sf -m60 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf","messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number iteratively."}],"max_tokens":180,"temperature":0,"seed":42}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' | tee "$OUT/coherence.txt"
echo "--- [$TAG] gate C2 tool call ---"
curl -sf -m60 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf","messages":[{"role":"user","content":"What is the weather in Paris? Use the get_weather tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"max_tokens":120,"temperature":0,"seed":42}' \
  | tee "$OUT/toolcall.json" | head -c 600; echo

# --- warm multi-turn TTFT/TPOT probe (the regime the fold targets) ---
echo "--- [$TAG] warm probe (${TURNS} turns) ---"
python3 $(dirname "$0")/warm_probe.py 8888 "$TAG" "$OUT/probe.json" --turns "$TURNS" --maxtok 300

sudo docker rm -f atlas-ab-probe >/dev/null 2>&1
echo "AB_PROBE_DONE tag=$TAG out=$OUT"
