#!/bin/bash
# One leg of the ATLAS_FP4_PREFILL A/B. Serves the frozen golden config, optionally
# with W4A4 dense-FFN prefill enabled, runs the Gate C2 smoke and the cold-prefill
# probe, then tears down. GB10 unified memory cannot host two serves at util 0.70,
# so the legs run sequentially against the same box.
#
# Usage: run_prefill_w4a4_ab.sh <tag> <binary> <fp4:0|1>
set -u
TAG="$1"; BIN="$2"; FP4="$3"
OUT="/workspace/.wt-golden/ab_prefill_${TAG}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

FP4_ENV=()
# Presence-checked flag: setting it to 0 would still ENABLE it, so the off leg
# must not pass it at all.
[ "$FP4" = "1" ] && FP4_ENV=(-e ATLAS_FP4_PREFILL=1)

sudo docker rm -f atlas-fp4-ab >/dev/null 2>&1; sleep 3
sudo docker run -d --name atlas-fp4-ab --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 "${FP4_ENV[@]}" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$BIN:/usr/local/bin/spark:ro" \
  atlas-gb10:followups serve centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --host 0.0.0.0 --port 8888 --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
  --speculative --num-drafts 3 --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1

for i in $(seq 1 180); do curl -sf -m4 http://localhost:8888/v1/models 2>/dev/null | grep -q Qwen && break
  sudo docker ps --format '{{.Names}}' | grep -q atlas-fp4-ab || { echo "SERVE_DIED tag=$TAG"; exit 1; }; sleep 5; done
echo "=== [$TAG] serve up (ATLAS_FP4_PREFILL=$FP4) ==="
# Confirm the flag actually took effect rather than assuming it did.
sudo docker logs atlas-fp4-ab 2>&1 | grep -c 'ATLAS_FP4_PREFILL=1' | xargs -I{} echo "[$TAG] fp4-prefill banner lines: {}"

echo "--- [$TAG] gate C2 coherence ---"
curl -sf -m60 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf","messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number iteratively."}],"max_tokens":180,"temperature":0,"seed":42}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' | tee "$OUT/coherence.txt"
echo "--- [$TAG] gate C2 tool call ---"
curl -sf -m60 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf","messages":[{"role":"user","content":"What is the weather in Paris? Use the get_weather tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"max_tokens":120,"temperature":0,"seed":42}' \
  | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin)["choices"][0]["message"].get("tool_calls")))' | tee "$OUT/toolcall.json"

echo "--- [$TAG] cold-prefill probe ---"
python3 -u "$HERE/prefill_w4a4_ab.py" 8888 "$TAG" "$OUT/probe.json"

sudo docker rm -f atlas-fp4-ab >/dev/null 2>&1
echo "PREFILL_AB_DONE tag=$TAG out=$OUT"
