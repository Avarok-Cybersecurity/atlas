#!/bin/bash
# One leg of the --ssm-cache-slots sweep. Everything is the frozen golden config
# except the slot count. Runs the multi-session eviction probe, which is the only
# instrument that can exercise slot pressure (a single conversation occupies one
# slot and never evicts anything).
#
# Usage: run_ssm_slots_ab.sh <slots> <binary> [sessions] [turns]
set -u
SLOTS="$1"; BIN="$2"; SESSIONS="${3:-24}"; TURNS="${4:-6}"
OUT="/workspace/.wt-golden/ab_slots_${SLOTS}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

sudo docker rm -f atlas-slots-ab >/dev/null 2>&1; sleep 3
sudo docker run -d --name atlas-slots-ab --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$BIN:/usr/local/bin/spark:ro" \
  atlas-gb10:followups serve centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --host 0.0.0.0 --port 8888 --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --ssm-cache-slots "$SLOTS" --ssm-checkpoint-interval 32 \
  --speculative --num-drafts 3 --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1

for i in $(seq 1 180); do curl -sf -m4 http://localhost:8888/v1/models 2>/dev/null | grep -q Qwen && break
  sudo docker ps --format '{{.Names}}' | grep -q atlas-slots-ab || { echo "SERVE_DIED slots=$SLOTS"; exit 1; }; sleep 5; done
echo "=== [slots=$SLOTS] serve up (sessions=$SESSIONS turns=$TURNS) ==="
# Confirm the slot count took effect AND record what it cost. Slots are ~151.5 MB
# each and come straight out of the KV budget, so the pool size and the resulting
# KV allocation must be captured together -- otherwise a later comparison can only
# infer the tradeoff instead of measuring it. Saved to disk: the container is
# removed at the end of the leg and the log goes with it.
sudo docker logs atlas-slots-ab 2>&1 \
  | grep -aiE 'Marconi [0-9]+ slots|KV cache: [0-9.]+ GB total|KV budget self-relative|Weights: ' \
  | tee "$OUT/alloc.txt"

python3 -u "$HERE/ssm_slots_probe.py" 8888 "slots${SLOTS}" "$OUT/probe.json" \
  --sessions "$SESSIONS" --turns "$TURNS"

sudo docker rm -f atlas-slots-ab >/dev/null 2>&1
echo "SLOTS_AB_DONE slots=$SLOTS out=$OUT"
