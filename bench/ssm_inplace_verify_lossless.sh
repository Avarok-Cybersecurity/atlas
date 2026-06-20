#!/usr/bin/env bash
# Lossless A/B harness for the STree-style in-place SSM verify commit.
#
# Serves Qwen3.6-35B-A3B-FP8 with MTP ON (--force-speculative) at temp=0 and
# captures greedy token streams for N prompts (incl. tool-call prompts). Run
# once with ATLAS_SSM_INPLACE_VERIFY unset (baseline) and once with =1; diff
# the two output JSONs. Identical => byte-identical losslessness.
#
# Usage:
#   bench/ssm_inplace_verify_lossless.sh <tag> [inplace0|inplace1]
# Produces /tmp/lossless_<tag>.json
set -euo pipefail

TAG="${1:?tag required}"
MODE="${2:-inplace0}"
PORT="${PORT:-8891}"
SPARK="${SPARK:-/workspace/atlas-mtp/target/release/spark}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
OUT="/tmp/lossless_${TAG}.json"
LOG="/tmp/serve_${TAG}.log"

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/workspace/.cache/huggingface
if [[ "$MODE" == "inplace1" ]]; then
  export ATLAS_SSM_INPLACE_VERIFY=1
else
  unset ATLAS_SSM_INPLACE_VERIFY || true
fi

echo "[$TAG] starting server (MODE=$MODE) ..."
nohup "$SPARK" serve "$MODEL" \
  --port "$PORT" \
  --max-seq-len 16384 \
  --speculative --force-speculative \
  --mtp-quantization bf16 \
  > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

# Wait for readiness (model load ~minutes).
for i in $(seq 1 240); do
  if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[$TAG] server ready after ${i}s"; break
  fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "[$TAG] server died; tail log:"; tail -30 "$LOG"; exit 1
  fi
  sleep 2
done

# Prompts: prose, reasoning, code, and tool-call (functions) to exercise
# the grammar/verify pipeline. 12 total (>= 10).
PROMPTS_FILE=$(mktemp)
cat > "$PROMPTS_FILE" <<'JSON'
[
 {"m":"What is the capital of France? Answer in one sentence."},
 {"m":"Explain Rayleigh scattering in two sentences."},
 {"m":"Write a Rust function that returns the nth Fibonacci number."},
 {"m":"List the first 8 prime numbers."},
 {"m":"Summarize the plot of Romeo and Juliet in three sentences."},
 {"m":"What is 17 * 23? Show your reasoning."},
 {"m":"Translate 'good morning, how are you?' to Spanish."},
 {"m":"Give three tips for writing clean code."},
 {"m":"Describe the water cycle briefly."},
 {"m":"Write a haiku about autumn leaves."},
 {"m":"What is the weather in Paris?","tool":1},
 {"m":"Book a table for two at 7pm tonight.","tool":1}
]
JSON

TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]}}},{"type":"function","function":{"name":"book_table","description":"Book a restaurant table","parameters":{"type":"object","properties":{"people":{"type":"integer"},"time":{"type":"string"}},"required":["people","time"]}}}]'

echo "[" > "$OUT"
N=$(python3 -c "import json,sys;print(len(json.load(open('$PROMPTS_FILE'))))")
for idx in $(seq 0 $((N-1))); do
  M=$(python3 -c "import json;print(json.load(open('$PROMPTS_FILE'))[$idx]['m'])")
  HASTOOL=$(python3 -c "import json;print(json.load(open('$PROMPTS_FILE'))[$idx].get('tool',0))")
  if [[ "$HASTOOL" == "1" ]]; then
    BODY=$(python3 -c "
import json
print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':'''$M'''}],'tools':json.loads('''$TOOLS'''),'temperature':0,'max_tokens':200,'seed':0}))")
  else
    BODY=$(python3 -c "
import json
print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':'''$M'''}],'temperature':0,'max_tokens':200,'seed':0}))")
  fi
  RESP=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -d "$BODY")
  # Extract content + tool_calls deterministically.
  EXTRACT=$(python3 -c "
import json,sys
r=json.loads('''$RESP''')
ch=r.get('choices',[{}])[0].get('message',{})
out={'idx':$idx,'content':ch.get('content'),'tool_calls':ch.get('tool_calls')}
print(json.dumps(out,sort_keys=True))
" 2>/dev/null || echo "{\"idx\":$idx,\"error\":true}")
  echo "  $EXTRACT," >> "$OUT"
  echo "[$TAG] prompt $idx done"
done
echo "  null" >> "$OUT"
echo "]" >> "$OUT"

echo "[$TAG] wrote $OUT"
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
