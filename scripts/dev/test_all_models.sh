#!/bin/bash
# Atlas comprehensive model test suite
# Usage: ./test_all_models.sh <model_id> <port> [extra_args...]
# Output: JSON lines to stdout

set -euo pipefail

MODEL="$1"
PORT="${2:-8888}"
shift 2
EXTRA_ARGS="$*"

IMAGE="atlas-gb10:latest"
CONTAINER="atlas-test-$$"
HF_CACHE="/workspace/.cache/huggingface/hub"

# Clean up on exit
cleanup() { sudo docker stop "$CONTAINER" 2>/dev/null; sudo docker rm "$CONTAINER" 2>/dev/null; }
trap cleanup EXIT

# Start container
sudo docker run -d --name "$CONTAINER" \
  --gpus all --ipc=host --network host \
  -v "$HF_CACHE:/root/.cache/huggingface/hub" \
  "$IMAGE" serve "$MODEL" --port "$PORT" --max-seq-len 16384 $EXTRA_ARGS >/dev/null 2>&1

# Wait for ready (max 5 min)
echo "Starting $MODEL on port $PORT..." >&2
READY=0
for i in $(seq 1 60); do
  if sudo docker logs "$CONTAINER" 2>&1 | grep -q "Listening on"; then
    READY=1
    break
  fi
  if sudo docker logs "$CONTAINER" 2>&1 | grep -qE "Error:|panic"; then
    echo "FAILED: $MODEL crashed during startup" >&2
    sudo docker logs "$CONTAINER" 2>&1 | grep -E "Error:|panic" | tail -3 >&2
    echo "{\"model\":\"$MODEL\",\"status\":\"CRASH\"}"
    exit 1
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "TIMEOUT: $MODEL did not start in 5 minutes" >&2
  echo "{\"model\":\"$MODEL\",\"status\":\"TIMEOUT\"}"
  exit 1
fi

echo "$MODEL ready, running tests..." >&2

# Helper: run a single test
run_test() {
  local test_name="$1"
  local prompt="$2"
  local max_tokens="${3:-150}"
  local expect_pattern="${4:-}"

  local result
  result=$(curl -s --max-time 60 "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}],
      \"max_tokens\": $max_tokens,
      \"temperature\": 0.0
    }" 2>/dev/null)

  if [ -z "$result" ]; then
    echo "{\"test\":\"$test_name\",\"status\":\"ERROR\",\"error\":\"no response\"}"
    return
  fi

  python3 -c "
import sys, json, re
r = json.loads('''$result''')
if 'choices' not in r:
    print(json.dumps({'test': '$test_name', 'status': 'ERROR', 'error': str(r)}))
    sys.exit()
c = r['choices'][0]
u = r['usage']
content = c['message']['content']
tok_s = u.get('response_token/s', 0)
ttft = u.get('time_to_first_token_ms', 0)
comp_tokens = u.get('completion_tokens', 0)
finish = c.get('finish_reason', '')
pattern = '''$expect_pattern'''
passed = bool(re.search(pattern, content, re.IGNORECASE)) if pattern else True
print(json.dumps({
    'test': '$test_name',
    'status': 'PASS' if passed else 'FAIL',
    'content': content[:200],
    'tokens': comp_tokens,
    'tok_s': round(tok_s, 1),
    'ttft_ms': round(ttft, 1),
    'finish': finish,
}))
" 2>/dev/null || echo "{\"test\":\"$test_name\",\"status\":\"PARSE_ERROR\"}"
}

# Run test suite
echo "{\"model\":\"$MODEL\",\"status\":\"TESTING\"}"

# 1. Capital of France (basic factual)
run_test "capital" "What is the capital of France?" 50 "Paris"

# 2. Fibonacci (math/sequence)
run_test "fibonacci" "Write the first 10 Fibonacci numbers separated by commas." 80 "0.*1.*1.*2.*3.*5.*8.*13.*21.*34"

# 3. Count to 20 (structured output)
run_test "count" "Count from 1 to 20, separated by commas." 100 "1.*2.*3.*4.*5.*6.*7.*8.*9.*10.*11.*12.*13.*14.*15.*16.*17.*18.*19.*20"

# 4. Coherence (multi-sentence)
run_test "coherence" "Explain what photosynthesis is in 2-3 sentences." 150 "sunlight|chloro|oxygen|carbon|glucose|plant"

# 5. Code generation
run_test "code" "Write a Python function called is_prime that returns True if a number is prime." 200 "def is_prime"

# 6. Multi-turn / tool readiness (structured response)
run_test "planets" "List all 8 planets in our solar system in order from the sun." 100 "Mercury.*Venus.*Earth.*Mars.*Jupiter.*Saturn.*Uranus.*Neptune"

echo "{\"model\":\"$MODEL\",\"status\":\"DONE\"}"
