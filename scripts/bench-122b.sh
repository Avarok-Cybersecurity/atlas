#!/usr/bin/env bash
#
# Structured benchmark for Qwen3.5-122B-A10B-NVFP4 on single GB10.
#
# Tests: correctness, throughput (single), throughput (concurrent)
# Baseline comparison: community docker INT4 = 26 tok/s single, 70+ concurrent

set -euo pipefail

API="http://localhost:8888/v1"
MODEL="Sehyo/Qwen3.5-122B-A10B-NVFP4"

echo "=========================================="
echo "  Atlas 122B-NVFP4 Benchmark Suite"
echo "=========================================="
echo ""

# ── 0. Verify server is up ──
echo "--- Test 0: Server Health ---"
MODELS=$(curl -s "$API/models" 2>/dev/null || echo "FAIL")
if echo "$MODELS" | grep -q "$MODEL"; then
    echo "PASS: Server is up, model loaded"
else
    echo "FAIL: Server not responding or model not loaded"
    echo "Response: $MODELS"
    exit 1
fi

# ── 1. Correctness: simple factual question ──
echo ""
echo "--- Test 1: Correctness (factual) ---"
RESP=$(curl -s "$API/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? Reply with just the city name.\"}],
    \"max_tokens\": 10,
    \"temperature\": 0
  }")
ANSWER=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
echo "Answer: $ANSWER"
if echo "$ANSWER" | grep -iq "paris"; then
    echo "PASS: Correct"
else
    echo "WARN: Unexpected answer"
fi

# ── 2. Correctness: reasoning ──
echo ""
echo "--- Test 2: Correctness (reasoning) ---"
RESP=$(curl -s "$API/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Reply with just the number.\"}],
    \"max_tokens\": 20,
    \"temperature\": 0
  }")
ANSWER=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
echo "Answer: $ANSWER"
if echo "$ANSWER" | grep -q "391"; then
    echo "PASS: Correct"
else
    echo "WARN: Expected 391"
fi

# ── 3. Single-call throughput (short output) ──
echo ""
echo "--- Test 3: Single-call throughput (50 tokens) ---"
for i in 1 2 3; do
    START=$(date +%s%N)
    RESP=$(curl -s "$API/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Explain quantum computing in simple terms.\"}],
        \"max_tokens\": 50,
        \"temperature\": 0.7
      }")
    END=$(date +%s%N)
    ELAPSED_MS=$(( (END - START) / 1000000 ))
    USAGE=$(echo "$RESP" | python3 -c "
import sys, json
r = json.load(sys.stdin)
u = r.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
print(f'prompt={pt} completion={ct}')
" 2>/dev/null || echo "PARSE_ERROR")
    CTOK=$(echo "$USAGE" | grep -oP 'completion=\K\d+' || echo "0")
    if [ "$CTOK" -gt 0 ]; then
        TOKS=$(python3 -c "print(f'{$CTOK / ($ELAPSED_MS / 1000):.1f}')")
        echo "  Run $i: ${ELAPSED_MS}ms, $USAGE, ${TOKS} tok/s"
    else
        echo "  Run $i: ${ELAPSED_MS}ms, $USAGE (could not compute tok/s)"
    fi
done

# ── 4. Single-call throughput (200 tokens) ──
echo ""
echo "--- Test 4: Single-call throughput (200 tokens) ---"
for i in 1 2 3; do
    START=$(date +%s%N)
    RESP=$(curl -s "$API/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Write a detailed explanation of how neural networks learn, covering backpropagation, gradient descent, and loss functions.\"}],
        \"max_tokens\": 200,
        \"temperature\": 0.7
      }")
    END=$(date +%s%N)
    ELAPSED_MS=$(( (END - START) / 1000000 ))
    USAGE=$(echo "$RESP" | python3 -c "
import sys, json
r = json.load(sys.stdin)
u = r.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
print(f'prompt={pt} completion={ct}')
" 2>/dev/null || echo "PARSE_ERROR")
    CTOK=$(echo "$USAGE" | grep -oP 'completion=\K\d+' || echo "0")
    if [ "$CTOK" -gt 0 ]; then
        TOKS=$(python3 -c "print(f'{$CTOK / ($ELAPSED_MS / 1000):.1f}')")
        echo "  Run $i: ${ELAPSED_MS}ms, $USAGE, ${TOKS} tok/s"
    else
        echo "  Run $i: ${ELAPSED_MS}ms, $USAGE (could not compute tok/s)"
    fi
done

# ── 5. Concurrent throughput (4 parallel requests) ──
echo ""
echo "--- Test 5: Concurrent throughput (4 parallel, 100 tokens each) ---"
TMPDIR=$(mktemp -d)
START=$(date +%s%N)
for j in 1 2 3 4; do
    curl -s "$API/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Request $j: Describe the process of photosynthesis in detail.\"}],
        \"max_tokens\": 100,
        \"temperature\": 0.7
      }" > "$TMPDIR/resp_$j.json" &
done
wait
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
TOTAL_TOKS=0
for j in 1 2 3 4; do
    CTOK=$(python3 -c "
import json
r = json.load(open('$TMPDIR/resp_$j.json'))
print(r.get('usage', {}).get('completion_tokens', 0))
" 2>/dev/null || echo "0")
    TOTAL_TOKS=$((TOTAL_TOKS + CTOK))
    echo "  Request $j: $CTOK tokens"
done
if [ "$TOTAL_TOKS" -gt 0 ]; then
    AGG_TOKS=$(python3 -c "print(f'{$TOTAL_TOKS / ($ELAPSED_MS / 1000):.1f}')")
    echo "  Total: $TOTAL_TOKS tokens in ${ELAPSED_MS}ms = ${AGG_TOKS} aggregate tok/s"
fi
rm -rf "$TMPDIR"

# ── 6. Concurrent throughput (8 parallel requests) ──
echo ""
echo "--- Test 6: Concurrent throughput (8 parallel, 100 tokens each) ---"
TMPDIR=$(mktemp -d)
START=$(date +%s%N)
for j in $(seq 1 8); do
    curl -s "$API/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Request $j: Explain the theory of relativity and its implications.\"}],
        \"max_tokens\": 100,
        \"temperature\": 0.7
      }" > "$TMPDIR/resp_$j.json" &
done
wait
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
TOTAL_TOKS=0
for j in $(seq 1 8); do
    CTOK=$(python3 -c "
import json
r = json.load(open('$TMPDIR/resp_$j.json'))
print(r.get('usage', {}).get('completion_tokens', 0))
" 2>/dev/null || echo "0")
    TOTAL_TOKS=$((TOTAL_TOKS + CTOK))
done
if [ "$TOTAL_TOKS" -gt 0 ]; then
    AGG_TOKS=$(python3 -c "print(f'{$TOTAL_TOKS / ($ELAPSED_MS / 1000):.1f}')")
    echo "  Total: $TOTAL_TOKS tokens in ${ELAPSED_MS}ms = ${AGG_TOKS} aggregate tok/s"
fi
rm -rf "$TMPDIR"

# ── 7. Long context test (2K system prompt) ──
echo ""
echo "--- Test 7: Long context (2K system prompt + question) ---"
SYSPROMPT=$(python3 -c "print('You are a helpful AI assistant. ' * 200)")
START=$(date +%s%N)
RESP=$(curl -s "$API/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSPROMPT\"},
      {\"role\": \"user\", \"content\": \"What is 2+2?\"}
    ],
    \"max_tokens\": 20,
    \"temperature\": 0
  }")
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
USAGE=$(echo "$RESP" | python3 -c "
import sys, json
r = json.load(sys.stdin)
u = r.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
print(f'prompt={pt} completion={ct}')
" 2>/dev/null || echo "PARSE_ERROR")
ANSWER=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
echo "  ${ELAPSED_MS}ms, $USAGE"
echo "  Answer: $ANSWER"

echo ""
echo "=========================================="
echo "  Benchmark Complete"
echo "=========================================="
echo ""
echo "Baseline comparison (community docker, INT4, single Spark):"
echo "  Single call: 26 tok/s"
echo "  Concurrent:  70+ tok/s"
