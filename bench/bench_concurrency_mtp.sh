#!/bin/bash
# Concurrency sweep benchmark for Atlas Spark
PORT=8890
MAX_TOKENS=150
PROMPT="Explain the theory of general relativity in detail, covering spacetime curvature, gravitational time dilation, and the equivalence principle."

echo "Waiting for server..."
until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do sleep 1; done
echo "Server ready"

# Warmup
echo "=== Warmup ==="
for i in 1 2; do
  curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":20,"stream":false}' > /dev/null
done
echo "  done"
sleep 1

# Single request function — writes "tokens elapsed_ms" to a temp file
run_one() {
  local id=$1 tmpfile=$2
  local t0=$(date +%s%N)
  local resp
  resp=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS,\"stream\":false}")
  local t1=$(date +%s%N)
  local ms=$(( (t1 - t0) / 1000000 ))
  local toks=$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo 0)
  echo "$toks $ms" >> "$tmpfile"
}

bench() {
  local C=$1
  echo ""
  echo "=== C=$C ==="
  local tmpfile=$(mktemp)
  local t0=$(date +%s%N)

  for i in $(seq 1 $C); do
    run_one $i "$tmpfile" &
  done
  wait

  local t1=$(date +%s%N)
  local wall_ms=$(( (t1 - t0) / 1000000 ))

  python3 -c "
import sys
lines = open('$tmpfile').read().strip().split('\n')
total_tok = sum(int(l.split()[0]) for l in lines)
C = $C
wall_ms = $wall_ms
print(f'  Requests: {C}, Total tokens: {total_tok}')
print(f'  Wall time: {wall_ms}ms')
print(f'  Aggregate: {total_tok/(wall_ms/1000):.1f} tok/s')
print(f'  Per-request: {total_tok/C/(wall_ms/1000):.1f} tok/s')
for i,l in enumerate(lines):
    t,ms = l.split()
    print(f'    req {i+1}: {t} tok in {ms}ms ({int(t)/(int(ms)/1000):.1f} tok/s)')
"
  rm -f "$tmpfile"
}

echo ""
echo "========================================="
echo "Concurrency Sweep (MTP enabled)"
echo "========================================="

bench 1
sleep 2
bench 1
sleep 2
bench 2
sleep 2
bench 4
sleep 2
bench 8

echo ""
echo "Done!"
