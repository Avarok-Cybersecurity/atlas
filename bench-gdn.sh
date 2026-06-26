#!/usr/bin/env bash
# bench-gdn.sh — GDN kernel baseline/comparison benchmark
# Usage: ./bench-gdn.sh <label>
# Example: ./bench-gdn.sh "before-norm-opt"
# Results saved to: bench-results/<label>-<timestamp>.txt

set -euo pipefail

LABEL="${1:-unnamed}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="$(dirname "$0")/bench-results"
OUTFILE="$OUTDIR/${LABEL}-${TIMESTAMP}.txt"
PORT="${PORT:-8888}"
RUNS=5
MAX_TOKENS=400
PROMPT="Write a complete Python implementation of a red-black tree with insert, delete, search, and in-order traversal. Use type hints throughout."

mkdir -p "$OUTDIR"

echo "=== GDN Benchmark: $LABEL ===" | tee "$OUTFILE"
echo "Date:      $(date)" | tee -a "$OUTFILE"
echo "Label:     $LABEL" | tee -a "$OUTFILE"
echo "Runs:      $RUNS" | tee -a "$OUTFILE"
echo "Port:      $PORT" | tee -a "$OUTFILE"
echo "Max tokens: $MAX_TOKENS" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

if ! curl -sf http://localhost:$PORT/v1/models > /dev/null 2>&1; then
    echo "ERROR: No server on port $PORT" | tee -a "$OUTFILE"
    exit 1
fi

PROMPT_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROMPT")

# Collect results to a temp file for python summary
TMPDATA=$(mktemp)

echo "--- Per-run results ---" | tee -a "$OUTFILE"

for i in $(seq 1 $RUNS); do
    RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"x\",
            \"messages\": [{\"role\": \"user\", \"content\": $PROMPT_JSON}],
            \"max_tokens\": $MAX_TOKENS,
            \"temperature\": 0,
            \"thinking\": {\"type\": \"disabled\"}
        }")

    python3 - "$RESP" "$i" "$OUTFILE" "$TMPDATA" << 'PYEOF'
import json, sys

resp_str, run_idx, outfile, tmpdata = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    r = json.loads(resp_str)
except Exception as e:
    print(f"  run {run_idx}: ERROR parsing response: {e}")
    sys.exit(0)

u = r.get("usage", {})
tok_s = u.get("response_token/s", 0)
tokens = u.get("completion_tokens", 0)
ttft = u.get("time_to_first_token_ms", 0)
step_ms = 1000 / tok_s if tok_s > 0 else 0
content = r.get("choices", [{}])[0].get("message", {}).get("content", "")
first_line = content.strip().split("\n")[0][:120]

line = f"  run {run_idx}: {tok_s:.3f} tok/s  ({tokens} tokens, step={step_ms:.2f}ms, TTFT={ttft:.1f}ms)"
print(line)
with open(outfile, "a") as f:
    f.write(line + "\n")
    f.write(f"    output[0]: {first_line}\n")

# Save tok_s to tmpdata for summary
with open(tmpdata, "a") as f:
    f.write(f"{tok_s}\n")

# Print first line of output for sanity check
print(f"    ↳ {first_line}")
PYEOF
done

# Summary
echo "" | tee -a "$OUTFILE"
echo "--- Summary ---" | tee -a "$OUTFILE"
python3 - "$OUTFILE" "$TMPDATA" << 'PYEOF'
import sys

outfile, tmpdata = sys.argv[1], sys.argv[2]
with open(tmpdata) as f:
    vals = [float(l.strip()) for l in f if l.strip()]

if not vals:
    print("  No data collected.")
    sys.exit(0)

vals_sorted = sorted(vals)
n = len(vals)
median = vals_sorted[n // 2] if n % 2 == 1 else (vals_sorted[n//2-1] + vals_sorted[n//2]) / 2
mean = sum(vals) / n
mn, mx = min(vals), max(vals)
step_median = 1000 / median if median > 0 else 0

lines = [
    f"tok/s    median={median:.3f}  mean={mean:.3f}  min={mn:.3f}  max={mx:.3f}",
    f"step_ms  median={step_median:.2f}ms",
    f"variance {((mx-mn)/median*100):.1f}%  (max-min spread)",
]
for line in lines:
    print("  " + line)
    with open(outfile, "a") as f:
        f.write("  " + line + "\n")
PYEOF
rm -f "$TMPDATA"

# Extract ATLAS_PROFILE kernel timing from server log
LOG_FILE="/tmp/atlas-27b-nomtp.log"
if [ ! -f "$LOG_FILE" ]; then
    LOG_FILE="/tmp/atlas-27b-profile.log"
fi
if [ ! -f "$LOG_FILE" ]; then
    LOG_FILE="/tmp/atlas-27b-k3.log"
fi

if [ -f "$LOG_FILE" ]; then
    echo "" | tee -a "$OUTFILE"
    echo "--- Kernel timing (ATLAS_PROFILE, last ~5 steps) ---" | tee -a "$OUTFILE"
    grep -E "SSM (qkvz|ba_gates|gdn_decode|gated_norm|out_proj)|PROFILE tok=" "$LOG_FILE" \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | tail -400 \
        | python3 - "$OUTFILE" << 'PYEOF'
import sys, re

lines = sys.stdin.read().splitlines()
outfile = sys.argv[1]
kernels = {}
profiles = []

for line in lines:
    m = re.search(r'SSM (\w+): (\d+)μs', line)
    if m:
        kernels.setdefault(m.group(1), []).append(int(m.group(2)))
    m = re.search(r'PROFILE tok=\d+: total=([\d.]+)ms attn=([\d.]+)ms\(\d+\) ssm=([\d.]+)ms\(\d+\) head=([\d.]+)ms', line)
    if m:
        profiles.append(tuple(float(x) for x in m.groups()))

output = []
for name, vals in sorted(kernels.items()):
    s = sorted(vals)
    med = s[len(s)//2]
    clean = [v for v in vals if v <= med * 3]
    avg = sum(clean) / len(clean) if clean else 0
    output.append(f"  SSM {name:12s}: avg={avg:.0f}μs  median={med}μs  n={len(vals)}")

if profiles:
    output.append("")
    for total, attn, ssm, head in profiles[-5:]:
        ssm_per = ssm / 48
        output.append(f"  PROFILE: total={total:.1f}ms  attn={attn:.1f}ms  ssm={ssm:.1f}ms({ssm_per:.2f}ms/layer)  head={head:.1f}ms")

for line in output:
    print(line)
    with open(outfile, "a") as f:
        f.write(line + "\n")
PYEOF
fi

echo "" | tee -a "$OUTFILE"
echo "Saved: $OUTFILE"
