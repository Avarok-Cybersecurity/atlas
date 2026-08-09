#!/bin/bash
# benchmark-pr Gate C2 (NVFP4 numerical smoke) + Gate C (warm-TTFT regression guard)
# for the ATLAS_GDN_REGRESIDENT default-flip. Runs on dgx1.
#
# Gate C is a RELATIVE, same-box, back-to-back A/B — never an absolute or stored
# TTFT number. Because this change is env-gated, both legs use the SAME binary and
# differ only by the kill switch, which the skill calls the strictest possible A/B
# and which additionally proves the kill switch actually works:
#   pr  leg = default            (regresident ON, the new default)
#   ctl leg = ATLAS_NO_GDN_REGRESIDENT=1  (regresident OFF, prior behaviour)
#
# TRAPS encoded here:
#  * The quick yaml hardcodes `http://localhost:8085`. Serving on 8888 yields
#    174/174 ConnectionRefusedError, **exit code 0**, a written result_summary.json
#    and a fast TTFT over nothing — a phantom win. Serve on 8085.
#  * dgx1 is GB10 unified memory: util MUST be <=0.70 or the box OOM-freezes. The
#    skill's C2 recipe says 0.85, which is unsafe here; a relative same-box A/B
#    stays valid as long as BOTH legs share identical flags, so 0.70 on both.
#  * `n_samples_failed != 0` means the leg is INVALID, not fast. Checked below.
#
# Usage: pr_gate_c2_and_c.sh <binary> <outdir>
set -u
BIN="${1:?path to the PR spark binary (27b target)}"
OUT="${2:?output dir}"
EP=/workspace/endpoints
YAML=$EP/examples/10_Edge_Agentic_Example/online_agentic_coding_atlas_quick.yaml
MODEL=unsloth/Qwen3.6-27B-NVFP4
SERVED_NAME="Qwen3.6-27B-Q4_K_M"   # must equal the yaml's model_params.name
PORT=8085
mkdir -p "$OUT"

serve() { # $1 = leg, $2 = extra -e args
  sudo docker rm -f atlas-prgate >/dev/null 2>&1; sleep 3
  # shellcheck disable=SC2086
  sudo docker run -d --name atlas-prgate --network host --ipc host --gpus all \
    $2 \
    -v /workspace/.cache/huggingface:/root/.cache/huggingface \
    -v "$BIN:/usr/local/bin/spark:ro" \
    atlas-gb10:followups serve "$MODEL" --host 0.0.0.0 --port $PORT \
    --model-name "$SERVED_NAME" \
    --max-seq-len 32768 --max-batch-size 1 --gpu-memory-utilization 0.70 \
    --kv-cache-dtype bf16 --enable-prefix-caching --ssm-cache-slots 128 \
    --ssm-checkpoint-interval 32 --speculative --num-drafts 1 --mtp-quantization bf16 \
    --tool-call-parser qwen3_coder --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
  for _ in $(seq 1 200); do
    curl -sf -m4 http://localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen && return 0
    sudo docker ps --format '{{.Names}}' | grep -q atlas-prgate || { echo "SERVE_DIED leg=$1"; return 1; }
    sleep 5
  done
  echo "SERVE_TIMEOUT leg=$1"; return 1
}

for leg in pr ctl; do
  case $leg in
    pr)  EXTRA="" ;;                                  # default => regresident ON
    ctl) EXTRA="-e ATLAS_NO_GDN_REGRESIDENT=1" ;;     # kill switch => OFF
  esac
  serve "$leg" "$EXTRA" || exit 1
  echo "=== leg=$leg serve up on :$PORT (${EXTRA:-<default>}) ==="

  if [ "$leg" = pr ]; then
    echo "--- GATE C2: NVFP4 numerical smoke (run once, on the PR leg) ---"
    python3 /workspace/.claude/skills/benchmark-pr/gate_c2_nvfp4_smoke.py \
      --url http://localhost:$PORT --model "$SERVED_NAME" 2>&1 | tee "$OUT/gate_c2.txt"
    echo "GATE_C2_EXIT=${PIPESTATUS[0]}" | tee -a "$OUT/gate_c2.txt"
  fi

  RD="results/prgate_${leg}_$(date +%H%M%S)"
  ( cd $EP && ./.venv/bin/inference-endpoint benchmark from-config \
      --config "$YAML" --report-dir "$RD" -v ) 2>&1 | tail -25 | tee "$OUT/$leg.bench.log"
  cp "$EP/$RD/performance/result_summary.json" "$OUT/$leg.summary.json" 2>/dev/null \
    || cp "$EP/$RD/result_summary.json" "$OUT/$leg.summary.json" 2>/dev/null || true

  # Banner proves which GDN path actually ran — fires on first replay, not startup.
  sudo docker logs atlas-prgate 2>&1 | grep -aE 'GDN prefill: (FLA chunked|REGISTER-RESIDENT)' \
    | sed 's/.*INFO.*: //' | sort -u | tee "$OUT/$leg.banner.txt"
  sudo docker rm -f atlas-prgate >/dev/null 2>&1
done

python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
def load(leg):
    p = os.path.join(out, f"{leg}.summary.json")
    if not os.path.exists(p): return None
    return json.load(open(p))
a, b = load("ctl"), load("pr")
print("\n=== GATE C: warm-TTFT regression guard (same box, same binary, flag-only A/B) ===")
if not a or not b:
    print("MISSING summary json for a leg -> INVALID, not a pass"); sys.exit(1)
for tag, d in (("ctl (regresident OFF)", a), ("pr  (regresident ON)", b)):
    f = d.get("n_samples_failed", -1)
    print(f"{tag}: n_completed={d.get('n_samples_completed')} n_failed={f}")
    if f != 0:
        print("  -> n_samples_failed != 0: this leg is INVALID (fast TTFT over surviving turns), NOT fast")
bad = (a.get("n_samples_failed") or 0) != 0 or (b.get("n_samples_failed") or 0) != 0
def g(d, k):
    return (d["ttft"]["percentiles"]["90.0"] if k == "p90" else d["ttft"][k]) / 1e6
print(f"\n{'metric':8s} {'ctl(OFF)':>12s} {'pr(ON)':>12s} {'delta':>10s}")
res = {}
for k in ("median", "avg", "p90"):
    x, y = g(a, k), g(b, k)
    res[k] = 100 * (y - x) / x
    print(f"{k:8s} {x:12.1f} {y:12.1f} {res[k]:+9.2f}%")
print(f"\nGate C rule: FAIL iff median regresses >3% OR p90 >5% (positive = regression)")
verdict = "FAIL" if (bad or res['median'] > 3 or res['p90'] > 5) else "PASS"
print(f"GATE_C_VERDICT={verdict}" + ("  (INVALID leg)" if bad else ""))
PY
echo "PR_GATE_C2C_DONE out=$OUT"
