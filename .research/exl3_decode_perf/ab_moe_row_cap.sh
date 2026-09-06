#!/bin/bash
# A/B for prefill lever 1: the fused EXL3 MoE prefill tier's per-expert row cap.
#
#   control  legacy128    ATLAS_NO_EXL3_MOE_WIDE_ROWS=1  (kill switch -> the pre-2026-09-05 cap, 128)
#   treat    default1024  (nothing set)                   (the new default, 1024 rows/expert)
#
# Same binary, same preset, same flags, fresh server per arm, kill switch the ONLY variable
# (measurement-discipline rules: fingerprint every number; nothing below is measured until this
# runs on the GPU — every expectation in the header is a HYPOTHESIS).
#
# Step 0 — the NUMERICS GATE for the regime this lever enables (runs BEFORE any server, GPU idle):
#   the exl3_native_parity harness, legs G + G2 only (EXL3_MOE_PREFILL_ONLY=1). Its wide sub-leg
#   (legs_moe_prefill_wide.rs) runs the T=192 skewed batch — one expert at ~460 rows — with that
#   expert INSIDE the fused kernel (caps 512 host-sync and 768 no-sync shortcut), gated against the
#   f64 reference at rel 8e-3 / z 8e-2, tier stats asserted (overflow_experts == 0), and its bf16
#   bits diffed against the cap-128 run of the SAME inputs per token class: tokens touching no
#   overflow expert must be BIT-IDENTICAL (asserted), the expert-0 tokens' mismatch fraction /
#   max ulp are recorded (the "one-time fp32-order change" QUANTIFIED). Exit != 0 aborts the A/B —
#   a perf number for a kernel path that just failed parity is not a result. SKIP_PARITY=1 to
#   bypass (logged loudly). PARITY_BIN=<path> to use a prebuilt example binary instead of building.
#
# Per arm:
#   1. boot `spark serve qwen3.8-flash-next-exl3 --model-from-path <ckpt> --bind 127.0.0.1 --port 8899`
#      from the branch's release binary; wait for /v1/models; REQUIRE the boot log's
#      "EXL3 native MoE state allocated" line to show the arm's cap (else the arm is inert -> abort);
#   2. greedy SHORT sample (the LRU-cache prompt, ~60 tokens x top-10 = ~600 slots, 200 tokens out):
#      ASSERTION — must be BYTE-IDENTICAL across arms. No expert can hold more than 60 rows in that
#      batch, so both arms run the identical fused kernel, identical grid (num_active >= C=6 groups
#      -> the no-sync shortcut's default grid equals the narrowed grid), identical deterministic
#      per-slot epilogue. The only difference is the legacy arm's one host D2H sync. A mismatch here
#      is a BUG, not a numerics footnote.
#   3. LONG fixed prompt (~6000 tokens, seed 20260905) x2, 24 greedy tokens:
#      within-arm equality = cold prefill vs prefix-cache warm restore agree (informational: the
#      prefix cache is ON in the preset, so run 2 is a restore, not a second prefill);
#      cross-arm COLD (run 1 vs run 1) equality is reported but is NOT the numerics gate — a greedy
#      argmax over 24 tokens can flip on a near-tie whichever arm is right, and can agree while one
#      arm is wrong. Experts with 128 < rows <= 1024 in a 4096-token batch move from the overflow
#      tier's cooperative exl3_gemm onto the fused kernel's 16x32x128 MoE tile: same trellis decode,
#      same f16 activation precision, different fp32 accumulation ORDER. Whether that order change is
#      benign is what step 0 measures (gate + quantified bit diff); this line only says whether it
#      reached the argmax on this prompt.
#   4. measure_prefill.py --tokens 8000 11000 --repeats 2   (the headline; baseline ~390 tok/s flat)
#   5. measure_decode.py --repeats 2 --max-tokens 300         (sanity: decode never takes this tier)
#   6. stop the server, wait for the port to close.
#
# Usage:  .research/exl3_decode_perf/ab_moe_row_cap.sh            # arms: legacy128 default1024
#         ORDER="default1024 legacy128" ...ab_moe_row_cap.sh       # counterbalance the order
#         ARM_EXTRA_default1024="ATLAS_EXL3_MOE_ROWS_PER_EXPERT=2048" ...  # e.g. vllm-exl3's 2048
#         SPARK_BIN=/path/to/spark CKPT=/tank/... PORT=8899 OUT=<dir> ...
#         SKIP_PARITY=1 (bypass step 0)  PARITY_BIN=target/release/examples/exl3_native_parity
# Preconditions (house rules): ONE Atlas instance on the box; GPU idle; `free -g` sane.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
BIN=${SPARK_BIN:-$WT/target/release/spark}
CKPT=${CKPT:-/tank/exl3-ckpt/qwen38-flash-next-4.05bpw}
PORT=${PORT:-8899}
MODEL=${MODEL:-qwen3.8-flash-next}
STAMP=$(date +%Y%m%dT%H%M%S)
OUT=${OUT:-$HERE/ab_moe_row_cap/$STAMP}
ORDER=${ORDER:-"legacy128 default1024"}
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export RUST_LOG=${RUST_LOG:-info}

[ -x "$BIN" ] || { echo "no binary at $BIN (build: cargo build --release -p spark-server)"; exit 2; }
if pgrep -af "spark serv[e]" >/dev/null; then
  echo "REFUSING: another spark serve is running (one Atlas instance at a time):"; pgrep -af "spark serv[e]" | cut -c1-120; exit 2
fi
{
  echo "FINGERPRINT ab_moe_row_cap date=$STAMP host=$(hostname) port=$PORT"
  echo "binary=$BIN sha256=$(sha256sum "$BIN" | cut -c1-16) git=$(git -C "$WT" rev-parse --short HEAD 2>/dev/null) branch=$(git -C "$WT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "ckpt=$CKPT preset=qwen3.8-flash-next-exl3 (max_seq_len 131072, 4 seqs, util 0.72, bf16 KV, 2 MTP drafts, prefix cache ON, default --max-prefill-tokens 8192 -> 11K prompt = 2 chunks; MoE prefill batch 4096)"
  echo "order=$ORDER"; free -g | head -2; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null
} | tee "$OUT/fingerprint.txt"

# ── Step 0: GPU parity gate for the >128-row fused regime (legs G + G2), before any server ──
if [ "${SKIP_PARITY:-}" = "1" ]; then
  echo "PARITY GATE SKIPPED (SKIP_PARITY=1) — the >128-row fused regime is UNGATED in this run" | tee -a "$OUT/summary.txt"
else
  PBIN=${PARITY_BIN:-}
  if [ -z "$PBIN" ]; then
    # Same env the release binary was built with; the example lands in target/release/examples/ and
    # does not touch target/release/spark.
    ( cd "$WT" && PATH=/usr/local/cuda/bin:$PATH CUTLASS_HOME=${CUTLASS_HOME:-/home/ms/cutlass} \
        FLASHINFER_HOME=${FLASHINFER_HOME:-/home/ms/flashinfer} \
        RUSTFLAGS=${RUSTFLAGS:-"-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"} \
        ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
        cargo build --release -p spark-model --features cuda,gpu-examples --example exl3_native_parity \
        > "$OUT/parity_build.log" 2>&1 ) || { echo "parity example BUILD FAILED — see $OUT/parity_build.log"; exit 2; }
    PBIN=$WT/target/release/examples/exl3_native_parity
  fi
  [ -x "$PBIN" ] || { echo "no parity binary at $PBIN"; exit 2; }
  echo "parity binary=$PBIN sha256=$(sha256sum "$PBIN" | cut -c1-16)" | tee -a "$OUT/fingerprint.txt"
  echo "=== step 0: parity legs G+G2 (EXL3_MOE_PREFILL_ONLY=1) $(date +%T)" | tee -a "$OUT/summary.txt"
  ( cd "$WT" && ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
      EXL3_MOE_PREFILL_ONLY=1 "$PBIN" ) > "$OUT/parity_moe_prefill.txt" 2>&1
  PRC=$?
  grep -aE "moe-prefill WIDE|FAIL|PASS" "$OUT/parity_moe_prefill.txt" | cut -c1-260 | tee -a "$OUT/summary.txt"
  if [ $PRC -ne 0 ]; then
    echo "PARITY GATE FAILED (exit $PRC) — ABORTING the A/B; full log: $OUT/parity_moe_prefill.txt" | tee -a "$OUT/summary.txt"
    exit 1
  fi
  echo "PARITY GATE PASS (exit 0)" | tee -a "$OUT/summary.txt"
fi

LONG_PROMPT_PY='
import json,random,sys
rng=random.Random(20260905)
W=("kernel stream tensor cache block schedule verify draft router expert layer norm gate highway carry snapshot "
   "commit rewind window hash gather slot arena page token prefix budget latency bandwidth roofline launch cooperative barrier reduction").split()
body=" ".join(rng.choice(W) for _ in range(int(6000/1.037)))
print(json.dumps({"model":sys.argv[1],"temperature":0.0,"max_tokens":24,"chat_template_kwargs":{"reasoning_effort":"low"},
  "messages":[{"role":"user","content":"Fixed determinism probe. Summarize the following log in one sentence.\n\n"+body+"\n\nOne sentence:"}]}))'

extract_py() { python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(m.get("reasoning_content") or m.get("reasoning") or ""); print("=====CONTENT====="); print(m.get("content")); print("=====USAGE====="); print(json.dumps(d.get("usage")))'; }

wait_port_closed() { for i in $(seq 1 60); do curl -s -m 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || return 0; sleep 1; done; return 1; }

run_arm() {
  local ARM=$1; local LOG=$OUT/serve_$ARM.log; local ENVSTR=""
  case $ARM in
    legacy128)   ENVSTR="ATLAS_NO_EXL3_MOE_WIDE_ROWS=1"; EXPECT="x 128 rows/expert" ;;
    default1024) ENVSTR="";                              EXPECT="x 1024 rows/expert" ;;
    *) echo "unknown arm $ARM"; return 1 ;;
  esac
  local EXTRA_VAR="ARM_EXTRA_$ARM"; local EXTRA=${!EXTRA_VAR:-}
  [ -n "$EXTRA" ] && { ENVSTR="$ENVSTR $EXTRA"; EXPECT="rows/expert"; }
  echo "=== arm $ARM  env: [${ENVSTR:-<none>}]  $(date +%T)" | tee -a "$OUT/summary.txt"
  # Fresh server per arm. setsid so the whole tree is killable by group.
  setsid env $ENVSTR "$BIN" serve qwen3.8-flash-next-exl3 --model-from-path "$CKPT" \
      --bind 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
  local SPID=$!
  for i in $(seq 1 1500); do
    curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "READY after ~${i}s"; break; }
    kill -0 "$SPID" 2>/dev/null || { echo "SERVER EXITED"; tail -30 "$LOG" | cut -c1-220; return 1; }
    sleep 1
  done
  # Arm-inertness gate: the resolved cap MUST appear in the boot log with the arm's value.
  grep -a "EXL3 native MoE state allocated" "$LOG" | cut -c1-400 | tee "$OUT/capline_$ARM.txt"
  grep -a "EXL3 native MoE state:" "$LOG" | grep -ai "warn" | cut -c1-300 | tee "$OUT/capwarn_$ARM.txt"
  if ! grep -aq "$EXPECT" "$OUT/capline_$ARM.txt"; then
    echo "ARM INERT: boot log does not show '$EXPECT' — aborting arm $ARM" | tee -a "$OUT/summary.txt"
    kill -- -"$SPID" 2>/dev/null; wait_port_closed; return 1
  fi

  # 2. short greedy sample (cross-arm BYTE-IDENTICAL assertion).
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model":"'"$MODEL"'","temperature":0.0,"max_tokens":200,"chat_template_kwargs":{"reasoning_effort":"low"},
    "messages":[{"role":"user","content":"Write a complete Rust implementation of an LRU cache with generic key and value types, using a HashMap and a doubly linked list of indices into a Vec arena. Include get, put, capacity handling, and unit tests. Explain each design decision briefly in comments."}]}' \
    | extract_py > "$OUT/sample_short_$ARM.txt" 2>&1
  echo "short sample: $(wc -c < "$OUT/sample_short_$ARM.txt") bytes sha=$(sha256sum "$OUT/sample_short_$ARM.txt" | cut -c1-16)" | tee -a "$OUT/summary.txt"

  # 3. long fixed prompt x2 (run 1 cold prefill, run 2 prefix-cache warm restore).
  python3 -c "$LONG_PROMPT_PY" "$MODEL" > "$OUT/long_req.json"
  for r in 1 2; do
    curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      --data-binary @"$OUT/long_req.json" | extract_py > "$OUT/sample_long_${ARM}_run$r.txt" 2>&1
    echo "long run$r: sha=$(sha256sum "$OUT/sample_long_${ARM}_run$r.txt" | cut -c1-16) $(tail -1 "$OUT/sample_long_${ARM}_run$r.txt" | cut -c1-120)" | tee -a "$OUT/summary.txt"
  done

  # 4. the headline: cold prefill throughput at 8K / 11K (salted unique prompts, max_tokens 1).
  python3 -u "$HERE/measure_prefill.py" --port "$PORT" --model "$MODEL" --tokens 8000 11000 --repeats 2 \
    > "$OUT/prefill_$ARM.txt" 2>&1; cat "$OUT/prefill_$ARM.txt" | tee -a "$OUT/summary.txt"

  # 5. decode sanity (this tier never runs at decode; a delta here is noise or a bug).
  python3 -u "$HERE/measure_decode.py" --port "$PORT" --model "$MODEL" --repeats 2 --max-tokens 300 \
    > "$OUT/decode_$ARM.txt" 2>&1; grep -E "FINGERPRINT|SUMMARY" "$OUT/decode_$ARM.txt" | tee -a "$OUT/summary.txt"

  # Per-batch tier stats are at trace level; at info the boot line above is the fingerprint.
  grep -aE "overflow_experts|num_active" "$LOG" | tail -3 | cut -c1-200 >> "$OUT/summary.txt"

  # 6. stop.
  kill -- -"$SPID" 2>/dev/null; kill "$SPID" 2>/dev/null
  wait_port_closed || { echo "port $PORT still open — killing spark serve by pattern"; pkill -f "spark serv[e].*--port $PORT"; sleep 5; }
  echo "arm $ARM done $(date +%T)" | tee -a "$OUT/summary.txt"
}

for ARM in $ORDER; do run_arm "$ARM" || echo "arm $ARM FAILED" | tee -a "$OUT/summary.txt"; sleep 5; done

# ── Verdicts ──────────────────────────────────────────────────────────────────
{
  echo; echo "=== VERDICTS ($OUT) ==="
  A=$OUT/sample_short_legacy128.txt; B=$OUT/sample_short_default1024.txt
  if [ -s "$A" ] && [ -s "$B" ]; then
    if cmp -s "$A" "$B"; then echo "ASSERT short-sample byte-identical across arms: PASS (same fused kernel, same grid, same epilogue at <= 128 rows/expert)";
    else echo "ASSERT short-sample byte-identical across arms: FAIL — a batch with no expert above 128 rows took a different path; treat as a BUG (diff below)"; diff "$A" "$B" | head -20; fi
  else echo "short-sample assertion: NOT EVALUATED (an arm did not produce a sample)"; fi
  for ARM in legacy128 default1024; do
    R1=$OUT/sample_long_${ARM}_run1.txt; R2=$OUT/sample_long_${ARM}_run2.txt
    [ -s "$R1" ] && [ -s "$R2" ] && { cmp -s "$R1" "$R2" && echo "INFO $ARM long prompt cold vs warm-restore: identical" || echo "INFO $ARM long prompt cold vs warm-restore: DIFFER (prefix-cache restore path, see qwen4exp-phantom-corruption)"; }
  done
  L1=$OUT/sample_long_legacy128_run1.txt; L2=$OUT/sample_long_default1024_run1.txt
  [ -s "$L1" ] && [ -s "$L2" ] && { cmp -s "$L1" "$L2" && echo "INFO long prompt COLD across arms: identical (the fp32-order change did not reach this prompt's argmax)" || echo "INFO long prompt COLD across arms: DIFFER — not a numerics verdict either way (argmax over 24 greedy tokens); the numerics gate is step 0 below"; }
  echo "--- step 0 numerics gate (the >128-row fused regime vs the f64 reference, and its bit diff vs the cap-128 tier):"
  if [ -s "$OUT/parity_moe_prefill.txt" ]; then grep -aE "moe-prefill WIDE|^PASS|^FAIL" "$OUT/parity_moe_prefill.txt" | cut -c1-260; else echo "NOT RUN (SKIP_PARITY=1) — the >128-row fused regime is ungated in this run"; fi
  for ARM in legacy128 default1024; do grep -h "SUMMARY" "$OUT/prefill_$ARM.txt" 2>/dev/null | sed "s/^/$ARM prefill /"; grep -h "SUMMARY" "$OUT/decode_$ARM.txt" 2>/dev/null | sed "s/^/$ARM decode  /"; done
  echo "Quote prefill tok/s ONLY with this fingerprint file: $OUT/fingerprint.txt (rule 1); one pass is not a result — rerun with ORDER reversed (rule: counterbalance)."
} | tee -a "$OUT/summary.txt"
