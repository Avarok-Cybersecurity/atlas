#!/bin/bash
# EXL3 cold-prefill kernel profile — the R1 step of vllm_exl3_prefill_review.md.
#
# Boots this branch's release binary through the named preset under
# `nsys launch --trace=cuda,nvtx` (capture OFF), warms the server with one
# short cold request, measures two UNCAPTURED ~8K cold prefills as the
# reference wall, then captures exactly ONE ~8K cold prefill (fresh salt,
# max_tokens=1 => one prefill chunk + one decode step) between nsys start/stop,
# stops the server and summarises:
#   $RUN_DIR/nsys_<arm>_cuda_gpu_kern_sum.csv   raw nsys kernel summary
#   $RUN_DIR/nsys_<arm>_cuda_api_sum.csv        host API summary (syncs, launches)
#   $RUN_DIR/kernel_table.md                    per-kernel + per-family % of GPU time
#   $RUN_DIR/prefill_*.txt                      measure_prefill.py output (fingerprinted)
#   $RUN_DIR/fingerprint.txt                    build/binary/flags/env/box record
#
# Usage:   ./prefill_profile.sh                     (default arm, no env)
#          ARM=cap512 ATLAS_SOME_SWITCH=1 ./prefill_profile.sh   (A/B arm: the
#          exported switch is the ONLY variable; everything else is the preset)
# Knobs:   PORT (8899) TARGET_TOKENS (8000) WARM_TOKENS (512) SPARK_BIN CKPT
#          EXTRA_SERVE_ARGS (appended to `spark serve`, e.g. "--max-prefill-tokens 4096")
#          EXL3_RUN_ROOT (where runs/ goes; gitignored by default)
#
# Refuses to start if another `spark serve` or an nsys session is alive, or if
# MemAvailable is below the pledge. Cleans up (nsys stop, server TERM/KILL) on
# any exit. Nothing here is a perf claim: the table is a kernel-time split of
# one captured request under injection, to be read with MEASUREMENT_PLAN.md.
set -euo pipefail
# shellcheck source=measure_common.sh
source "$(dirname "$0")/measure_common.sh"

TARGET_TOKENS=${TARGET_TOKENS:-8000}
WARM_TOKENS=${WARM_TOKENS:-512}
# The overflow-tier statistics (`overflow_experts` per 4096-token batch) are a
# trace!-level line in forward_prefill_exl3 — R1 asks for them, so widen the
# filter for that one module only.
export RUST_LOG="${RUST_LOG:-info},spark_model::layers::moe::forward_prefill_exl3=trace"

refuse_if_busy
make_run_dir prefill_profile
write_fingerprint "$RUN_DIR/fingerprint.txt"
echo "harness=measure_prefill.py target_tokens=$TARGET_TOKENS warm_tokens=$WARM_TOKENS max_tokens=1 temp=0 salted-unique" >> "$RUN_DIR/fingerprint.txt"
log "run dir: $RUN_DIR"

boot_server "$RUN_DIR/serve.log" nsys
start_mem_watchdog
wait_ready "$RUN_DIR/serve.log" || die "boot failed"

# Model id as the server advertises it (the preset names it qwen3.8-flash-next-exl3).
MODEL=$(curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
log "model id: $MODEL"

# 1. Warm-up: first-request one-time costs (allocations, cuBLASLt heuristics,
#    PLE row-cache faults) must not land in the captured trace.
log "warm-up ${WARM_TOKENS}-token cold prefill (uncaptured)"
python3 -u "$SCRIPT_DIR/measure_prefill.py" --port "$PORT" --model "$MODEL" --tokens "$WARM_TOKENS" --repeats 1 \
    > "$RUN_DIR/prefill_warmup.txt" 2>&1
cat "$RUN_DIR/prefill_warmup.txt"

# 2. Reference wall under injection but WITHOUT capture (n=2), so the captured
#    request's wall can be checked for capture perturbation.
log "reference ${TARGET_TOKENS}-token cold prefill x2 (injected, uncaptured)"
python3 -u "$SCRIPT_DIR/measure_prefill.py" --port "$PORT" --model "$MODEL" --tokens "$TARGET_TOKENS" --repeats 2 \
    > "$RUN_DIR/prefill_reference.txt" 2>&1
cat "$RUN_DIR/prefill_reference.txt"

# 3. The captured request. measure_prefill.py salts from the wall clock, so
#    this prompt shares no prefix with the two above (prefix cache is ON in the
#    preset; a hit would profile a restore, not a prefill).
REP="$RUN_DIR/nsys_${ARM}"
log "nsys start (session $NSYS_SESSION) -> $REP.nsys-rep"
"$NSYS" start --session="$NSYS_SESSION" --output="$REP" --force-overwrite=true
sleep 1
python3 -u "$SCRIPT_DIR/measure_prefill.py" --port "$PORT" --model "$MODEL" --tokens "$TARGET_TOKENS" --repeats 1 \
    > "$RUN_DIR/prefill_captured.txt" 2>&1
cat "$RUN_DIR/prefill_captured.txt"
sleep 1
log "nsys stop"
"$NSYS" stop --session="$NSYS_SESSION"
NSYS_SESSION=""   # stopped; cleanup must not stop it twice

# Overflow-tier statistics for the captured window (last batches in the log).
{ grep -a "EXL3 MoE prefill batch" "$RUN_DIR/serve.log" || true; } | sed 's/\x1b\[[0-9;]*m//g' | tail -12 \
    > "$RUN_DIR/overflow_stats.txt"
log "overflow stats (last batches):"; cat "$RUN_DIR/overflow_stats.txt"

stop_server

# 4. Summaries. `--output <base>` yields <base>_cuda_gpu_kern_sum.csv etc.
log "nsys stats"
"$NSYS" stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv --output "$REP" "$REP.nsys-rep" \
    > "$RUN_DIR/nsys_stats.log" 2>&1 || { tail -20 "$RUN_DIR/nsys_stats.log"; die "nsys stats failed"; }
ls -la "$RUN_DIR"/*.csv

CAPTURED_WALL=$(awk '/^target~/ {for (i=1;i<=NF;i++) if ($i ~ /^wall=/) {sub("wall=","",$i); sub("s","",$i); print $i}}' "$RUN_DIR/prefill_captured.txt" | tail -1)
python3 -u "$SCRIPT_DIR/nsys_kern_table.py" "${REP}_cuda_gpu_kern_sum.csv" \
    --api "${REP}_cuda_api_sum.csv" --wall-s "${CAPTURED_WALL:-0}" --top 40 \
    | tee "$RUN_DIR/kernel_table.md"

log "artefacts: $RUN_DIR (kernel_table.md, nsys_*_cuda_gpu_kern_sum.csv, prefill_*.txt, fingerprint.txt)"
