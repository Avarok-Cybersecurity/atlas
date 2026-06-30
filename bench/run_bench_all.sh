#!/usr/bin/env bash
# run_bench_all.sh — Run FINDINGS benchmark across all 5 configs.
# NEVER start two servers simultaneously.
# Usage: bash run_bench_all.sh [A|B|C|D|E]  (no arg = run all)

set -euo pipefail

SPARK="/home/isolo/Projects/atlas/target/release/spark"
MODEL="/home/isolo/Projects/isolorg/models/Qwen3.6-27B-NVFP4"
DFLASH_MODEL="/home/isolo/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106"
SERVER_LOG="/tmp/atlas_bench_server.log"
BENCH_SCRIPT="$(dirname "$0")/bench_findings.py"
PORT=8888
URL="http://localhost:${PORT}"

export RUSTFLAGS="-L /tmp/nccl-stubs"
export LD_LIBRARY_PATH="/home/isolo/.cache/uv/archive-v0/V0RWp7iPS0kW3pWE/nvidia/nccl/lib"

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[bench] $*"; }

check_no_server() {
    if pgrep -f "spark serve" > /dev/null 2>&1; then
        echo "ERROR: A spark server is already running. Kill it first."
        pgrep -fa "spark serve"
        exit 1
    fi
}

start_server() {
    local label="$1"; shift
    log "Starting server for config: $label"
    log "Command: $*"
    echo "=== $label ===" >> "$SERVER_LOG"
    # Start under systemd scope for memory cap, detached
    systemd-run --user --scope -p MemoryMax=110G \
        env RUSTFLAGS="$RUSTFLAGS" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$@" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    log "Server PID group started (systemd scope)"
}

wait_server_ready() {
    log "Waiting for server at $URL ..."
    for i in $(seq 1 120); do
        if curl -sf "$URL/v1/models" > /dev/null 2>&1; then
            log "Server ready after ${i}s"
            return 0
        fi
        sleep 2
    done
    log "ERROR: Server did not become ready in 240s"
    return 1
}

stop_server() {
    log "Stopping server..."
    pkill -f "spark serve" 2>/dev/null || true
    sleep 5
    if pgrep -f "spark serve" > /dev/null 2>&1; then
        pkill -9 -f "spark serve" 2>/dev/null || true
        sleep 3
    fi
    log "Server stopped"
}

run_bench() {
    local label="$1"
    log "Running benchmark: $label"
    python3 "$BENCH_SCRIPT" --label "$label" --url "$URL" --runs 2 --warmup 1
}

# ── base flags shared across all configs ─────────────────────────────────────

BASE_FLAGS=(
    "$SPARK" serve "$MODEL"
    --port "$PORT"
    --kv-cache-dtype nvfp4
    --kv-high-precision-layers 4
    --scheduling-policy slai
)

DFLASH_FLAGS=(
    "${BASE_FLAGS[@]}"
    --dflash
    --draft-model "$DFLASH_MODEL"
    --gpu-memory-utilization 0.75
    --max-batch-size 1
)

# ── config definitions ────────────────────────────────────────────────────────

run_config_A() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    start_server "A: No-MTP" "${BASE_FLAGS[@]}"
    wait_server_ready
    run_bench "A: No-MTP"
    stop_server
}

run_config_B() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    start_server "B: MTP fp8" "${BASE_FLAGS[@]}" \
        --speculative --mtp-quantization fp8 --num-drafts 1
    wait_server_ready
    run_bench "B: MTP fp8"
    stop_server
}

run_config_C() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    start_server "C: MTP bf16" "${BASE_FLAGS[@]}" \
        --speculative --num-drafts 1
    wait_server_ready
    run_bench "C: MTP bf16"
    stop_server
}

run_config_D() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=512
    unset ATLAS_DFLASH_DRAFT_CAP
    start_server "D: DFlash cap=1 T=0" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    run_bench "D: DFlash cap=1 T=0"
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW
}

# DFlash K=2 at T=0.6 — rejection sampling (server reused from D, no restart)
run_config_D06() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=64
    unset ATLAS_DFLASH_DRAFT_CAP
    start_server "D06: DFlash cap=1 T=0.6" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    python3 "$BENCH_SCRIPT" --label "D06: DFlash cap=1 T=0.6" --url "$URL" --runs 2 --warmup 1 --temperature 0.6
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW
}

# DFlash K=2 at T=1.0 — rejection sampling
run_config_D10() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=64
    unset ATLAS_DFLASH_DRAFT_CAP
    start_server "D10: DFlash cap=1 T=1.0" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    python3 "$BENCH_SCRIPT" --label "D10: DFlash cap=1 T=1.0" --url "$URL" --runs 2 --warmup 1 --temperature 1.0
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW
}

# DFlash K=16 sequential (baseline, no prefill SSM)
run_config_E() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=64
    export ATLAS_DFLASH_DRAFT_CAP=15
    start_server "E: DFlash K=16 seq T=0" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    run_bench "E: DFlash K=16 seq T=0"
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW ATLAS_DFLASH_DRAFT_CAP
}

# DFlash K=4 + PREFILL_SSM (WY4 GDN) — smallest cap that hits step_verify_dflash
run_config_F4() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=64
    export ATLAS_DFLASH_DRAFT_CAP=3
    export ATLAS_VERIFY_PREFILL_SSM=1
    start_server "F4: DFlash K=4 prefill_ssm T=0" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    run_bench "F4: DFlash K=4 prefill_ssm T=0"
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW ATLAS_DFLASH_DRAFT_CAP ATLAS_VERIFY_PREFILL_SSM
}

# DFlash K=8 + PREFILL_SSM
run_config_F8() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=64
    export ATLAS_DFLASH_DRAFT_CAP=7
    export ATLAS_VERIFY_PREFILL_SSM=1
    start_server "F8: DFlash K=8 prefill_ssm T=0" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    run_bench "F8: DFlash K=8 prefill_ssm T=0"
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW ATLAS_DFLASH_DRAFT_CAP ATLAS_VERIFY_PREFILL_SSM
}

# DFlash K=12 + PREFILL_SSM
run_config_F12() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=64
    export ATLAS_DFLASH_DRAFT_CAP=11
    export ATLAS_VERIFY_PREFILL_SSM=1
    start_server "F12: DFlash K=12 prefill_ssm T=0" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    run_bench "F12: DFlash K=12 prefill_ssm T=0"
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW ATLAS_DFLASH_DRAFT_CAP ATLAS_VERIFY_PREFILL_SSM
}

# DFlash K=16 + PREFILL_SSM
run_config_F16() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=64
    export ATLAS_DFLASH_DRAFT_CAP=15
    export ATLAS_VERIFY_PREFILL_SSM=1
    start_server "F16: DFlash K=16 prefill_ssm T=0" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    run_bench "F16: DFlash K=16 prefill_ssm T=0"
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW ATLAS_DFLASH_DRAFT_CAP ATLAS_VERIFY_PREFILL_SSM
}

# DFlash K=2, capture_layer_offset=+1 → layers [2,17,32,47,62]
run_config_G_offset1() {
    check_no_server
    truncate -s 0 "$SERVER_LOG"
    export ATLAS_DFLASH_CTX_WINDOW=64
    export ATLAS_DFLASH_DRAFT_CAP=1
    export ATLAS_DFLASH_CAPTURE_LAYER_OFFSET=1
    start_server "G: DFlash K=2 offset+1 T=0" "${DFLASH_FLAGS[@]}"
    wait_server_ready
    run_bench "G: DFlash K=2 offset+1 T=0"
    stop_server
    unset ATLAS_DFLASH_CTX_WINDOW ATLAS_DFLASH_DRAFT_CAP ATLAS_DFLASH_CAPTURE_LAYER_OFFSET
}

# ── entrypoint ────────────────────────────────────────────────────────────────

CONFIGS="${1:-ABCDE}"
log "Running configs: $CONFIGS"

[[ "$CONFIGS" == *A* ]] && run_config_A
[[ "$CONFIGS" == *B* ]] && run_config_B
[[ "$CONFIGS" == *C* ]] && run_config_C
[[ "$CONFIGS" =~ (^|[^0-9])D([^0-9]|$) ]] && run_config_D
[[ "$CONFIGS" == *"D06"* ]] && run_config_D06
[[ "$CONFIGS" == *"D10"* ]] && run_config_D10
[[ "$CONFIGS" == *E* ]] && run_config_E
[[ "$CONFIGS" == *"F4"* ]] && run_config_F4
[[ "$CONFIGS" == *"F8"* ]] && run_config_F8
[[ "$CONFIGS" == *"F12"* ]] && run_config_F12
[[ "$CONFIGS" == *"F16"* ]] && run_config_F16
[[ "$CONFIGS" == *"G"* ]] && run_config_G_offset1

log "All done. Server log: $SERVER_LOG"
