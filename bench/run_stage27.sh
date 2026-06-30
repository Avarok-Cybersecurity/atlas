#!/usr/bin/env bash
# Stage 27 bench: 14 configs, 1 warmup + 2 runs each, median reported.
set -euo pipefail

SPARK="/home/isolo/Projects/atlas/target/release/spark"
MODEL="/home/isolo/Projects/isolorg/models/Qwen3.6-27B-NVFP4"
DFLASH_MODEL="/home/isolo/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106"
SERVER_LOG="/tmp/atlas_bench_server.log"
BENCH="python3 /home/isolo/Projects/atlas/bench/bench_findings.py"
PORT=8888
URL="http://localhost:${PORT}"

export RUSTFLAGS="-L /tmp/nccl-stubs"
export LD_LIBRARY_PATH="/home/isolo/.cache/uv/archive-v0/V0RWp7iPS0kW3pWE/nvidia/nccl/lib"

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

stop_server() {
    pkill -f "spark serve" 2>/dev/null || true
    sleep 6
}

start_and_wait() {
    local label="$1"; shift
    echo ""
    echo "=== Starting server: $label ==="
    truncate -s 0 "$SERVER_LOG"
    systemd-run --user --scope -p MemoryMax=110G \
        env RUSTFLAGS="$RUSTFLAGS" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$@" >> "$SERVER_LOG" 2>&1 &
    for i in $(seq 1 120); do
        if curl -sf "$URL/v1/models" > /dev/null 2>&1; then
            echo "  Ready after ${i}s"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: server not ready"
    tail -20 "$SERVER_LOG"
    exit 1
}

bench() {
    local label="$1"; shift
    $BENCH --label "$label" --url "$URL" --runs 2 --warmup 1 "$@"
}

# ── 1. No-MTP baseline ──────────────────────────────────────────────────────
stop_server
start_and_wait "No-MTP" "${BASE_FLAGS[@]}"
bench "1: No-MTP (NVFP4)"
stop_server

# ── 2. MTP fp8 ──────────────────────────────────────────────────────────────
start_and_wait "MTP fp8" "${BASE_FLAGS[@]}" \
    --speculative --mtp-quantization fp8 --num-drafts 1
bench "2: MTP fp8 K=2"
stop_server

# ── 3-5. DFlash CAP=1 (WY2 graphed) — vary T only, one server ───────────────
start_and_wait "DFlash CAP=1" \
    env ATLAS_DFLASH_DRAFT_CAP=1 ATLAS_DFLASH_CTX_WINDOW=512 \
    "${DFLASH_FLAGS[@]}"
bench "3: DFlash CAP=1 T=0  (WY2)"  --temperature 0.0
bench "4: DFlash CAP=1 T=0.6 (WY2)" --temperature 0.6
bench "5: DFlash CAP=1 T=1.0 (WY2)" --temperature 1.0
stop_server

# ── 6-8. DFlash CAP=3 (WY4 native) ─────────────────────────────────────────
start_and_wait "DFlash CAP=3" \
    env ATLAS_DFLASH_DRAFT_CAP=3 ATLAS_DFLASH_CTX_WINDOW=512 \
    "${DFLASH_FLAGS[@]}"
bench "6: DFlash CAP=3 T=0  (WY4)"  --temperature 0.0
bench "7: DFlash CAP=3 T=0.6 (WY4)" --temperature 0.6
bench "8: DFlash CAP=3 T=1.0 (WY4)" --temperature 1.0
stop_server

# ── 9-11. DFlash CAP=7 + prefill_ssm ────────────────────────────────────────
start_and_wait "DFlash CAP=7 prefill_ssm" \
    env ATLAS_DFLASH_DRAFT_CAP=7 ATLAS_DFLASH_CTX_WINDOW=512 \
        ATLAS_VERIFY_PREFILL_SSM=1 \
    "${DFLASH_FLAGS[@]}"
bench "9:  DFlash CAP=7 T=0  (prefill_ssm)" --temperature 0.0
bench "10: DFlash CAP=7 T=0.6 (prefill_ssm)" --temperature 0.6
bench "11: DFlash CAP=7 T=1.0 (prefill_ssm)" --temperature 1.0
stop_server

# ── 12-14. DFlash CAP=15 + prefill_ssm ──────────────────────────────────────
start_and_wait "DFlash CAP=15 prefill_ssm" \
    env ATLAS_DFLASH_DRAFT_CAP=15 ATLAS_DFLASH_CTX_WINDOW=512 \
        ATLAS_VERIFY_PREFILL_SSM=1 \
    "${DFLASH_FLAGS[@]}"
bench "12: DFlash CAP=15 T=0  (prefill_ssm)" --temperature 0.0
bench "13: DFlash CAP=15 T=0.6 (prefill_ssm)" --temperature 0.6
bench "14: DFlash CAP=15 T=1.0 (prefill_ssm)" --temperature 1.0
stop_server

echo ""
echo "=== Stage 27 bench complete ==="
