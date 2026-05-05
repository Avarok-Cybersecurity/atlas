#!/usr/bin/env bash
#
# Run smoke test + coherence + concurrency benchmark for all 6 Atlas models.
# Single-node models: max-batch-size=16, concs=1,2,4,8,16
# 27B Dense: conc=1 only (too slow for batched decode)
# 122B EP=2: conc=1 only (EP forces batch=1)
#
set -euo pipefail

IMAGE="${IMAGE:-atlas-gb10:latest}"
EP_IMAGE="${EP_IMAGE:-atlas-gb10:latest}"
RESULTS_DIR="/workspace/atlas/conc-bench-results"
mkdir -p "$RESULTS_DIR"
HF_CACHE="${HOME}/.cache/huggingface"
# IP of the EP-rank-1 node. Single-node default: localhost. For multi-
# node (true EP=2 across two physical machines) override via:
#   EP_RANK1_HOST=<remote-ip> bash scripts/run_conc_benchmark.sh
EP_RANK1_HOST="${EP_RANK1_HOST:-127.0.0.1}"

CONCS_FULL="1 2 4 8 16"
ISLS_FULL="128 512 1024 2048"
ISLS_27B="128 512 1024"

wait_for_server() {
    local url="${1:-http://localhost:8888}"
    local timeout="${2:-600}"
    echo "  Waiting for server at $url (timeout ${timeout}s)..."
    for i in $(seq 1 "$timeout"); do
        if curl -s --max-time 2 "$url/health" >/dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server not ready after ${timeout}s"
    return 1
}

run_model() {
    local name="$1"
    local model_id="$2"
    local container="$3"
    local extra_flags="$4"
    local isls="$5"
    local concs="$6"
    local max_batch="${7:-16}"
    local gpu_util="${8:-0.88}"

    echo ""
    echo "================================================================"
    echo "  MODEL: $name"
    echo "  ID:    $model_id"
    echo "  Batch: $max_batch | Concs: $concs | ISLs: $isls"
    echo "================================================================"

    # Stop any existing container
    sudo docker rm -f "$container" 2>/dev/null || true
    sleep 2

    # Start
    echo "  Starting container $container..."
    sudo docker run -d \
        --name "$container" \
        --gpus all \
        --ipc=host \
        --network host \
        -e RUST_LOG=warn \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        "$IMAGE" serve "$model_id" \
            --port 8888 \
            --kv-cache-dtype nvfp4 \
            --gpu-memory-utilization "$gpu_util" \
            --scheduling-policy slai \
            --max-seq-len 8192 \
            --max-batch-size "$max_batch" \
            $extra_flags

    if ! wait_for_server "http://localhost:8888" 600; then
        echo "  SKIPPED — server did not start" | tee "$RESULTS_DIR/${name}.txt"
        sudo docker logs "$container" 2>&1 | tail -30 >> "$RESULTS_DIR/${name}.txt"
        sudo docker rm -f "$container" 2>/dev/null
        return
    fi

    # Smoke test
    echo "  Running smoke test..."
    local smoke
    smoke=$(curl -s --max-time 90 http://localhost:8888/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Just the number.\"}],\"max_tokens\":8}" 2>&1)
    if echo "$smoke" | grep -q '"content"'; then
        echo "  Smoke test PASSED"
    else
        echo "  Smoke test FAILED: $smoke"
        echo "SMOKE TEST FAILED" > "$RESULTS_DIR/${name}.txt"
        sudo docker rm -f "$container" 2>/dev/null
        return
    fi

    # Coherence (don't fail sweep on test failures)
    echo "  Running coherence tests..."
    python3 scripts/dev/coherence_test.py --url http://localhost:8888 --model "$model_id" -v \
        2>&1 | tee "$RESULTS_DIR/${name}-coherence.txt" || true

    # Benchmark
    echo "  Running concurrency benchmark..."
    python3 bench/bench_concurrency.py \
        --url http://localhost:8888 \
        --model "$model_id" \
        --osl 128 --warmup 1 \
        --isls $isls \
        --concs $concs \
        2>&1 | tee "$RESULTS_DIR/${name}-bench.txt"

    # Stop
    echo "  Stopping $container..."
    sudo docker rm -f "$container" 2>/dev/null
    echo "  Done: $name"
}

echo "=== Atlas Concurrency Benchmark Sweep ==="
echo "Date: $(date)"
echo "Image: $IMAGE"
echo ""

# 1. 27B Dense — conc=1 only
run_model "27B-Dense" \
    "Kbenkhaled/Qwen3.5-27B-NVFP4" \
    "atlas-bench" \
    "" \
    "$ISLS_27B" \
    "1" \
    1 \
    0.88

# 2. VL-30B — no MTP, full concs
run_model "VL-30B" \
    "ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4" \
    "atlas-bench" \
    "" \
    "$ISLS_FULL" \
    "$CONCS_FULL" \
    16 \
    0.88

# 3. 35B MoE — MTP, full concs
run_model "35B-MoE" \
    "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4" \
    "atlas-bench" \
    "--speculative --mtp-quantization nvfp4" \
    "$ISLS_FULL" \
    "$CONCS_FULL" \
    16 \
    0.88

# 4. 80B MoE — MTP, full concs
run_model "80B-MoE" \
    "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4" \
    "atlas-bench" \
    "--speculative --mtp-quantization nvfp4" \
    "$ISLS_FULL" \
    "$CONCS_FULL" \
    16 \
    0.88

# 5. Nemotron-H 30B — no MTP, full concs
run_model "Nemotron-H" \
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
    "atlas-bench" \
    "" \
    "$ISLS_FULL" \
    "$CONCS_FULL" \
    16 \
    0.88

echo ""
echo "=== Single-node models complete ==="
echo ""

# 6. 122B EP=2 — conc=1 only (EP forces batch=1)
echo "================================================================"
echo "  MODEL: 122B EP=2"
echo "  EP forced max-batch-size=1, conc=1 only"
echo "================================================================"

sudo docker rm -f atlas-ep0 2>/dev/null || true
ssh "$EP_RANK1_HOST" "sudo docker rm -f atlas-ep1 2>/dev/null" 2>/dev/null || true
sleep 2

GPU_MEM_UTIL=0.55 IMAGE="$EP_IMAGE" bash scripts/start-ep2.sh

if ! wait_for_server "http://localhost:8888" 600; then
    echo "  SKIPPED — EP=2 server did not start" | tee "$RESULTS_DIR/122B-EP2.txt"
    sudo docker logs atlas-ep0 2>&1 | tail -30 >> "$RESULTS_DIR/122B-EP2.txt"
    sudo docker rm -f atlas-ep0 2>/dev/null
    ssh "$EP_RANK1_HOST" "sudo docker rm -f atlas-ep1 2>/dev/null" 2>/dev/null
else
    # Smoke test
    echo "  Running smoke test..."
    smoke=$(curl -s --max-time 90 http://localhost:8888/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Sehyo/Qwen3.5-122B-A10B-NVFP4","messages":[{"role":"user","content":"What is 2+2? Just the number."}],"max_tokens":8}' 2>&1)
    if echo "$smoke" | grep -q '"content"'; then
        echo "  Smoke test PASSED"
    else
        echo "  Smoke test FAILED: $smoke"
    fi

    # Coherence
    echo "  Running coherence tests..."
    python3 coherence_test.py --url http://localhost:8888 \
        --model "Sehyo/Qwen3.5-122B-A10B-NVFP4" -v \
        2>&1 | tee "$RESULTS_DIR/122B-EP2-coherence.txt" || true

    # Benchmark
    echo "  Running benchmark..."
    python3 bench_concurrency.py \
        --url http://localhost:8888 \
        --model "Sehyo/Qwen3.5-122B-A10B-NVFP4" \
        --osl 128 --warmup 1 \
        --isls 128 512 1024 2048 \
        --concs 1 \
        2>&1 | tee "$RESULTS_DIR/122B-EP2-bench.txt"

    sudo docker rm -f atlas-ep0 2>/dev/null
    ssh "$EP_RANK1_HOST" "sudo docker rm -f atlas-ep1 2>/dev/null" 2>/dev/null
    echo "  Done: 122B EP=2"
fi

echo ""
echo "=== ALL MODELS COMPLETE ==="
echo "Results in: $RESULTS_DIR/"
echo "Finished at $(date)"
