#!/bin/bash
# Comprehensive model sweep across both DGX Spark nodes
# Usage: bash scripts/sweep_all_models.sh

set -e

IMAGE="atlas-gb10:latest"
NODE1="localhost"
NODE2="${NODE2:-127.0.0.1}"
PORT_A=8888
PORT_B=8889
PROMPT="What is the capital of France? Answer in one sentence."
MAX_TOKENS=30
TEMP=0.1

test_model() {
    local name="$1"
    local hf_id="$2"
    local extra_args="$3"
    local host="$4"
    local port="$5"
    local node_name="$6"

    echo "=== Testing: $name ($node_name:$port) ==="

    # Start container
    if [ "$host" = "localhost" ]; then
        docker run -d --name "sweep-${name}" --gpus all --ipc=host -p ${port}:8888 \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            $IMAGE serve "$hf_id" --max-seq-len 16384 $extra_args 2>/dev/null
    else
        ssh $host "docker run -d --name sweep-${name} --gpus all --ipc=host -p ${port}:8888 \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            $IMAGE serve '$hf_id' --max-seq-len 16384 $extra_args" 2>/dev/null
    fi

    # Wait for loading
    local connect_host="$host"
    [ "$host" = "localhost" ] && connect_host="127.0.0.1"

    local ready=0
    for i in $(seq 1 60); do
        sleep 5
        if curl -s --connect-timeout 2 "http://${connect_host}:${port}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1,\"temperature\":0}" \
            2>/dev/null | grep -q "choices"; then
            ready=1
            break
        fi
    done

    if [ "$ready" = "0" ]; then
        echo "TIMEOUT loading $name"
        # Cleanup
        if [ "$host" = "localhost" ]; then
            docker rm -f "sweep-${name}" 2>/dev/null
        else
            ssh $host "docker rm -f sweep-${name}" 2>/dev/null
        fi
        echo "$name | TIMEOUT | - | -"
        return
    fi

    # Run quality test
    local result=$(curl -s --connect-timeout 10 "http://${connect_host}:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":$TEMP}" 2>/dev/null)

    local content=$(echo "$result" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:80])" 2>/dev/null || echo "ERROR")
    local tps=$(echo "$result" | python3 -c "import json,sys; r=json.load(sys.stdin); print(f\"{r['usage']['response_token/s']:.1f}\")" 2>/dev/null || echo "?")

    local quality="FAIL"
    echo "$content" | grep -qi "paris" && quality="PASS"

    echo "$name | $quality | $tps tok/s | $content"

    # Cleanup
    if [ "$host" = "localhost" ]; then
        docker rm -f "sweep-${name}" 2>/dev/null
    else
        ssh $host "docker rm -f sweep-${name}" 2>/dev/null
    fi
}

echo "=========================================="
echo "Atlas Model Sweep — $(date)"
echo "=========================================="
echo ""

# Clean up any existing containers
docker rm -f $(docker ps -aq --filter "name=sweep-") 2>/dev/null || true
ssh $NODE2 "docker rm -f \$(docker ps -aq --filter 'name=sweep-') 2>/dev/null" 2>/dev/null || true

echo "--- Phase 1: Parallel testing on both nodes ---"
echo ""

# Node 1 models (run sequentially on node 1)
# Node 2 models (run sequentially on node 2)

RESULTS_FILE="/tmp/sweep_results.txt"
> $RESULTS_FILE

# Run models in parallel batches (one per node)
run_batch() {
    local n1_name="$1" n1_hf="$2" n1_args="$3"
    local n2_name="$4" n2_hf="$5" n2_args="$6"

    # Start both in parallel
    test_model "$n1_name" "$n1_hf" "$n1_args" localhost $PORT_A "node1" >> $RESULTS_FILE &
    local pid1=$!

    if [ -n "$n2_name" ]; then
        test_model "$n2_name" "$n2_hf" "$n2_args" $NODE2 $PORT_A "node2" >> $RESULTS_FILE &
        local pid2=$!
        wait $pid1 $pid2
    else
        wait $pid1
    fi
}

echo "Batch 1: Qwen3.5-35B (node1) + Nemotron Nano (node2)"
run_batch \
    "qwen35-35b-fp8" "Sehyo/Qwen3.5-35B-A3B-NVFP4" "--kv-cache-dtype fp8" \
    "nemotron-nano-fp8" "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" "--kv-cache-dtype fp8"

echo "Batch 2: Qwen3.5-35B MTP (node1) + Gemma 31B (node2)"
run_batch \
    "qwen35-35b-mtp" "Sehyo/Qwen3.5-35B-A3B-NVFP4" "--kv-cache-dtype fp8 --speculative --mtp-quantization nvfp4" \
    "gemma-31b" "nvidia/Gemma-4-31B-IT-NVFP4" "--kv-cache-dtype fp8 --max-batch-size 2"

echo "Batch 3: Gemma 26B MoE (node1) + Qwen3.5-27B (node2)"
run_batch \
    "gemma-26b-moe" "bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16" "" \
    "qwen35-27b" "Kbenkhaled/Qwen3.5-27B-NVFP4" "--kv-cache-dtype fp8"

echo "Batch 4: Qwen3.5-122B (node1) + Mistral Small 4 (node2)"
run_batch \
    "qwen35-122b" "Sehyo/Qwen3.5-122B-A10B-NVFP4" "--kv-cache-dtype fp8 --kv-high-precision-layers 2 --max-batch-size 1 --max-prefill-tokens 2048 --oom-guard-mb 512" \
    "mistral-small4" "mistralai/Mistral-Small-4-119B-2603-NVFP4" ""

echo "Batch 5: Nemotron Super 120B (node1) + Qwen3-Next 80B (node2)"
run_batch \
    "nemotron-super" "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" "--kv-cache-dtype fp8 --kv-high-precision-layers 2" \
    "qwen3-next-80b" "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4" "--kv-cache-dtype fp8"

echo "Batch 6: Turbo variants"
run_batch \
    "qwen35-35b-turbo4" "Sehyo/Qwen3.5-35B-A3B-NVFP4" "--kv-cache-dtype turbo4" \
    "nemotron-turbo3" "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" "--kv-cache-dtype turbo3"

echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="
sort $RESULTS_FILE
