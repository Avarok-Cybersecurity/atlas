#!/usr/bin/env bash
# run_full_benchmark.sh — Full model sweep: coherence + concurrency benchmark
#
# Runs all GB10 models with atlas-gb10:latest, writes BENCHMARK_RESULTS.md
#
# Usage:
#   bash scripts/run_full_benchmark.sh [--quick]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${IMAGE:-atlas-gb10:latest}"
EP_IMAGE="${EP_IMAGE:-atlas-gb10:latest}"
PORT=8888
OUTPUT="$REPO_ROOT/BENCHMARK_RESULTS.md"
QUICK="${1:-}"
HF_CACHE="${HOME}/.cache/huggingface"

if [ "$QUICK" = "--quick" ]; then
  MOE_ISLS="128 1024 4096"
  MOE_CONCS="1 4"
  DENSE_ISLS="128 512 1024"
  DENSE_CONCS="1 4"
else
  # MoE models are fast enough to sweep large ISL and high concurrency
  MOE_ISLS="128 512 1024 4096 8192"
  MOE_CONCS="1 4 16"
  # Dense 27B is ~15 tok/s — cap ISL at 1024, concurrency at 4
  DENSE_ISLS="128 512 1024"
  DENSE_CONCS="1 4"
fi

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/bench-sweep.log; }

wait_server() {
  local url="$1"
  local label="$2"
  log "Waiting for $label..."
  for i in $(seq 1 120); do
    if curl -sf "${url}/health" > /dev/null 2>&1; then
      log "$label ready after ${i}×5s"
      return 0
    fi
    sleep 5
  done
  log "ERROR: $label not ready after 600s"
  return 1
}

count_grep() {
  # Count matches without failing on 0 matches.
  # grep -c exits 1 on no matches but still prints "0"; capture it first
  # to avoid || echo 0 appending a second "0" to stdout.
  local n
  n=$(echo "$1" | grep -c "$2" 2>/dev/null) || n=0
  echo "$n"
}

append_model_results() {
  local label="$1"
  local model="$2"
  local config="$3"
  local coherence_out="$4"
  local bench_out="$5"
  local cpass
  local cfail
  cpass=$(count_grep "$coherence_out" "PASS")
  cfail=$(count_grep "$coherence_out" "FAIL")

  {
    echo ""
    echo "---"
    echo ""
    echo "## ${label}"
    echo ""
    echo "**Model:** \`${model}\`  "
    echo "**Config:** ${config}  "
    echo "**Coherence:** ${cpass}/$((cpass + cfail)) passed"
    echo ""
    echo "### Coherence Test Output"
    echo ""
    echo "\`\`\`"
    echo "${coherence_out}"
    echo "\`\`\`"
    echo ""
    echo "### Concurrency Benchmark"
    echo ""
    echo "\`\`\`"
    echo "${bench_out}"
    echo "\`\`\`"
    echo ""
  } >> "$OUTPUT"
}

# ── Single-node model benchmark ───────────────────────────────────────────────
# Globals set by caller before each invocation:
#   BENCH_ISLS  — space-separated ISL values (default: $MOE_ISLS)
#   BENCH_CONCS — space-separated concurrency values (default: $MOE_CONCS)
run_model() {
  local label="$1"
  local model="$2"
  local config="$3"
  shift 3
  local extra_args=("$@")
  local isls="${BENCH_ISLS:-$MOE_ISLS}"
  local concs="${BENCH_CONCS:-$MOE_CONCS}"
  local container="atlas-bench"
  local url="http://localhost:${PORT}"

  log "=== START: ${label} ==="

  sudo docker rm -f "$container" 2>/dev/null || true

  sudo docker run -d \
    --name "$container" \
    --gpus all \
    --ipc=host \
    --network host \
    -e RUST_LOG=warn \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "$IMAGE" serve "$model" \
      --port "$PORT" \
      --kv-cache-dtype nvfp4 \
      --gpu-memory-utilization 0.88 \
      --scheduling-policy slai \
      "${extra_args[@]+"${extra_args[@]}"}"

  local coherence_out bench_out
  if wait_server "$url" "$label"; then
    log "Running coherence tests..."
    coherence_out=$(python3 "$REPO_ROOT/coherence_test.py" \
      --url "$url" --model "$model" -v 2>&1) || true

    log "Running bench_concurrency (ISLs=$isls, Concs=$concs)..."
    bench_out=$(python3 "$REPO_ROOT/bench_concurrency.py" \
      --url "$url" --model "$model" \
      --osl 128 --warmup 1 \
      --isls $isls \
      --concs $concs \
      2>&1) || bench_out="ERROR: bench failed"
  else
    log "Server failed to start — skipping tests"
    local docker_log
    docker_log=$(sudo docker logs "$container" 2>&1 | tail -20)
    coherence_out="SKIPPED — server did not start"
    bench_out="SKIPPED — server did not start\n\nDocker log tail:\n${docker_log}"
  fi

  sudo docker rm -f "$container" 2>/dev/null || true

  append_model_results "$label" "$model" "$config" "$coherence_out" "$bench_out"
  log "=== DONE: ${label} ==="
  sleep 5
}

# ── EP=2 model benchmark (122B) ───────────────────────────────────────────────
run_ep2_model() {
  local label="$1"
  local model="$2"
  local head_ip="${HEAD_IP:-127.0.0.1}"
  local worker_ip="${WORKER_IP:-127.0.0.1}"
  local master_port="29500"
  local url="http://localhost:${PORT}"

  log "=== START: ${label} (EP=2) ==="

  sudo docker rm -f atlas-ep0 2>/dev/null || true
  ssh "$worker_ip" "sudo docker rm -f atlas-ep1 2>/dev/null || true"

  # Start rank 0 (head)
  sudo docker run -d \
    --name atlas-ep0 \
    --gpus all \
    --ipc=host \
    --network host \
    --device=/dev/infiniband \
    --cap-add=IPC_LOCK \
    --ulimit memlock=-1 \
    -e RUST_LOG=warn \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA=rocep1s0f0 \
    -e NCCL_IB_GID_INDEX=0 \
    -e NCCL_NET_GDR_LEVEL=0 \
    -e NCCL_PROTO=Simple \
    -e NCCL_ALGO=Ring \
    -e NCCL_MIN_NCHANNELS=1 \
    -e NCCL_MAX_NCHANNELS=2 \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "$EP_IMAGE" serve "$model" \
      --rank 0 --world-size 2 \
      --master-addr "$head_ip" --master-port "$master_port" \
      --port "$PORT" \
      --max-batch-size 1 \
      --kv-cache-dtype nvfp4 \
      --gpu-memory-utilization 0.70 \
      --max-seq-len 4096 \
      --scheduling-policy slai \
      --speculative --mtp-quantization nvfp4

  # Start rank 1 (worker)
  ssh "$worker_ip" "sudo docker run -d \
    --name atlas-ep1 \
    --gpus all --ipc=host --network host \
    --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 \
    -e RUST_LOG=warn \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA=rocep1s0f0 \
    -e NCCL_IB_GID_INDEX=0 \
    -e NCCL_NET_GDR_LEVEL=0 \
    -e NCCL_PROTO=Simple \
    -e NCCL_ALGO=Ring \
    -e NCCL_MIN_NCHANNELS=1 \
    -e NCCL_MAX_NCHANNELS=2 \
    -v \"\${HOME}/.cache/huggingface:/root/.cache/huggingface\" \
    ${EP_IMAGE} serve ${model} \
      --rank 1 --world-size 2 \
      --master-addr ${head_ip} --master-port ${master_port} \
      --port 0 \
      --max-batch-size 1 \
      --kv-cache-dtype nvfp4 \
      --gpu-memory-utilization 0.70 \
      --max-seq-len 4096 \
      --speculative --mtp-quantization nvfp4"

  local coherence_out bench_out
  if wait_server "$url" "$label"; then
    log "Running coherence tests..."
    coherence_out=$(python3 "$REPO_ROOT/coherence_test.py" \
      --url "$url" --model "$model" -v 2>&1) || true

    log "Running bench_concurrency (122B, 4k context)..."
    bench_out=$(python3 "$REPO_ROOT/bench_concurrency.py" \
      --url "$url" --model "$model" \
      --osl 128 --warmup 1 \
      --isls 128 512 1024 2048 \
      --concs 1 2 4 \
      2>&1) || bench_out="ERROR: bench failed"
  else
    local docker_log
    docker_log=$(sudo docker logs atlas-ep0 2>&1 | tail -20)
    coherence_out="SKIPPED — server did not start"
    bench_out="SKIPPED — server did not start\n\nDocker log tail:\n${docker_log}"
  fi

  sudo docker rm -f atlas-ep0 2>/dev/null || true
  ssh "$worker_ip" "sudo docker rm -f atlas-ep1 2>/dev/null || true"

  append_model_results "$label" "$model" \
    "EP=2, NVFP4 KV, MTP K=2, max-seq-len=4096" \
    "$coherence_out" "$bench_out"
  log "=== DONE: ${label} ==="
  sleep 5
}

# ── Initialize markdown ───────────────────────────────────────────────────────
log "Initializing $OUTPUT"
cat > "$OUTPUT" << HEADER
# Atlas GB10 — Full Model Benchmark Results

**Date:** $(date '+%Y-%m-%d')
**Hardware:** 2× NVIDIA GB10 Grace Blackwell (119.7 GB GPU memory each)
**Image:** \`atlas-gb10:latest\` (ATLAS_TARGET_MODEL=*)
**KV Cache:** NVFP4 (all models)
**Scheduler:** SLAI (SLO-aware: shortest-prompt-first prefill, decode-priority near TBT deadline)
**Benchmark:** count-prompt mode, OSL=128, warmup=1

## Metric Definitions

| Metric | Description |
|--------|-------------|
| TTFT   | Client Time To First Token (prefill latency), ms |
| TPOT   | Client Time Per Output Token (decode inter-token), ms |
| E2E    | Client end-to-end latency (start → last token), ms |
| sTTFT  | Server TTFT (server-side, excludes network RTT), ms |
| sTPS   | Server decode throughput (tok/s per sequence) |
| Tput   | Aggregate output tok/s across concurrent batch |

All latency metrics: **p50 / p90 / p99**.

HEADER

log "Starting full benchmark sweep..."

# ── 1. Qwen3.5-27B Dense ─────────────────────────────────────────────────────
# Dense model: ~15 tok/s — cap at ISL=1024, Conc=4 to keep runtime reasonable
BENCH_ISLS="$DENSE_ISLS"
BENCH_CONCS="$DENSE_CONCS"
run_model \
  "Qwen3.5-27B Dense (NVFP4)" \
  "Kbenkhaled/Qwen3.5-27B-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, no MTP (hybrid SSM), ISL≤1024" \
  --max-seq-len 8192
BENCH_ISLS="$MOE_ISLS"
BENCH_CONCS="$MOE_CONCS"

# ── 2. Qwen3-VL-30B ──────────────────────────────────────────────────────────
run_model \
  "Qwen3-VL-30B MoE Vision (NVFP4)" \
  "ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, no MTP (no MTP weights in this checkpoint)" \
  --max-seq-len 8192

# ── 3. Qwen3.5-35B ───────────────────────────────────────────────────────────
run_model \
  "Qwen3.5-35B MoE (NVFP4, MTP K=2)" \
  "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, speculative MTP K=2" \
  --max-seq-len 8192 \
  --speculative \
  --mtp-quantization nvfp4

# ── 4. Qwen3-Next-80B ────────────────────────────────────────────────────────
run_model \
  "Qwen3-Next-80B MoE (NVFP4, MTP K=2)" \
  "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, speculative MTP K=2" \
  --max-seq-len 8192 \
  --speculative \
  --mtp-quantization nvfp4

# ── 5. Nemotron-H 30B ────────────────────────────────────────────────────────
run_model \
  "Nemotron-H 30B MoE (NVFP4)" \
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, no MTP" \
  --max-seq-len 8192

# ── 6. Qwen3.5-122B EP=2 ─────────────────────────────────────────────────────
# atlas-122b:latest (16:52, head) pushed to worker; use it on both nodes.
log "Checking worker node image..."
if ssh ${WORKER_IP:-127.0.0.1} "sudo docker images ${EP_IMAGE} --format '{{.ID}}'" 2>/dev/null | grep -q .; then
  log "Worker has ${EP_IMAGE}, running 122B EP=2"
  run_ep2_model \
    "Qwen3.5-122B MoE (NVFP4, EP=2, MTP K=2)" \
    "Sehyo/Qwen3.5-122B-A10B-NVFP4"
else
  log "Worker missing ${EP_IMAGE} — skipping 122B"
  {
    echo ""
    echo "---"
    echo ""
    echo "## Qwen3.5-122B MoE (NVFP4, EP=2, MTP K=2)"
    echo ""
    echo "> **SKIPPED** — ${EP_IMAGE} not available on worker node ${WORKER_IP:-127.0.0.1}"
    echo "> Push with: \`sudo docker save ${EP_IMAGE} | ssh ${WORKER_IP:-127.0.0.1} sudo docker load\`"
    echo ""
  } >> "$OUTPUT"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
{
  echo ""
  echo "---"
  echo ""
  echo "## Sweep Complete"
  echo ""
  echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$OUTPUT"

log "=== SWEEP COMPLETE ==="
log "Results: $OUTPUT"
