#!/usr/bin/env bash
# bench_per_model.sh — Rebuild atlas-gb10 + benchmark all models, per-model MD output.
#
# Usage:
#   bash scripts/bench_per_model.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="atlas-gb10:latest"
PORT=8888
HF_CACHE="${HOME}/.cache/huggingface"
RESULTS_DIR="$REPO_ROOT/bench-results"
mkdir -p "$RESULTS_DIR"

MOE_ISLS="128 512 1024 4096 8192"
MOE_CONCS="1 4 16"
DENSE_ISLS="128 512 1024"
DENSE_CONCS="1 4"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a /tmp/bench-sweep.log; }

wait_server() {
  local url="$1" label="$2"
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
  local n
  n=$(echo "$1" | grep -c "$2" 2>/dev/null) || n=0
  echo "$n"
}

write_model_md() {
  local outfile="$1" label="$2" model="$3" config="$4"
  local coherence_out="$5" bench_out="$6"
  local cpass cfail
  cpass=$(count_grep "$coherence_out" "PASS")
  cfail=$(count_grep "$coherence_out" "FAIL")

  cat > "$outfile" << HEADER
# ${label}

**Date:** $(date '+%Y-%m-%d')
**Hardware:** NVIDIA GB10 Grace Blackwell (119.7 GB GPU memory)
**Model:** \`${model}\`
**Config:** ${config}
**Image:** \`${IMAGE}\`
**Coherence:** ${cpass}/$((cpass + cfail)) passed
**KV Cache:** NVFP4
**Scheduler:** SLAI
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

## Coherence Test Output

\`\`\`
${coherence_out}
\`\`\`

## Concurrency Benchmark

\`\`\`
${bench_out}
\`\`\`

---
Finished at $(date '+%Y-%m-%d %H:%M:%S')
HEADER
}

run_model() {
  local outfile="$1" label="$2" model="$3" config="$4"
  shift 4
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
    coherence_out=$(python3 "$REPO_ROOT/scripts/dev/coherence_test.py" \
      --url "$url" --model "$model" -v 2>&1) || true

    log "Running bench_concurrency (ISLs=$isls, Concs=$concs)..."
    bench_out=$(python3 "$REPO_ROOT/bench/bench_concurrency.py" \
      --url "$url" --model "$model" \
      --osl 128 --warmup 1 \
      --isls $isls \
      --concs $concs \
      2>&1) || bench_out="ERROR: bench failed"
  else
    local docker_log
    docker_log=$(sudo docker logs "$container" 2>&1 | tail -20)
    coherence_out="SKIPPED — server did not start"
    bench_out="SKIPPED — server did not start\n\nDocker log tail:\n${docker_log}"
  fi

  sudo docker rm -f "$container" 2>/dev/null || true

  write_model_md "$outfile" "$label" "$model" "$config" "$coherence_out" "$bench_out"
  log "=== DONE: ${label} → $outfile ==="
  sleep 5
}

run_ep2_model() {
  local outfile="$1" label="$2" model="$3"
  local head_ip="${HEAD_IP:-127.0.0.1}"
  local worker_ip="${WORKER_IP:-127.0.0.1}"
  local master_port="29500"
  local url="http://localhost:${PORT}"

  log "=== START: ${label} (EP=2) ==="

  sudo docker rm -f atlas-ep0 2>/dev/null || true
  ssh "$worker_ip" "sudo docker rm -f atlas-ep1 2>/dev/null || true"

  sudo docker run -d \
    --name atlas-ep0 \
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
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "$IMAGE" serve "$model" \
      --rank 0 --world-size 2 \
      --master-addr "$head_ip" --master-port "$master_port" \
      --port "$PORT" \
      --max-batch-size 1 \
      --kv-cache-dtype nvfp4 \
      --gpu-memory-utilization 0.70 \
      --max-seq-len 4096 \
      --scheduling-policy slai \
      --speculative --mtp-quantization nvfp4

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
    ${IMAGE} serve ${model} \
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
    coherence_out=$(python3 "$REPO_ROOT/scripts/dev/coherence_test.py" \
      --url "$url" --model "$model" -v 2>&1) || true

    log "Running bench_concurrency (122B, 4k context)..."
    bench_out=$(python3 "$REPO_ROOT/bench/bench_concurrency.py" \
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

  write_model_md "$outfile" "$label" "$model" \
    "EP=2, NVFP4 KV, MTP K=2, max-seq-len=4096" \
    "$coherence_out" "$bench_out"
  log "=== DONE: ${label} → $outfile ==="
  sleep 5
}

# ── Build single multi-target image ─────────────────────────────────────────

log "Building ${IMAGE} (multi-target, all models)..."
sudo docker build -f docker/gb10/Dockerfile -t "$IMAGE" "$REPO_ROOT" 2>&1 | tail -5
if [ $? -ne 0 ]; then
  log "ERROR: Build failed for ${IMAGE}"
  exit 1
fi
log "Built ${IMAGE}"

# Push to worker for EP=2
log "Pushing ${IMAGE} to worker node..."
sudo docker save "$IMAGE" | ssh "${WORKER_IP:-127.0.0.1}" "sudo docker load" 2>&1 | tail -2
log "Worker image updated"

# ── Run benchmarks ───────────────────────────────────────────────────────────

log "Starting benchmark sweep..."

# 1. 27B Dense
BENCH_ISLS="$DENSE_ISLS"
BENCH_CONCS="$DENSE_CONCS"
run_model \
  "$RESULTS_DIR/27b-dense.md" \
  "Qwen3.5-27B Dense (NVFP4)" \
  "Kbenkhaled/Qwen3.5-27B-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, no MTP (hybrid SSM)"  \
  --max-seq-len 8192
BENCH_ISLS="$MOE_ISLS"
BENCH_CONCS="$MOE_CONCS"

# 2. VL-30B
run_model \
  "$RESULTS_DIR/vl-30b.md" \
  "Qwen3-VL-30B MoE Vision (NVFP4)" \
  "ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, no MTP" \
  --max-seq-len 8192

# 3. 35B MTP
run_model \
  "$RESULTS_DIR/35b-mtp.md" \
  "Qwen3.5-35B MoE (NVFP4, MTP K=2)" \
  "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, speculative MTP K=2" \
  --max-seq-len 8192 \
  --speculative \
  --mtp-quantization nvfp4

# 4. 80B MTP
run_model \
  "$RESULTS_DIR/80b-mtp.md" \
  "Qwen3-Next-80B MoE (NVFP4, MTP K=2)" \
  "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, speculative MTP K=2" \
  --max-seq-len 8192 \
  --speculative \
  --mtp-quantization nvfp4

# 5. Nemotron-H 30B
run_model \
  "$RESULTS_DIR/nemotron-30b.md" \
  "Nemotron-H 30B MoE (NVFP4)" \
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
  "NVFP4 KV cache, max-seq-len=8192, no MTP" \
  --max-seq-len 8192

# 6. 122B EP=2
if ssh "${WORKER_IP:-127.0.0.1}" "sudo docker images ${IMAGE} --format '{{.ID}}'" 2>/dev/null | grep -q .; then
  run_ep2_model \
    "$RESULTS_DIR/122b-ep2.md" \
    "Qwen3.5-122B MoE (NVFP4, EP=2, MTP K=2)" \
    "Sehyo/Qwen3.5-122B-A10B-NVFP4"
else
  log "Worker missing ${IMAGE} — skipping 122B"
  echo "> **SKIPPED** — ${IMAGE} not available on worker node" > "$RESULTS_DIR/122b-ep2.md"
fi

log "=== SWEEP COMPLETE ==="
log "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
