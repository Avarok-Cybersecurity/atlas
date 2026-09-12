#!/usr/bin/env bash
# Qwen3.8-Flash-Next NATIVE EXL3 with MTP and concurrency, 1 node or EP=2.
#
#   single node :  ./serve_exl3_mtp.sh
#   EP=2        :  WORLD=2 ./serve_exl3_mtp.sh 1   (on gx10-9959, start FIRST)
#                  WORLD=2 ./serve_exl3_mtp.sh 0   (on dgx-00, serves :8889)
#
# The existing `.research/exl3_decode_perf/serve_exl3.sh` is a C=1 decode
# PROFILING recipe: max-num-seqs 1, no MTP, no batched verify. This one serves.
#
# ── WHY 2.05bpw ────────────────────────────────────────────────────────────
# 59 GB on disk against 101 for 4.05, so roughly half the resident footprint —
# which is the whole appeal: NVFP4 at EP=2 rations 14.6 GB of KV out of a
# 79 GB budget, and this should leave far more. bits=2.05, head_bits=4,
# mtp_bits=2, codebook mul1. `exl3_gemv_serves_k` covers (2..=4), so K=2
# experts and the K=4 head are both in the ladder.
#
# ── 🪤 QSA CAP ─────────────────────────────────────────────────────────────
# STRICTLY greater than max-seq-len: the guard is `tokens <= cap` and the
# generated token makes it max_seq_len + 1. The older EXL3 script set cap ==
# max-seq-len (32768/32768), which fails on the LAST token of a full-context
# request rather than at boot.
#
# ── EP=2 STATUS: WIRED, NEVER RUN ──────────────────────────────────────────
# Three places say native EXL3 is EP-aware, and no measurement says it works:
#   * the loader shards experts by `config.local_expert_range()`
#     (`weight_loader/qwen4_exp/ffn.rs`), and the DRAFT module deliberately
#     replicates its experts on every rank;
#   * `forward_exl3_decode` carries `local_start`/`num_local` from the expert
#     tables and `all_reduce_async`s when `ep_world_size > 1`;
#   * the MTP module is SHARED with the main layers' `NativeExl3`, not refused.
# The only stated incompatibility is EXL3 native MoE x CUTLASS grouped.
# Treat a first EP=2 boot as UNPROVEN: read the accept counters and the
# known-answer probes, not the throughput.
set -uo pipefail
cd "$(dirname "$0")/../.."

RANK="${1:-0}"
WORLD="${WORLD:-1}"
CKPT="${EXL3_CKPT:-/tank/exl3-ckpt/qwen38-flash-next-2.05bpw}"
MASTER="${MASTER:-192.168.177.11}"
PORT="${PORT:-8889}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
NUM_SEQS="${NUM_SEQS:-4}"
DRAFTS="${DRAFTS:-2}"
MTP="${MTP:-1}"
# 2.05bpw is roughly half of 4.05's footprint; 0.72 was tuned for 4.05 on one
# node. Watch `free -g` UNDER LOAD before raising it — over-allocation has
# hard-rebooted these boxes.
GPU_UTIL="${GPU_UTIL:-0.72}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"

if [ "$WORLD" = "1" ]; then
  BIN="${BIN:-$PWD/target/release/spark}"
  BIND="127.0.0.1"
elif [ "$RANK" = "0" ]; then
  BIN="${BIN:-$PWD/target/release/spark}"
  BIND="0.0.0.0"
else
  BIN="${BIN:-/home/ms/spark-pr973-ep}"
  BIND="0.0.0.0"
  PORT=0
fi

[ -e "$CKPT/config.json" ] || { echo "checkpoint not found on $(hostname): $CKPT" >&2; exit 1; }
[ -x "$BIN" ] || { echo "binary not found on $(hostname): $BIN" >&2; exit 1; }

# ── Native EXL3: MoE + dense both packed-trellis. ──────────────────────────
export ATLAS_EXL3_NATIVE=1 ATLAS_EXL3_NATIVE_MOE=1 ATLAS_EXL3_NATIVE_DENSE=1
export ATLAS_PLE_MAX_TOKENS="${ATLAS_PLE_MAX_TOKENS:-9216}"
export ATLAS_PLE_CACHE_SLOTS="${ATLAS_PLE_CACHE_SLOTS:-4194304}"
export ATLAS_QSA_MAX_TOKENS="${ATLAS_QSA_MAX_TOKENS:-$((MAX_SEQ_LEN + 4096))}"
export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0 ATLAS_NO_HW_PRECHECK=1
export RUST_LOG="${RUST_LOG:-info}"
export LD_LIBRARY_PATH="/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ── Cross-sequence batched verify (quant-agnostic; see the qwen4exp series) ──
# ATLAS_HC_BATCH_VERIFY=0 does NOT merely disable cross-sequence batching — it
# gates highway models out of `supports_verify_layout` entirely, costing MTP
# its benefit even at C=1.
export ATLAS_HC_BATCH_VERIFY="${ATLAS_HC_BATCH_VERIFY:-1}"
export ATLAS_QWEN4EXP_MTP_HC_BATCHED="${ATLAS_QWEN4EXP_MTP_HC_BATCHED:-1}"
export ATLAS_VERIFY_ROW_PROJ="${ATLAS_VERIFY_ROW_PROJ:-1}"

SPEC_ARGS=""
if [ "$MTP" = "1" ]; then
  export ATLAS_QWEN4EXP_MTP=1 ATLAS_QWEN4EXP_MTP_VERIFY=1
  export ATLAS_DFLASH_SPEC_THINK=1 ATLAS_NO_THINKENDED_GPU_ARGMAX=1
  export ATLAS_MTP_ACCEPT_DEBUG=1
  SPEC_ARGS="--speculative --num-drafts ${DRAFTS} --mtp-gate force"
fi

# One variable for the whole topology, rather than two half-built ones.
TOPO_ARGS="--world-size 1"
if [ "$WORLD" != "1" ]; then
  export ATLAS_EP_PROTOCOL="${ATLAS_EP_PROTOCOL:-v2}"
  export ATLAS_MTP_EP_BATCH_VERIFY="${ATLAS_MTP_EP_BATCH_VERIFY:-1}"
  [ "$MTP" = "1" ] && export ATLAS_EP_MTP=1
  export NCCL_IB_DISABLE=0 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
  export NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_NVLS_ENABLE=0 NCCL_PROTO=Simple
  export NCCL_CROSS_NIC=1 NCCL_IB_QPS_PER_CONNECTION=4 NCCL_IB_SPLIT_DATA_ON_QPS=1
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  TOPO_ARGS="--rank $RANK --world-size 2 --tp-size 1 --ep-size 2 --master-addr $MASTER --master-port 29501"
fi

echo "EXL3 native $(basename "$CKPT")  world=$WORLD rank=$RANK host=$(hostname)"
echo "  ctx=$MAX_SEQ_LEN seqs=$NUM_SEQS util=$GPU_UTIL mtp=$MTP drafts=$DRAFTS chunk=$PREFILL_CHUNK"
echo "  qsa_cap=$ATLAS_QSA_MAX_TOKENS (> ctx)  bind=$BIND:$PORT"
free -g | sed -n 2p

exec "$BIN" serve \
  --model-from-path "$CKPT" \
  --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next \
  $TOPO_ARGS \
  --bind "$BIND" --port "$PORT" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-num-seqs "$NUM_SEQS" --max-batch-size "$NUM_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype bf16 \
  --ssm-cache-slots "${SSM_CACHE_SLOTS:-16}" \
  --max-prefill-tokens "$PREFILL_CHUNK" \
  --request-timeout 1800 --fast-load-prefetch-shards \
  --enable-prefix-caching \
  $SPEC_ARGS \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  ${EXTRA_ARGS:-} "${@:2}"
