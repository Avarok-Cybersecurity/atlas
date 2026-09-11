#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4, 128K context x 4 sequences, MTP ON, across 2x DGX
# Spark (GB10) at EP=2. Rank 0 serves on 0.0.0.0:8888.
#
#   on dgx-00   (rank 0, master, serves :8888):  ./serve_qwen4exp_128k_mtp.sh 0
#   on gx10-9959 (rank 1, no port):              ./serve_qwen4exp_128k_mtp.sh 1
#
# Start rank 1 FIRST or at the same time; rank 0 blocks in the NCCL bootstrap
# until its peer appears.
#
# ── WHY 2 NODES ────────────────────────────────────────────────────────────
# 128K x 4 needs ~12.6 GB of KV (24 KiB/token over the 12 full-attention
# layers, x4 seqs) plus the MTP drafter's own 1-layer KV. One GB10 cannot do
# it: the weights alone are 85.6 GB of a 92.4 GB budget at util 0.76. EP=2
# halves the 512 routed experts per rank, and that is where the room comes
# from. EP-ONLY and not by choice — the qwen4_exp loader REFUSES --tp-size > 1.
#
# ── ROLLBACK=replay: WORKS, COSTS ~11% DECODE, SAVES 864 MB ────────────────
# `--ssm-rollback-mode replay` is now wired (it used to refuse every
# speculative verify). MEASURED on this exact config, 128K x 4, EP=2, util
# 0.65, DRAFTS=2, median of 4 reps:
#
#                     snapshot        replay
#   SSM MTP pools     1732 MB         868 MB      (-864 MB)
#   KV blocks         39969           42429       (+39K tokens, ~+7% context)
#   C=4 aggregate     56.65 tok/s     50.49       (-10.9%)
#   p1                0.767-0.858     0.763-0.814
#   tok_step          2.03-2.57       1.91-2.40
#   known-answer      4/4             4/4
#
# So it is CORRECT and it is NOT free. Where the 11% goes:
#
#   -4.1%  lower acceptance. The reconstruction re-runs the accepted rows
#          through the sequential chain, so the restored state is not bit-equal
#          to what the WY forward committed, and the next draft is marginally
#          off. (Direction is arguable: sequential is what spec-off decode
#          computes, so replay lands on the spec-off reference and snapshot
#          lands on the WY approximation of it.)
#   -7.0%  the reconstruction itself — `na` tokens x 36 layers of conv+GDN on
#          the critical path, where snapshot does one d2d copy per layer.
#   ~1%    the batched single-launch WY arm, which replay declines (shared
#          intermediates cannot serve concurrent sequences). MEASURED by
#          declining it under snapshot too: 55.93 vs 56.65. It is NOT the
#          cause, which is worth recording because it was the obvious suspect.
#
# The cost is therefore INTRINSIC — recomputation instead of a copy — not an
# implementation artifact to be tuned away.
#
# WHEN TO USE IT. Only when memory is the binding constraint. At bs=4 it is
# not: 864 MB buys ~7% context for ~11% decode. The saving scales with batch
# (ATLAS_EP_PROTOCOL=v2 makes `mtp_state_slots` = max_batch_size, uncapped)
# while the 11% stays roughly flat, so there is a batch size above which the
# memory is worth more than the throughput. bs=4 is well below it.
#
# ── MTP CORRECTNESS: READ THE ACCEPT COUNTERS, NOT THE OUTPUT ──────────────
# A rejected draft is harmless — the verify emits the target's own token — so
# a broken drafter produces byte-correct output at full quality while doing no
# useful work. That is exactly how the highway-row bug (fixed in 7ee130bfc)
# survived a 4/4 known-answer gate while holding p1 at 0.19. Judge this serve
# by `mtp_accept_debug` on RANK 0:
#
#     p1 ~0.85, tok_step ~2.4   healthy
#     p1 ~0.2,  tok_step ~1.3   drafts are being thrown away
#
# ── MEASURED at 32K x 4 on this build (median of 5 reps, EP=2, util 0.58) ──
#     MTP off (control)   49.0 tok/s aggregate at C=4
#     MTP on (this stack) 56.1 tok/s aggregate at C=4, p1 0.83
# 128K x 4 is a CAPACITY shape, not that throughput shape: expect lower decode
# at long context and much longer TTFT on a cold 128K prompt.
set -uo pipefail
cd "$(dirname "$0")/../.."

RANK="${1:?usage: serve_qwen4exp_128k_mtp.sh <rank 0|1>}"

MODEL_DIR="${QWEN4EXP_PATH:-/home/ms/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47}"
MASTER="${MASTER:-192.168.177.11}"          # dgx-00 = rank 0 = the box that serves
if [ "$RANK" = "0" ]; then
  BIN="${BIN:-$PWD/target/release/spark}"
  BIND="0.0.0.0"
  PORT="${PORT:-8888}"
else
  BIN="${BIN:-/home/ms/spark-pr973-ep}"
  BIND="0.0.0.0"
  PORT=0
fi

MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
NUM_SEQS="${NUM_SEQS:-4}"
SSM_CACHE_SLOTS="${SSM_CACHE_SLOTS:-32}"
DRAFTS="${DRAFTS:-2}"
# snapshot (default, wired production path) or replay. See the header: replay
# reconstructs a partial accept instead of storing per-token state snapshots.
ROLLBACK="${ROLLBACK:-snapshot}"

# 🪤 util RESERVES its whole fraction of TOTAL box memory up front, and the KV
# pool then expands to fill whatever the weights leave over. 0.65 is the
# measured-comfortable point for the 128K x 4 shape at EP=2; 0.58 is the 32K
# profiling default. Do NOT raise this without watching `free -g` UNDER LOAD:
# util 0.80 at EP=2 reached 112 GB used / 9 GB available and had to be killed,
# and over-allocation has HARD REBOOTED these boxes. ~21 GB of n-gram page
# cache lives OUTSIDE the engine budget and is invisible to the pledge.
GPU_UTIL="${GPU_UTIL:-0.65}"

if [ ! -e "$MODEL_DIR/config.json" ]; then
  echo "checkpoint not found on $(hostname): $MODEL_DIR" >&2; exit 1
fi
if [ ! -x "$BIN" ]; then
  echo "binary not found on $(hostname): $BIN" >&2; exit 1
fi

# ── Fabric: the f1 rail carries 192.168.177.0/24 on these two boxes. ────────
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_NVLS_ENABLE=0
export NCCL_PROTO=Simple
export NCCL_CROSS_NIC=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export LD_LIBRARY_PATH="/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export RUST_LOG="${RUST_LOG:-info}"

# ── Long-context caps. These fail MID-PREFILL, each at its own length, so all
# three must clear 128K or a long prompt dies partway in. ───────────────────
#
#   ATLAS_QSA_MAX_TOKENS  must be STRICTLY GREATER than max-seq-len: the guard
#                         is `tokens <= cap` and the generated token makes it
#                         max_seq_len + 1. Costs cap * indexer_head_dim(128) * 2
#                         bytes per QSA layer.
#   ATLAS_PLE_MAX_TOKENS  bounds the prefill CHUNK, not the prompt. 9000 covers
#                         any prompt length; do NOT scale it with context
#                         (it costs tokens*10240*14 bytes).
#   ATLAS_PLE_CACHE_SLOTS n-gram row cache; cheap.
export ATLAS_QSA_MAX_TOKENS="${ATLAS_QSA_MAX_TOKENS:-$((MAX_SEQ_LEN + 4096))}"
export ATLAS_PLE_MAX_TOKENS="${ATLAS_PLE_MAX_TOKENS:-9000}"
export ATLAS_PLE_CACHE_SLOTS="${ATLAS_PLE_CACHE_SLOTS:-1048576}"

export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0
export ATLAS_NO_HW_PRECHECK=1
export ATLAS_QWEN4EXP_BF16_GDN="${ATLAS_QWEN4EXP_BF16_GDN:-0}"

# ── EP wire protocol. v1 FORCES max_batch_size=1 at world_size > 1, so v2 is
# what makes C>1 possible at all. BOTH RANKS MUST AGREE: the preamble changes
# the wire shape, so a one-sided setting desynchronises the command stream.
# Exported before the rank split for exactly that reason.
export ATLAS_EP_PROTOCOL="${ATLAS_EP_PROTOCOL:-v2}"

# ── Cross-sequence batched verify. TWO gates, both needed, both on BOTH ranks.
# ATLAS_HC_BATCH_VERIFY=0 does NOT merely disable cross-sequence batching — it
# gates highway models out of `supports_verify_layout` entirely, so MTP loses
# its benefit even at C=1 (measured 27.8 tok/s against 41.6 with it on).
export ATLAS_HC_BATCH_VERIFY="${ATLAS_HC_BATCH_VERIFY:-1}"
export ATLAS_MTP_EP_BATCH_VERIFY="${ATLAS_MTP_EP_BATCH_VERIFY:-1}"

# ── The #972 gates. Per-process OnceLocks with no cross-rank agreement, so if
# the two sides disagree the head and worker take different arms with
# different collective counts — an NCCL spin. Set explicitly on BOTH ranks.
export ATLAS_QWEN4EXP_MTP_HC_BATCHED="${ATLAS_QWEN4EXP_MTP_HC_BATCHED:-1}"
export ATLAS_VERIFY_ROW_PROJ="${ATLAS_VERIFY_ROW_PROJ:-1}"

# ── MTP. Priority #1 for this config. ──────────────────────────────────────
export ATLAS_QWEN4EXP_MTP=1 ATLAS_QWEN4EXP_MTP_VERIFY=1
# Speculation is INERT inside <think> without this, and thinking is ON here,
# so without it MTP would do nothing for most of a reasoning turn.
export ATLAS_DFLASH_SPEC_THINK=1
export ATLAS_NO_THINKENDED_GPU_ARGMAX=1
# Opt past the EP refusal in probe_mtp.rs: the draft MoE is replicated per
# rank, so drafts agree only if the all-reduce leaves every rank identical.
# If they drift the target still VERIFIES the draft — acceptance drops, output
# does not corrupt.
export ATLAS_EP_MTP=1
# The only liveness proof that MTP is doing work. Counters live on RANK 0.
export ATLAS_MTP_ACCEPT_DEBUG=1

echo "Qwen3.8-Flash-Next NVFP4  EP=2 (TP=1)  rank=$RANK host=$(hostname)"
echo "  ctx=$MAX_SEQ_LEN seqs=$NUM_SEQS util=$GPU_UTIL ssm_slots=$SSM_CACHE_SLOTS"
echo "  MTP=on drafts=$DRAFTS  rollback=$ROLLBACK"
echo "  thinking=on reasoning_effort=low  prefix-cache=on"
echo "  qsa_cap=$ATLAS_QSA_MAX_TOKENS (> ctx) ple_chunk=$ATLAS_PLE_MAX_TOKENS"
echo "  bind=$BIND:$PORT  master=$MASTER"
echo "  bin=$(sha256sum "$BIN" | cut -c1-16)"
free -g | sed -n 2p

exec "$BIN" serve \
  --model-from-path "$MODEL_DIR" \
  --model-name qwen4exp-nvfp4 --kernel-target qwen3.8-flash-next \
  --rank "$RANK" --world-size 2 \
  --tp-size 1 --ep-size 2 \
  --master-addr "$MASTER" --master-port 29500 \
  --bind "$BIND" --port "$PORT" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-num-seqs "$NUM_SEQS" --max-batch-size "$NUM_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype bf16 \
  --ssm-cache-slots "$SSM_CACHE_SLOTS" \
  --ssm-rollback-mode "$ROLLBACK" \
  --max-prefill-tokens 2048 \
  --request-timeout 1800 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  --speculative --num-drafts "$DRAFTS" --mtp-gate force \
  --default-chat-template-kwargs '{"reasoning_effort":"low","enable_thinking":true}' \
  ${EXTRA_ARGS:-} \
  "${@:2}"
