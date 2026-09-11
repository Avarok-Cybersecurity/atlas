#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 across 2x DGX Spark (GB10) at EP=2 (TP=1), on the
# PR #973 stack plus the ported multi-rank prefix-cache chain
# (branch `pr973-ep-prefix-cache`).
#
# WHY 2 NODES: 128K context x 4 sequences needs ~14 GB of KV (24 KiB/token over
# 12 full-attention layers, x4 seqs, plus the MTP drafter's 1-layer KV). One
# GB10 cannot do it — the weights are 85.6 GB of a 92.4 GB budget at util 0.76,
# leaving 1.9 GB, and util 0.80 THRASHES this box (~23 GB of n-gram page cache
# lives outside the engine budget; measured decode 0.3 tok/s, load >20).
# EP=2 halves the expert weights per rank, which is where the room comes from.
#
#   on dgx-00  (rank 0, master, serves :8892, TUI here):  ./serve_pr973_ep2.sh 0
#   on gx10-9959 (rank 1, no port):                       ./serve_pr973_ep2.sh 1
#
# Start rank 1 FIRST or at the same time; rank 0 blocks in the NCCL bootstrap
# until its peer appears.
#
# EP-ONLY, and not by choice: the qwen4_exp weight loader REFUSES --tp-size > 1
# ("TP is not supported by the qwen4_exp weight loader. Run with --tp-size 1
# (EP-only)"), pointing at `crate::tp_shard::slice_for_rank` and
# `weight_loader/minimax.rs` as the work needed to add it. That is fine here:
# EP is the dimension that splits the 512 routed experts, which is where the
# memory actually is. It also means the "pure TP admits an unverified MTP path"
# hole in probe_mtp.rs is unreachable on this model — the loader stops first.
#
# ONE Atlas instance at a time per box: --gpu-memory-utilization RESERVES its
# whole fraction up front. Watch `free -g`, NOT nvidia-smi — this is a UNIFIED
# memory box and nvidia-smi is blind to the carveout. Over-allocation has hard
# rebooted these machines.
#
# ── STATUS: NOT YET VALIDATED AT 2 RANKS ────────────────────────────────────
#
# Two things here are explicitly unproven and must be READ OFF THE LOG, not
# assumed:
#
#  1. ATLAS_EP_MTP=1 opts past a deliberate refusal in
#     `weight_loader/qwen4_exp/probe_mtp.rs`. The draft MoE is replicated per
#     rank, so drafts agree only if the main model's all-reduce leaves every
#     rank bit-identical. If they drift, the target still VERIFIES the draft —
#     you lose acceptance, you do not corrupt output. Proof is the accept line
#     on RANK 0: `mtp_accept_debug ... p1=... mean_na=... tok_step=...`.
#     Single-node reference on this build: p1=0.930 mean_na=1.664 tok_step=2.664.
#     p1 near 0 => drafts diverge under EP; serve with MTP=0 instead.
#
#  2. Prefix caching multi-rank. The four ported commits make retirement
#     symmetric, allocate snapshot slots by lowest-free-index, agree the FINAL
#     anchor decision across ranks, and name the gate on a disagreement. The
#     agreement is the backstop: any residual divergence becomes a DECLINED
#     anchor (cold prefill), not an NCCL hang. Watch for
#     `Marconi anchor DISAGREES across ranks` — occasional is the backstop
#     working; every request means warm restore is effectively off.
#
# A hang with both GPUs at ~96% util / ~13 W and no `Done:` line is the NCCL
# spin. First suspect is an ENV MISMATCH between the two nodes, not the model:
# ATLAS_QWEN4EXP_MTP_HC_ATTN_ROWS / _QKV / ATLAS_HC_DECODE_ROWS are per-process
# OnceLocks with no cross-rank agreement, so if the two sides disagree the head
# and worker take different arms with different collective counts. This script
# sets every one of them explicitly on BOTH ranks for that reason — do not
# override one side only.
set -uo pipefail
cd "$(dirname "$0")/../.."

RANK="${1:?usage: serve_pr973_ep2.sh <rank 0|1>}"

# Same absolute path on both boxes (verified: 24 files, 124 GB each).
MODEL_DIR="${QWEN4EXP_PATH:-/home/ms/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47}"
# rank 0 = dgx-00 = 192.168.177.11 (the box with the terminal, so the TUI is here)
MASTER="${MASTER:-192.168.177.11}"
if [ "$RANK" = "0" ]; then
  BIN="${BIN:-$PWD/target/release/spark}"
  PORT=8892
else
  BIN="${BIN:-/home/ms/spark-pr973-ep}"
  PORT=0
fi
MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
NUM_SEQS="${NUM_SEQS:-4}"
# 🪤 0.65, NOT the single-node 0.76. util multiplies TOTAL box memory and the KV
# pool then expands to fill whatever the weights leave over. At EP=2 the expert
# weights are HALVED, so pre-KV drops from 85.6 GB to roughly half that and a
# 0.76 budget would inflate KV to ~40 GB for a workload that needs ~14 —
# on top of the ~21 GB of n-gram page cache that lives OUTSIDE the budget.
# That is the overcommit path: the GLM EP=2 run at 0.80 reached 112 GB used /
# 9 GB available and had to be killed, and this box has hard-rebooted from
# over-allocation before. 0.65 settles around 100 GB used / 21 GB available and
# still leaves ~24 GB of KV — about 1M tokens, against the 524,288 that
# 128K x 4 needs. Raise it only while watching `free -g` UNDER LOAD.
GPU_UTIL="${GPU_UTIL:-0.65}"
MTP="${MTP:-1}"
DRAFTS="${DRAFTS:-2}"

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
# three must clear the context or a long prompt dies partway in. ────────────
#
#   ATLAS_QSA_MAX_TOKENS  must be STRICTLY GREATER than max-seq-len: the guard
#                         is `tokens <= cap` and the generated token makes it
#                         max_seq_len + 1. Costs max_tokens*hd*2 per QSA layer.
#   ATLAS_PLE_MAX_TOKENS  bounds the prefill CHUNK, not the prompt. 9000 covers
#                         any prompt length; do NOT scale it with context
#                         (it costs tokens*10240*14 bytes).
#   ATLAS_PLE_CACHE_SLOTS n-gram row cache; 1048576 slots is cheap.
export ATLAS_QSA_MAX_TOKENS="${ATLAS_QSA_MAX_TOKENS:-$((MAX_SEQ_LEN + 4096))}"
export ATLAS_PLE_MAX_TOKENS="${ATLAS_PLE_MAX_TOKENS:-9000}"
export ATLAS_PLE_CACHE_SLOTS="${ATLAS_PLE_CACHE_SLOTS:-1048576}"

export ATLAS_INTHINK_TOOL_LEAK_OPENERS=0
export ATLAS_NO_HW_PRECHECK=1
export ATLAS_QWEN4EXP_BF16_GDN="${ATLAS_QWEN4EXP_BF16_GDN:-0}"

# ── EP wire protocol. v1 FORCES max_batch_size=1 at world_size > 1 ("EP v1
# active: forcing max_batch_size=1" in serve_load.rs), because each v1 command
# addressed a single slot and the head's per-token broadcast loop could not name
# slot N. v2 adds a per-command seq_id preamble, so the worker routes by
# slot_idx and decodes per sequence — that is what makes C>1 possible at all.
#
# 🪤 BOTH RANKS MUST AGREE (`types.rs`): the preamble changes the wire shape, so
# a one-sided setting desynchronises the command stream. It is exported here,
# before the rank split, for exactly that reason.
#
# It is not free: `ssm_reserve.rs` treats v2 as implying FULL-WIDTH slot
# reserve (v2 pins slots in place rather than recycling one), so the SSM pool
# grows with max-num-seqs. Watch `free -g` on the first boot after changing it.
export ATLAS_EP_PROTOCOL="${ATLAS_EP_PROTOCOL:-v2}"

# ── The #972 gates. Set on BOTH ranks, always — see the header. ─────────────
export ATLAS_QWEN4EXP_MTP_HC_BATCHED="${ATLAS_QWEN4EXP_MTP_HC_BATCHED:-1}"
export ATLAS_VERIFY_ROW_PROJ="${ATLAS_VERIFY_ROW_PROJ:-1}"

if [ "$MTP" = "1" ]; then
  export ATLAS_QWEN4EXP_MTP=1 ATLAS_QWEN4EXP_MTP_VERIFY=1 ATLAS_DFLASH_SPEC_THINK=1
  export ATLAS_NO_THINKENDED_GPU_ARGMAX=1
  # Opt past the EP refusal. See note 1 in the header — unproven, and the
  # accept counters on rank 0 are the only proof either way.
  export ATLAS_EP_MTP=1
  # Accept counters live on RANK 0; cheap, and the only liveness proof.
  export ATLAS_MTP_ACCEPT_DEBUG=1
  SPEC_ARGS="--speculative --num-drafts ${DRAFTS} --mtp-gate force"
else
  SPEC_ARGS=""
fi

echo "Qwen3.8-Flash-Next NVFP4  EP=2 (TP=1)  rank=$RANK host=$(hostname)"
echo "  ctx=$MAX_SEQ_LEN seqs=$NUM_SEQS util=$GPU_UTIL mtp=$MTP prefix-cache=on"
echo "  qsa_cap=$ATLAS_QSA_MAX_TOKENS (> ctx) ple_chunk=$ATLAS_PLE_MAX_TOKENS"
echo "  bin=$(sha256sum "$BIN" | cut -c1-16)  master=$MASTER"
free -g | sed -n 2p

exec "$BIN" serve \
  --model-from-path "$MODEL_DIR" \
  --model-name qwen4exp-nvfp4 --kernel-target qwen3.8-flash-next \
  --rank "$RANK" --world-size 2 \
  --tp-size 1 --ep-size 2 \
  --master-addr "$MASTER" --master-port 29500 \
  --bind 0.0.0.0 --port "$PORT" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-num-seqs "$NUM_SEQS" --max-batch-size "$NUM_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype bf16 \
  --ssm-cache-slots "$NUM_SEQS" \
  --max-prefill-tokens 2048 \
  --request-timeout 1800 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  $SPEC_ARGS \
  ${EXTRA_ARGS:-} \
  "${@:2}"
