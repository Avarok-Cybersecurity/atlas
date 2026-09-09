#!/usr/bin/env bash
# Serve GLM-5.3-Flash-EXL3 across 2x DGX Spark (GB10) at TP=2 / EP=2.
#
#   ./serve_glm53_exl3_ep2.sh 0     # on gx10-9959 (master, serves :8890)
#   ./serve_glm53_exl3_ep2.sh 1     # on dgx-00
#
#   PACK=k2   (default)  ~2.2 bpw experts, 92 GB   — MTP FITS
#   PACK=4bpw            4 bpw experts,   164 GB   — MTP DOES NOT FIT (see below)
#
# ONE Atlas instance at a time: --gpu-memory-utilization RESERVES its whole
# fraction up front, so a second server fails its OOM pre-flight. Watch
# `free -g`, never nvidia-smi — this box is UNIFIED memory and nvidia-smi is
# blind to the carveout. Over-allocation has hard-rebooted it three times.
#
# ─────────────────────────────────────────────────────────────────────────────
# MEASURED, 2026-09-09, binary 91f7f4def3eb38d8 (branch research/glm-exl3,
# f800f4725), prefix caching ON, salted cold prompts, arm liveness verified on
# RANK 0 (`spark::scheduler::mtp_accept_debug` runs on the serving rank — a
# grep of rank 1 reports "never engaged" for an arm running at mtp=0.96).
#
#   K2, util 0.65                decode      prefill 4K   prefill 16K   tok/step
#   MTP off                      13.99 tok/s   333.8        332.0        1.000
#   MTP on, drafter ctx ON       23.88         299.8        298.8        2.632
#   MTP on, drafter ctx OFF *    23.53         339.0        333.4        2.591
#
#   * SHIPPED HERE. `ATLAS_NO_MTP_DRAFTER_CONTEXT=1` recovers the ENTIRE prefill
#     cost of MTP (+13%) for a 1.5% decode change and an essentially unchanged
#     accept rate. The whole MTP prefill penalty is the drafter's own prefill
#     pass, which is purely additive and matches the logged drafter time almost
#     exactly (4K: +1.3 s wall vs 1375 ms logged; 16K: +5.2 s vs 5398 ms).
#
#     🪤 That default (prefill ON + carry ON) was set from an MLPerf-edge run on
#     DIFFERENT hardware and a different model, where TTFT IMPROVED because
#     drafter prefill lifts accept on the earliest decode steps. This probe uses
#     max_tokens=8 for prefill, so it measures the cost and almost none of that
#     benefit, and it does not exercise `carry` (drafter state reused ACROSS
#     turns) at all. If you run multi-turn or agentic work, re-measure with the
#     line commented out before trusting it.
#
#   4bpw, util 0.85              prefill 5.4K 315.4 | prefill 21K 301.2
#                                decode: UNMEASURED (see below)
#
# ─────────────────────────────────────────────────────────────────────────────
# 🔴 4bpw + MTP DOES NOT FIT, and the reason is structural, not a tuning miss:
#
#     Insufficient GPU memory for inference buffers. After loading 90.79 GB of
#     weights, only 8.03 GB remains but 9.87 GB is needed for SSM state pool
#     (1 slots x 34 layers) + scratch buffers.
#
# Short by ~1.8 GB. Halving --max-seq-len 32768 -> 16384 reclaims only 0.23 GB
# because that pool is the KDA recurrent state — 34 layers x ~290 MB, fixed and
# CONTEXT-INDEPENDENT. MTP costs ~12 GB on this pack (the inference reserve goes
# 1679 MB -> 6104 MB, plus drafter weights), against 4bpw's 90.79 GB/rank versus
# K2's 54.51 GB. There is no util that fixes it: below ~0.80 the budget no
# longer covers the weights, above it the box runs out.
#
# 🪤 Counter-intuitive: 4bpw wants a HIGHER util than K2, not a lower one. util
# multiplies TOTAL box memory and the KV pool then expands to fill whatever is
# left over, so on K2 a high util inflates KV (98,754 blocks = 16.6 GB for a
# batch-1 workload that needs 2,050) until the box overcommits. On 4bpw the
# weights leave nothing spare, KV self-clamps to 2,050 blocks / 0.3 GB, and the
# util only has to be large enough to cover the weights.
set -uo pipefail
RANK="${1:?usage: serve_glm53_exl3_ep2.sh <rank 0|1>}"
PACK="${PACK:-k2}"

case "$PACK" in
  k2)
    # NVMe-backed NFS share, mounted at the SAME path on both nodes so one
    # MODEL_DIR resolves for rank 0 (local) and rank 1 (NFS over 200 GbE).
    # Booting off this instead of the SATA /tank took READY 212s -> 116s.
    MODEL_DIR="${MODEL_DIR:-/srv/nvme-models/glm53-k2}"
    # 0.65, NOT 0.85. Higher inflates the KV pool until the box overcommits:
    # at 0.80 with MTP the run reached 112 GB used / 9 GB available and had to
    # be killed. At 0.65 it settles at ~100 GB used / 21 GB available.
    GPU_UTIL="${GPU_UTIL:-0.65}"
    MTP="${MTP:-1}"
    ;;
  4bpw)
    MODEL_DIR="${MODEL_DIR:-/tank/hf/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/024db9f7e9871e8efdf21538ba55af7442be3cd5}"
    # Needs the high util just to cover 90.79 GB/rank of weights; KV self-clamps.
    GPU_UTIL="${GPU_UTIL:-0.85}"
    # Refuses at boot with the message quoted above if forced on.
    MTP="${MTP:-0}"
    if [ "$MTP" = "1" ]; then
      echo "WARNING: PACK=4bpw with MTP=1 is expected to REFUSE at boot (~1.8 GB short)." >&2
      echo "         See the header. Proceeding because you asked explicitly." >&2
    fi
    ;;
  *) echo "unknown PACK=$PACK (expected k2 or 4bpw)" >&2; exit 2 ;;
esac
if [ ! -e "$MODEL_DIR/config.json" ]; then
  echo "checkpoint not found: $MODEL_DIR" >&2; exit 1
fi

MASTER="${MASTER:-192.168.177.12}"
BIN="${BIN:-/home/ms/spark-glm53-exl3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
DRAFTS="${DRAFTS:-2}"

# ── Fabric. The f1 rail carries 192.168.177.0/24 here; upstream names f0
# because that is their fabric, not because f0 is required. ──
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_NVLS_ENABLE=0
export NCCL_PROTO=Simple
export NCCL_CROSS_NIC=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export LD_LIBRARY_PATH="/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export RUST_LOG="${RUST_LOG:-info}"

# ── Speed settings ───────────────────────────────────────────────────────────
#
# Everything below is a DEFAULT in this branch; they are named explicitly so a
# run is self-describing and so each one's kill-switch is discoverable. Prefill
# went 94.2 -> 340.6 tok/s at 5.4K and 88.6 -> 333.1 at 21K (3.6x) across these.
#
#   ATLAS_GLM_PREFILL_ROWS=256   sub-chunk width. MEASURED OPTIMUM — 512
#                                regressed twice (329.8/304.2 vs 337.0/319.3),
#                                so do not raise it without re-measuring.
#   ATLAS_GLM_MOE_PREFILL_MIN=64 fused sort-by-expert MoE prefill arm.
#   ATLAS_EXL3_MOE_ROWS_PER_EXPERT=4096
#                                fused-tier row cap; the overflow tier fires on
#                                100% of calls at the stock value against an
#                                8192 server chunk.
#
# Default-ON, kill-switch only (listed so they can be bisected):
#   ATLAS_GLM_CUBLAS_PROJ=0          wide projections back to the scalar GEMM
#   ATLAS_GLM_KDA_CHUNK_PREFILL=0    KDA back to the per-token recurrent walk
#   ATLAS_GLM_DSA_BATCH_QIDX=0       DSA indexer q back to one GEMV per row
#   ATLAS_GLM_DSA_BATCH_KV_WRITE=0   per-row KV latent write
#   ATLAS_DSA_SELECT_ROWS=0          per-row DSA selection
export ATLAS_GLM_PREFILL_ROWS="${ATLAS_GLM_PREFILL_ROWS:-256}"
export ATLAS_GLM_MOE_PREFILL_MIN="${ATLAS_GLM_MOE_PREFILL_MIN:-64}"
export ATLAS_EXL3_MOE_ROWS_PER_EXPERT="${ATLAS_EXL3_MOE_ROWS_PER_EXPERT:-4096}"

if [ "$MTP" = "1" ]; then
  # `factory/build.rs` loads the GLM drafter on
  # `model_type == "glm5_next" && use_speculative`, so --speculative is the switch.
  SPEC_ARGS="--speculative --num-drafts ${DRAFTS}"
  # See the header table: recovers MTP's entire prefill cost. Strict `=1` —
  # this module presence-checks nothing, and `ATLAS_*=0` has burned this
  # codebase before, so `=0` is NOT how you turn it off; unset it instead.
  export ATLAS_NO_MTP_DRAFTER_CONTEXT="${ATLAS_NO_MTP_DRAFTER_CONTEXT:-1}"
  # Accept counters on rank 0. Cheap, and the only proof the arm is live.
  export ATLAS_MTP_ACCEPT_DEBUG="${ATLAS_MTP_ACCEPT_DEBUG:-1}"
else
  SPEC_ARGS=""
fi

if [ "$RANK" = "0" ]; then PORT=8890; else PORT=0; fi
echo "GLM-5.3-Flash-EXL3  pack=$PACK rank=$RANK host=$(hostname)"
echo "  util=$GPU_UTIL ctx=$MAX_SEQ_LEN mtp=$MTP drafts=$DRAFTS prefix-cache=on"
echo "  model=$MODEL_DIR"
echo "  bin=$(sha256sum "$BIN" | cut -c1-16)"
free -g | sed -n 2p

exec "$BIN" serve \
  --model-from-path "$MODEL_DIR" \
  --rank "$RANK" --world-size 2 \
  --tp-size 2 --ep-size 2 \
  --master-addr "$MASTER" --master-port 29500 \
  --bind 0.0.0.0 --port "$PORT" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --oom-guard-mb "${OOM_GUARD_MB:-1024}" \
  --max-batch-size 1 \
  --swap-space-gb 0 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  $SPEC_ARGS \
  ${EXTRA_ARGS:-} \
  --no-tui
