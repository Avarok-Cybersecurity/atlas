#!/usr/bin/env bash
# ── Qwen3.8-27B + DFlash2 drafter — interactive TUI serve ──────────────
# Runs the FULL upstream stack, branch `wip/dflash2-full`:
#
#   #649  base layer (rrstesiak): DFlash2/DSpark drafter, rt2 register-tiled
#         batch8 GEMV, FP8 propose twin, and OPTION_B / EAGLE_FIX /
#         DRAFTER_FP8 flipped to defaults — a bare --dflash is the record path
#   #650  memory: allocation ledger, lazy ViT scratch, DFlash buffers sized to
#         what a request can reach, drafter footprint pre-flight
#   #651  LoRA on the hybrid architecture + both perf regressions closed
#   #652  TUI prefill visibility + the DSpark/DFlash metrics fixes
#   #653  w8a16 batched-GEMV bit-parity fix (verify arm now byte-identical to
#         the M=1 decode arm) + its standing gate
#
# NOT a PR branch — an integration branch for driving the whole thing at once.
# Rebuild after switching branches; the guard below refuses a stale binary.
#
#   ./serve_dflash_tui.sh                  # 128K ctx, 8 concurrent (KV overcommit)
#   ./serve_dflash_tui.sh small            # 32K ctx, 4 concurrent
#   ./serve_dflash_tui.sh default novelist # ...with a LoRA adapter loaded
#   ./serve_dflash_tui.sh small cyber
#
# Measured on the LEAN profile (32K ctx, 8 seqs, no prefix cache), which is
# the one with a clean baseline, aggregate tok/s at C=1/2/4/8:
#   base   54.7 / 54.8 / 54.1 / 52.8    accept 84/84/77/62%
#
# Note the shape: accept is healthy but aggregate is FLAT. Per-sequence verify
# costs the same at any width, so N streams cost N times one stream. The work
# that amortises it — cross-sequence batched verify and batched propose — is
# NOT on this branch: it was written against the WIP branch's own drafter head
# and #649 supplies a different head for the same lane, so it needs porting
# rather than picking. On the WIP branch it took C=2 from 25 to 47.8 tok/s.
#
# The older numbers this header used to carry (39.7 / 72.1 / 76.8 / 101.4)
# came from that WIP branch WITH the batched work, on the 128K profile. They
# are not comparable and are not what this branch does.
#
# Verified at C=8: ZERO "batched verify DECLINED" lines, i.e. the cross-
# sequence batched verify — the thing that makes concurrency amortise — is
# engaging at this width, not silently falling back to the per-sequence loop.
# `ATLAS_MTP_MAX_SEQS` defaults to 16 so speculation stays on at 8; dropping
# it below the seq count would turn DFlash off exactly when it matters most.
#
# The LoRA row threshold is ALSO right for this width and needs no override:
# at 8 seqs the verify presents 8*9=72 rows, which correctly takes the GEMM.
# Forcing the GEMV loop past it (ATLAS_LORA_GEMV_MAX_M=96) measured WORSE —
# 65.9 -> 62.9 at C=8 — because 72 sequential GEMVs lose to one GEMM.
set -uo pipefail
cd "$(dirname "$0")"

# Guard: this script runs ./target/release/spark, which is whatever was built
# LAST — not necessarily this branch. Serving a stale binary is the kind of
# thing that costs an hour of confused benchmarking, so refuse instead.
WANT_BRANCH=wip/dflash2-full
HAVE_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
if [ "$HAVE_BRANCH" != "$WANT_BRANCH" ]; then
  echo "WARNING: on branch '$HAVE_BRANCH', expected '$WANT_BRANCH'." >&2
  echo "         git checkout $WANT_BRANCH  (then rebuild)" >&2
fi
if [ ! -x ./target/release/spark ]; then
  echo "no ./target/release/spark — build first:" >&2
  echo "  ATLAS_TARGET_MODEL='*' RUSTFLAGS=\"-L native=/home/ms/nccl/build/lib\" \\" >&2
  echo "    cargo build --release -p spark-server --bin spark" >&2
  exit 1
fi
NEWEST_SRC=$(find crates kernels -newer ./target/release/spark -name '*.rs' -o -newer ./target/release/spark -name '*.cu' 2>/dev/null | head -1)
if [ -n "$NEWEST_SRC" ]; then
  echo "WARNING: $NEWEST_SRC is newer than the binary — rebuild before trusting numbers." >&2
fi

MODE="${1:-default}"
ADAPTER="${2:-}"

# HEADLESS=1 runs the identical config with --no-tui, so a benchmark measures
# exactly what the TUI serves rather than a hand-retyped approximation.
TUI_ARGS=()
[ "${HEADLESS:-0}" = "1" ] && TUI_ARGS=(--no-tui)

# GDN FlashInfer: worth ~25% of prefill (715 -> 895 tok/s). It fails OPEN —
# without ATLAS_GDN_LIB on LD_LIBRARY_PATH it silently falls back and you
# just lose the speed, no error. Keep both together.
export LD_LIBRARY_PATH="/home/ms/atlas-gdn-libs:/home/ms/nccl/build/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_GDN_FLASHINFER=1
export ATLAS_GDN_LIB=/home/ms/atlas-gdn-libs/libatlasgdn.so

# FP8 drafter weights are now DEFAULT-ON in the engine, so this export is
# redundant and kept only as documentation of intent. Measured head-to-head
# on this build (BF16 -> FP8): C=1 34.8 -> 37.7 tok/s, C=2 33.5 -> 47.9,
# C=4 55.5 -> 57.1, prefill unchanged, and acceptance equal-or-better on
# every leg (85->86%, 78->79%, 69->70%). Costs ~1.9 GB of FP8 mirrors.
# `ATLAS_DFLASH_DRAFTER_FP8=0` opts back out.

# Serve at INFO so the log is useful if something misbehaves.
export RUST_LOG=info

# Everything else that matters is DEFAULT-ON as of this branch and needs no
# env var: Option B paged drafter KV, unified ctx commit, GPU candidate
# selector, the C<=2 gate pin, batched verify, batched propose.
# Opt-outs if you ever need to A/B:
#   ATLAS_DFLASH_BATCH_VERIFY=0   per-sequence verify
#   ATLAS_DFLASH_GATE_PIN_C2=0    let the throughput gate arbitrate at C<=2
#   ATLAS_DFLASH_OPTION_B=0       legacy full-rebuild drafter ctx
#   ATLAS_MTP_GATE_FORCE=1        always verify (peaks C=4 at ~55 tok/s,
#                                 but costs aggregate at higher widths)

# No --dflash-gamma here: this branch resolves gamma from the drafter itself.
# WATCH THE BOOT LINE ANYWAY — it should say γ=8:
#   DFlash speculative decoding: ENABLED (γ=8, ...)
# On a branch without that fix the same command comes up at γ=16 against an
# 8-block drafter, and this is NOT a subtle degradation: measured here, it
# returns 5-token completions at 14.8 tok/s, and it sizes the drafter's
# per-sequence pools for twice the block they will ever hold (which is what
# pushed the 128K x 8 profile under 8 GB free on boot). If you see γ=16,
# either the branch is wrong or pass --dflash-gamma 8 explicitly.
TARGET=$(ls -d /mnt/gx10-hf-hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/*/ | head -1)
DRAFT=$(ls -d /mnt/gx10-hf-hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/*/ | head -1)

# ── LoRA (optional second argument) ────────────────────────────────────
# All three tested adapters load with NO opt-in flag: dense-FFN deltas are
# applied on all 64 layers (hybrid included), GDN out_proj is supported, and
# PEFT regex `target_modules` parses. Select per request with either
# {"model":"<name>"} or {"adapter":"<name>"}; an unnamed request gets the
# ACTIVE adapter, not base.
#
# The LoRA perf knobs are all correct by default and need no env here:
#   ATLAS_LORA_GEMV_MAX_M=48      small-m deltas run as row GEMVs, not a
#                                 16-row-tiled GEMM (this is what took C=2
#                                 from 4.8 to 11.9 tok/s/stream)
#   batched speculative verify is ALLOWED under an adapter (flat ~34 tok/s
#   at every concurrency before that)
# Bisect handles if output ever looks wrong with an adapter loaded:
#   ATLAS_LORA_NO_APPLY=1         adapter resident, deltas skipped
#   ATLAS_LORA_NO_FFN=1           skip FFN/SSM deltas, keep attention
#   ATLAS_LORA_PREFILL_BGMV=1     old per-row prefill path
#   ATLAS_LORA_NO_BATCH_VERIFY=1  old refusal of batched verify under LoRA
#   ATLAS_LORA_ALLOW_PARTIAL=1    load an adapter naming unsupported modules
#
# Mixed adapters: v0 is single-active, so a batch may only hold ONE adapter's
# sequences. Requests naming a different resident adapter are now HELD at
# admission and run when the batch drains — serialised, not failed (they used
# to take the whole batch down with them, including innocent requests routed
# to the active adapter). Correct but serial, so throughput on a genuinely
# mixed workload is roughly one adapter's worth.
LORA_ARGS=()
if [ -n "$ADAPTER" ]; then
  case "$ADAPTER" in
    novelist) APATH=$(ls -d /mnt/gx10-hf-hub/models--Dxniz--Novelist1.0-27b-Adapter/snapshots/*/ | head -1) ;;
    cyber)    APATH=/home/ms/lora-test/cyber ;;
    heresy)   APATH=/home/ms/lora-test/heresy ;;
    *)        APATH="$ADAPTER" ;;   # or pass a path directly
  esac
  if [ ! -e "$APATH" ]; then
    echo "adapter '$ADAPTER' not found at $APATH" >&2
    exit 1
  fi
  LORA_ARGS=(--lora-adapter "${ADAPTER}=${APATH}")
  echo "LoRA: serving adapter '$ADAPTER' from $APATH"
fi

if [ "$MODE" = "small" ]; then
  MAX_SEQ_LEN=32768
  SEQS=4
else
  MAX_SEQ_LEN=131072
  SEQS=8
fi

# KV OVERCOMMIT — this is what makes 128K x 8 boot at all. The KV pool cannot
# hold 8 sequences at the FULL 128K ceiling, and without overcommit that is a
# boot-time HARD REFUSAL. Overcommit downgrades it to a warning: the scheduler
# admits up to --max-batch-size and the paged pool fills on demand, so a
# genuinely over-long burst is back-pressured at the block allocator (spill /
# requeue for later resume) instead of being refused at startup.
#
# It is DEFAULT ON; set here explicitly because it is load-bearing for this
# config rather than incidental. `ATLAS_KV_OVERCOMMIT=0` restores the hard
# refusal, which is the honest way to find out whether a config really fits.
export ATLAS_KV_OVERCOMMIT=1
#
# NOTE: the separate ATLAS_KV_ADMIT_WATERMARK knob is deliberately NOT set.
# Admission already reserves `prompt + min(request.max_tokens, watermark)`, so
# with the default watermark each request reserves only its OWN max_tokens —
# lowering it would just add preemption churn on legitimately long turns.
#
# The adapter pool now SIZES ITSELF to the resident adapters' real rank, so
# there is nothing to tune here: a small adapter no longer pays a large cap's
# memory or bandwidth (cyber r=8 went 5392 -> 674 MiB and +20% prefill just
# from that). Pass --max-lora-rank explicitly only to reserve headroom for a
# LARGER adapter staged in later, since the pool layout freezes at startup.

# Vision input bound: --vision-max-pixels is an AREA in pixels, so 256K =
# 262144 (512x512, or any same-area shape). 0 would mean "use the
# checkpoint's own bound". Note the cost model: vision tokens grow with AREA,
# so raising this is charged against the context budget — at 256K px and a
# 2x2 merge that is roughly 256 tokens per image, which is cheap; a 4096^2
# image would be ~16k tokens, which is not.
# Video: --video-allow-ffmpeg opts INTO subprocess decoding for real
# containers (mp4/mkv/webm). It is off by default on purpose — it lets the
# server exec a subprocess — so it must be granted explicitly, not inherited
# by upgrading. Decoding is sandboxed: no shell, no temp file, stdin is the
# pipe with -nostdin, and frames/output/wall-clock are all capped.
# --video-ffmpeg-path pins the binary instead of trusting PATH.
# SLAI (SLO-aware) scheduling instead of FIFO: prioritises decode for
# sequences approaching the TBT deadline, and orders prefills
# shortest-prompt-first. That matters here because one long prefill (~9s for
# an 8K prompt at ~890 tok/s) otherwise stalls every active decode behind it.
# --tbt-deadline-ms is the per-token budget SLAI steers by (100ms default).
# Revert with `--scheduling-policy fifo`.
#
# Prefix caching (RadixAttention). Off by default in Atlas; on here so you
# can exercise it. Two things worth knowing on THIS model:
#
#  * Qwen3.8 is a HYBRID (GDN/SSM + attention). Reusing KV alone is not
#    enough — without an SSM snapshot the recurrent state must be rebuilt,
#    so you get block-table reuse but NO TTFT win. The Marconi snapshot
#    slots are what make a warm turn fast — but they are EXPENSIVE here:
#    ~151 MB per slot, so 16 slots ~= 2.4 GB and 64 slots ~= 9.7 GB of GPU
#    memory. At --gpu-memory-utilization 0.65 that is a big bite, so this
#    stays at the default 16. Raise it only if you have headroom, and NOT
#    together with 128K ctx unless you have checked free memory.
#  * Costs ~1.7% aggregate throughput at C=8 on this profile (101.4 -> 103.1
#    with it off) — within run-to-run noise, so it is NOT the reason to keep
#    or drop it. The 2.4 GB is.
#  * MEASURED on this build: the radix tree DOES hit (full 1561-token
#    prompt matched on a repeat) but the warm turn was NOT faster
#    (2.13s -> 2.12s) and usage reported cached_prompt_tokens=0. So today
#    you get the block-table reuse and the hit-rate telemetry, not a TTFT
#    win. That matches the known chunk-alignment gap, not a misconfig.
#  * Prefix caching is the feature with the most correctness history in
#    this codebase — cross-request contamination and block aliasing have
#    both been real bugs on other model families. If you see a reply that
#    belongs to a DIFFERENT conversation, or output that degenerates only
#    on repeat prompts, drop --enable-prefix-caching first and tell me.
exec ./target/release/spark serve \
  --model-from-path "$TARGET" \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --draft-model "$DRAFT" --dflash \
  "${LORA_ARGS[@]}" \
  --bind 0.0.0.0 --port 8888 \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-num-seqs "$SEQS" --max-batch-size "$SEQS" \
  --gpu-memory-utilization 0.65 \
  --request-timeout 900 \
  --max-prefill-tokens 8192 \
  --enable-prefix-caching true \
  --ssm-cache-slots 16 \
  --scheduling-policy slai \
  --tbt-deadline-ms 100 \
  --video-allow-ffmpeg \
  --video-ffmpeg-path /usr/bin/ffmpeg \
  --vision-max-pixels 262144 \
  "${TUI_ARGS[@]}"
