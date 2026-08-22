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
#   base   54.6 / 65.5 / 105.3 / 130.5  accept 84/83/79/73%   (fp8 head)
# Drop --lm-head-dtype fp8 for the nvfp4 head: 54.6 / 84.2 / 114.4 / 149.9,
# faster but it does not reproduce the BF16 answer (see the flag note below).
#
# Cross-sequence batched verify IS engaging here (it was not until the
# eligibility contradiction in can_batch_verify was removed — before that the
# same build read 54.7 / 54.4 / 51.7 / 55.5, i.e. completely flat). If you
# ever see that flat shape again, the gate says why at
#   RUST_LOG=info,spark::scheduler::mtp_step=debug
# which prints raw_argmax / n_verify / lever / kill-switch once, plus a
# "batched verify DECLINED" line when can_batch_verify refuses.
#
# Both cross-sequence batched VERIFY and batched PROPOSE are engaging here.
# If throughput ever goes flat across concurrency again, the two gates say why
# at RUST_LOG=info,spark::scheduler::mtp_step=debug — and if ACCEPTANCE drops
# while throughput holds, suspect a row-count bound rather than the drafter:
# `ATLAS_DFLASH_BATCH_PROPOSE=2` caps the batch at 2 sequences, and if that
# restores acceptance the fault is width-dependent (that bisect is what found
# the lm_head M_TILE=16 bound).
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

# SimHash semantic-loop guard OFF for this serve: it is ONE-STRIKE at
# Jaccard 0.55 over a 16-sentence ring and kills the stream mid-reply, which
# legitimate structured output (per-method docstrings, enumerations) crosses
# easily. Post-#699 its fires skew false-positive — observed killing a
# healthy TUI session 2026-08-21. Remove this line to re-arm it.
export ATLAS_SIMHASH_LOOP=0

# Serve at INFO so the log is useful if something misbehaves.
export RUST_LOG="${RUST_LOG:-info}"

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
# --lm-head-dtype fp8: serve the vocab projection at the precision the
# CHECKPOINT actually stores. This model ships `lm_head.weight` as FP8 E4M3
# with a per-row scale; `default` re-quantizes it down to NVFP4, which is
# FP8 -> dequant BF16 -> NVFP4 — a double quantization of the single most
# precision-sensitive projection in the model. Measured here: the FP8 head
# reproduces the full-BF16 greedy answer BYTE-FOR-BYTE, while NVFP4 walks a
# different one.
#
# It costs throughput, and the cost is real — aggregate tok/s at C=1/2/4/8 on
# the lean profile:
#     nvfp4   54.7 / 84.1 / 122.7 / 152.2   (different answer)
#     fp8     54.6 / 65.5 / 105.3 / 130.5   (BF16's answer, half the memory)
#     bf16    46.4 / 71.4 /  70.5 / 116.4
# So fp8 beats bf16 at C=1/4/8 for half the memory, and trails nvfp4 by
# ~15% at C=4/8. Drop this flag to get nvfp4 back.
#
# The gap used to be far worse: the FP8 decode path fell to a PER-TOKEN loop
# above M=2, re-reading the whole 1.27 GB head once per token (64 times at a
# C=8 cross-sequence verify). `dense_gemv_fp8w_batchm` reads it once per
# chunk of 8, bit-identically — that is where C=8 77.5 -> 130.5 came from.
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
  SEQS=16
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

# ── POST-#699: prefix caching ON, --exact-verify dropped ───────────────
# Both prior pins were workarounds for ONE bug, fixed in 8867c6de / PR #699:
# k4_apply_verdict rewound the sequence by drafts.len() instead of the
# forward's row count, so a K=4 verify dispatched onto a γ=7-draft DFlash
# sequence EMITTED its accepted tokens and then erased them from history.
# That single defect was: the temp-0 degeneration ("1, 2, 100, 100...", the
# markdown-table garbage), the temp-0 NON-determinism, the video-fidelity
# 0/2 / 0/4 at C=2/C=4, the concurrent shared-prefix derailment, and the
# apparent prefix-cache throughput collapse (the cache only changed when the
# gate's lane flips triggered the collision).
#
# Post-fix, measured on this binary (default verify chain, cache ON):
#   count/colors/two-sentences  correct AND deterministic at temp 0
#   video-fidelity              1/1, 2/2, 4/4 at C=1/2/4
#   MinHeap x4 shared prefix    4/4 coherent, accept 79%
#
# --exact-verify remains available for the #459 1-ULP row-count residual
# (wording-level divergence), but it is no longer needed for correctness.
# Prefix caching is back on because agentic flows depend on it and it is
# no longer implicated in anything.

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
#  * MEASURED 2026-08-22: multi-turn warm hits WORK — 2.2K-token system
#    prompt, TTFT 2.80s cold -> 1.00s/0.89s on turns 2/3 (~3x), Marconi
#    checkpoint restored, only the new suffix replayed, cached_tokens
#    reported correctly. The earlier "no TTFT win (2.13s -> 2.12s)" note
#    was measured on an IDENTICAL-repeat full-prompt hit, whose exact-leaf
#    snapshot shortcut is deliberately bypassed as unsound — that case
#    recomputes by design and says nothing about real conversations.
#    Remaining warm-turn budget: ~0.36s suffix prefill + snapshots, ~0.41s
#    DFlash drafter bootstrap (no cross-turn drafter carry yet — also why
#    warm turns open with 0/7 accepts), ~0.2s first steps.
#  * Prefix caching is the feature with the most correctness history in
#    this codebase — cross-request contamination and block aliasing have
#    both been real bugs on other model families. If you see a reply that
#    belongs to a DIFFERENT conversation, or output that degenerates only
#    on repeat prompts, drop --enable-prefix-caching first and tell me.
# Util 0.80 (was 0.65), 2026-08-22: the preflight now reserves the DFlash
# SSM verify pools honestly (they were allocated OUTSIDE the pledge before —
# 13.7 GB tracked vs a 1.3 GB reserve; the old 0.65 serve actually consumed
# ~95 GB against its 79 GB promise). At an honest 0.65 this profile does not
# fit and the boot refuses; 0.80 pledges 97 GB, which covers the same real
# footprint WITH the pools inside it and gives KV more than the old boot had.
# 0.80 is the hard ceiling for this box — never raise it further.
exec ./target/release/spark serve \
  --model-from-path "$TARGET" \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --draft-model "$DRAFT" --dflash \
  "${LORA_ARGS[@]}" \
  --bind 0.0.0.0 --port 8888 \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-num-seqs "$SEQS" --max-batch-size "$SEQS" \
  --gpu-memory-utilization 0.80 \
  --request-timeout 900 \
  --max-prefill-tokens 8192 \
  --enable-prefix-caching true \
  --ssm-cache-slots 16 \
  --scheduling-policy slai \
  --tbt-deadline-ms 100 \
  --video-allow-ffmpeg \
  --video-ffmpeg-path /usr/bin/ffmpeg \
  --vision-max-pixels 262144 \
  --lm-head-dtype fp8 \
  "${TUI_ARGS[@]}"
