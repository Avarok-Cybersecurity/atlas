#!/usr/bin/env bash
# Serve GLM-5.3-Flash-EXL3 across 2x DGX Spark (GB10) at TP=2 / EP=2.
#
#   ./serve_glm53_exl3_ep2.sh 0     # on gx10-9959 (master, serves :8890)
#   ./serve_glm53_exl3_ep2.sh 1     # on dgx-00
#
#   PACK=k2   (default)  ~2.2 bpw experts, 92 GB   — MTP FITS
#   PACK=4bpw            4 bpw experts,   164 GB   — MTP DOES NOT FIT (see below)
#
# INTERACTIVE TUI. It lives on rank 0, the serving rank, and needs a real TTY — so
# rank 0 has to run in the foreground of an ssh session with `-t`, not under nohup:
#
#   # on dgx-00, start the worker in the background first (it retries the connect
#   # for ~10 min, so either order works):
#   PACK=k2 nohup ./serve_glm53_exl3_ep2.sh 1 > /tmp/glm53_r1.log 2>&1 &
#
#   # then the serving rank WITH the dashboard:
#   ssh -t gx10-9959 'PACK=k2 TUI=1 /home/ms/serve_glm53_exl3_ep2.sh 0'
#
# Without `ssh -t` there is no TTY and the TUI silently falls back to the log
# stream — which looks like the flag was ignored.
#
# ONE Atlas instance at a time: --gpu-memory-utilization RESERVES its whole
# fraction up front, so a second server fails its OOM pre-flight. Watch
# `free -g`, never nvidia-smi — this box is UNIFIED memory and nvidia-smi is
# blind to the carveout. Over-allocation has hard-rebooted it three times.
#
# ─────────────────────────────────────────────────────────────────────────────
# HANDOFF — state as of 2026-09-09, branch `research/glm-exl3`.
# Written so the next session can pick this up cold.
#
# WHAT SHIPPED THIS SESSION
#
#   2bb9933ad  the tool-envelope guard counted GLM/Laguna file writes as
#              envelope junk. `update_tool_param_state` exempts tool ARGUMENT
#              VALUES from the 1024-token envelope cap so a large write can
#              stream — but the exemption recognised only Qwen's
#              `<parameter=KEY>` form, matched by hardcoded Qwen3.6 token ids.
#              GLM/Laguna use poolside_v1 (`<arg_value>`…`</arg_value>`), so
#              NOTHING was exempt and every write past the cap was force-ended
#              mid-file with "Stuck in tool-call ENVELOPE". Delimiters are now
#              TOKENIZER-DERIVED (GLM: 154849/154850). `ATLAS_TOOL_ENVELOPE_
#              WATCHDOG` also became REAL — it was named in two source comments
#              and in operator recipes while nothing read it.
#   4352645d7  DFlash2 SSM rollback after gamma-verify (hybrid targets).
#   b1fe16445  TUI=1 on the serving rank.
#
# BOOT PROOF for the envelope fix — if this line is ABSENT the exemption did
# not resolve and long writes will truncate again:
#   "Tool argument-value delimiters: <arg_value> (154849) .. </arg_value> (154850)"
#
# MEASURED — Gate A `agentic-webserver`, 2026-09-09, binary fe8a53a1ededa20d.
# The REAL harness (`spark benchmark run agentic-webserver`), not a hand-rolled
# probe: the model writes an Axum ping/pong server with write_file/read_file/
# bash in a sandbox, and the scorer builds it and curls /ping.
#
#   AGENTIC=1, reasoning_effort=low pinned server-side, preserve-thinking on
#   (MODEL.toml [behavior] + client ATLAS_AGENTIC_PRESERVE_THINKING=1),
#   iterations=3, wall_budget_s=30000, max_turns=40.
#
#                        webserver_ok  followed_dirs  s/turn   Σwall   decode
#     K2   util 0.65 MTP on    2/3          2/3       26.80s    699s   17-19 tok/s
#     K4   util 0.80 MTP off   3/3          2/3       19.44s    528s   13.6-13.9
#
#   ZERO tool-envelope and ZERO inter-tool-prose guard fires in either tier —
#   the envelope watchdog stayed ARMED and never tripped. That is the fix
#   confirmed under the real agentic loop.
#
#   🪤 NOT a one-variable A/B: quantization, MTP (forced off — 4bpw+MTP refuses
#   at boot ~1.8 GB short), util AND checkpoint storage all differ. At n=3 with
#   a 0/1 outcome metric, 3/3 vs 2/3 is ONE run's difference and is NOT
#   significant. `followed_directions` was identical. K4-looks-better is a
#   hypothesis needing more iterations, not a finding. The one durable
#   observation: K4 decodes SLOWER per token yet finishes turns FASTER
#   (19.4 vs 26.8 s/turn at similar turn counts), i.e. it emits less text to
#   reach the same place.
#
#   🪤 K4 was run at util 0.80, NOT this script's 0.85 default. At 0.85 the box
#   sat at 116 GB used / 1 GB AVAILABLE, and the agentic scorer runs cargo
#   builds on that same box — the OOM shape that has hard-rebooted this machine
#   three times. 0.80 still clears the 90.79 GB/rank of weights.
#
#   Neither tier GATES: the harness warns GLM is not a declared variant of
#   agentic-webserver, so no committed thresholds exist and `GATE_A_EXIT=2` is
#   the schema's followed_directions bound, not a GLM threshold.
#
# REASONING EFFORT IS A CHAT-TEMPLATE KWARG, and the model card CANNOT express
# it: `ModelBehavior` carries max_thinking_budget / thinking_default /
# effort_capped_at_ceiling / preserve_thinking but has NO default-effort field.
# The only server-side lever is
#   --default-chat-template-kwargs '{"reasoning_effort":"low"}'
# Confirm it took: a request with NO reasoning params must log
#   "Thinking enabled, budget=Some(1024)"   (= max_thinking_budget/2 = low)
# The agentic benchmark omits reasoning_effort from its body ON PURPOSE
# (benchmarks/agentic/agent.rs:367) precisely so this serve default governs.
#
# VISION — vision-fidelity is now 100% PASS on the K2 pack. It was 5/14, and the
# ENGINE WAS CORRECT ON ALL FOURTEEN FIXTURES the whole time; the ladder had only
# ever been taught Qwen3-VL. Fixed in a5ae0a547 (+ cherry-picks 9a1eefb28,
# d004b6bef bringing PR #661's vision_max_pixels predictor onto this branch).
#
#   before   geometry  5/14   probes 2/3   integrity  9/17   FAIL
#   after    geometry 14/14   probes 3/3   integrity 17/17   PASS
#
# Run it as:
#   ./target/release/spark benchmark run vision-fidelity \
#     --url http://192.168.177.12:8890 --model glm53-k2 \
#     --param vision_geometry=glm5 --param max_tokens=2048 --param vision_max_pixels=0
#
#   vision_geometry=glm5   GLM is patch 14 / merge 2 on a CEIL canvas with a
#                          min_image_tokens floor of 16. The ladder hardcoded
#                          Qwen's patch 16 / merge 2 / round / no floor.
#   max_tokens=2048        the integrity probes carried literal 8-24 token
#                          budgets; GLM restates the question before answering,
#                          so `length` landed before the answer and the cell
#                          scored an EMPTY reply that reads as a vision failure.
#   vision_max_pixels=0    this serve passes no --vision-max-pixels and the
#                          checkpoint bound (3,211,264 px) is above every
#                          fixture. If you DO cap, pass the same area.
#
# 🪤 THE MISS HID ITSELF, which is why it looked like an engine defect. Template
# overhead is calibrated as total(224) - predicted(224), so a wrong prediction at
# the anchor is absorbed into a compensating wrong overhead (31 vs the true 16)
# and every 224-sized rung passes BY CONSTRUCTION. A vision ladder that shows
# small-pass / large-fail should have its GEOMETRY PROFILE suspected first.
#
# 🪤 --vision-max-pixels 262144 (the prod value) is measured to break OCR:
# commit 694c18ff3 found the label probe fails HEAD-INDEPENDENTLY under it, both
# fp8 and nvfp4 reading the ~1.9x-downscaled 1280x720 label as "1380". Uncapped
# passes. That is resolution, not the model.
#
# NEXT UP — video-fidelity, which is NOT fixed: 1/8 legs passed, 6 SKIPPED.
#   - The 6 skips are a DEPLOYMENT setting, not a defect: MP4/MOV decoding needs
#     --video-allow-ffmpeg, which this script does not pass.
#   - The GIF path decoded and found something real: 03_colors_fwd.gif came back
#     with the colour sequence exactly REVERSED ([yellow, blue, green, red] for
#     [red, green, blue, yellow]). video-before-image returned empty and
#     video-in-history denied a clip was attached.
#   - NOT a geometry problem: mixed-media-pads arithmetic was EXACT
#     (image 108 + video 300 - text 42 = 366 served).
#   - video/driver.rs:294 still hardcodes tokens_per_group(224, 224, 16, 2),
#     the same Qwen assumption the still-image ladder just shed.
#   - Read the buf_out overrun note before running video against a CAPPED serve:
#     multi-group video can walk past the packed buffer (CUDA 700, wedged 503),
#     and the only bound is a debug_assert stripped in release.
#
# STILL OPEN
#   - verify_dflash_batch_step.rs has neither the EP broadcast nor the SSM
#     rollback. Unreachable at --max-batch-size 1; left untested rather than
#     changed blind.
#   - 2 pre-existing spark-model test failures from 3a121dbbb (arena leak +
#     vision sidecar).
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
# TUI=1 keeps the interactive dashboard on the SERVING rank. It is off by default
# because both ranks are normally launched with nohup/setsid, where the TUI would
# auto-disable anyway (no TTY) — and because rank 1 is a worker with nothing to show.
#
# 🪤 Only rank 0 serves and only rank 0 gets the TUI. Asking for it on rank 1 is
# almost certainly a mistake, so say so rather than silently ignoring it.
TUI="${TUI:-0}"
if [ "$TUI" = "1" ] && [ "$RANK" != "0" ]; then
  echo "TUI=1 ignored: rank $RANK is a worker; the dashboard lives on rank 0 (the" >&2
  echo "  serving rank, on \$MASTER). Run the TUI there instead." >&2
  TUI=0
fi
if [ "$TUI" = "1" ]; then TUI_ARG=""; else TUI_ARG="--no-tui"; fi

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

# KV cache dtype. fp8 is the default here and halves the KV footprint, but this
# checkpoint ships NO k_scale/v_scale, so fp8 falls back to a scale of 1.0 and
# clips bf16 into E4M3's [-448, 448] — the boot log says so in as many words.
# KV_DTYPE=bf16 is the control for anything that looks like a PERCEPTION or
# coherence defect rather than a speed one.
KV_DTYPE="${KV_DTYPE:-fp8}"

# ── Concurrency ──────────────────────────────────────────────────────────────
# EP_PROTOCOL selects the expert-parallel wire protocol.
#
#   v1 (default, historical)  a single implicit sequence. serve_load.rs REFUSES
#                             to honor --max-batch-size under world_size>1 and
#                             logs "EP v1 active: forcing max_batch_size=1",
#                             because the worker has no way to tell which
#                             sequence a command belongs to and N>1 would
#                             cross-write KV.
#   v2                        every command carries a slot-aware seq_id
#                             preamble, so the worker routes each one to the
#                             right SSM slot and --max-batch-size is honored.
#
# 🪤 BOTH RANKS MUST AGREE. `rank_agree` treats this as a collective-shaping
# scalar and prints it at boot as `ATLAS_EP_PROTOCOL(v2)=0|1`; a mismatch means
# one rank frames the wire differently from the other. Setting it here, in the
# script both ranks run, is what keeps them together.
EP_PROTOCOL="${EP_PROTOCOL:-v1}"
export ATLAS_EP_PROTOCOL="$EP_PROTOCOL"

# Concurrent sequences. Only honored under EP v2 (see above); at v1 the server
# clamps it to 1 no matter what is passed here.
MAX_BATCH="${MAX_BATCH:-1}"
if [ "$EP_PROTOCOL" != "v2" ] && [ "$MAX_BATCH" != "1" ]; then
  echo "NOTE: MAX_BATCH=$MAX_BATCH will be CLAMPED to 1 — EP_PROTOCOL is $EP_PROTOCOL, not v2." >&2
  echo "      Pass EP_PROTOCOL=v2 to actually run concurrent sequences." >&2
fi

# ---------------------------------------------------------------------------
# AGENTIC=1 — loosen the decode-time GUARDS for tool-driven coding sessions.
#
# These are heuristics that end a response early. Each is defensible on chat
# traffic and each misfires on agentic traffic, where the model legitimately
# emits long, repetitive, structured output (file writes, diffs, enumerations).
# A misfire is not a degraded answer: it TRUNCATES the turn mid-tool-call, and
# the harness then sees a malformed call and retries forever.
#
# 🪤 Do NOT set these for a quality/benchmark run. They are real guards; this
# profile trades their protection for turns that finish. Measure with them at
# their defaults.
#
#   ATLAS_SIMHASH_LOOP=0
#     F4 semantic-loop guard. ONE-STRIKE at Jaccard 0.55 over a 16-sentence
#     ring, which per-method docstrings and boilerplate-heavy code cross
#     honestly; a fire KILLS the stream mid-reply. Its own source comment
#     records a false positive on a healthy TUI session (2026-08-21).
#
#   ATLAS_LOOP_NO_SUPPRESS=1
#     Drops the API-layer loop-detect logit mask, which vLLM does not apply.
#
#   ATLAS_MAX_INTER_TOOL_PROSE=0  (0 => u32::MAX, i.e. disabled)
#     Cap on free text BETWEEN tool calls. A legitimate PLAN / analysis turn
#     is subject to it and gets guillotined mid-sentence at the budget.
#
# NOT set here, deliberately: ATLAS_TOOL_ENVELOPE_WATCHDOG. The
# "Stuck in tool-call ENVELOPE for 1024+ tokens" kill this model was hitting on
# every long write was a BUG, not a tuning problem — the guard's argument-value
# exemption only recognised Qwen's `<parameter=KEY>` form, so on GLM's
# `<arg_value>...</arg_value>` form NO write content was exempt and the cap
# counted file bytes. That is fixed at the source (the delimiters are now
# tokenizer-derived), so the guard is left ARMED and still does its real job of
# catching a `<tool_call>` that never closes. Confirm at boot with:
#   "Tool argument-value delimiters: <arg_value> (154849) .. </arg_value> (154850)"
# If that line is absent, the exemption did NOT resolve — then, and only then,
# set ATLAS_TOOL_ENVELOPE_WATCHDOG=0 as a stopgap.
AGENTIC="${AGENTIC:-0}"
if [ "$AGENTIC" = "1" ]; then
  export ATLAS_SIMHASH_LOOP="${ATLAS_SIMHASH_LOOP:-0}"
  export ATLAS_LOOP_NO_SUPPRESS="${ATLAS_LOOP_NO_SUPPRESS:-1}"
  export ATLAS_MAX_INTER_TOOL_PROSE="${ATLAS_MAX_INTER_TOOL_PROSE:-0}"
  echo "AGENTIC=1: simhash-loop OFF, loop-suppress OFF, inter-tool-prose cap OFF" >&2
  echo "  (tool-envelope watchdog stays ARMED — its GLM exemption is fixed at source)" >&2
fi

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
echo "  util=$GPU_UTIL ctx=$MAX_SEQ_LEN mtp=$MTP drafts=$DRAFTS prefix-cache=on agentic=$AGENTIC"
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
  --kv-cache-dtype "$KV_DTYPE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --oom-guard-mb "${OOM_GUARD_MB:-1024}" \
  --max-batch-size "$MAX_BATCH" \
  --swap-space-gb 0 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  $SPEC_ARGS \
  ${EXTRA_ARGS:-} \
  $TUI_ARG
