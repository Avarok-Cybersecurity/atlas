# Popping (or Not) the GPU Bubble: Atlas Holo-3.1 Prefill, Grounded

**Scope:** Why Atlas prefill is ~3× slower than vLLM on Holo-3.1-35B-A3B (GDN+MoE hybrid)
on a DGX GB10 (sm_121, 48 SMs, ~121 GB unified), and whether the Moondream "GPU bubble"
framing points at the right lever. This is an argued analysis, not a summary. Claims are
tagged **[verified]** (read in code/kernels this session), **[note]** (from memory notes,
treated as prior evidence), or **[inferred]**.

---

## 0. TL;DR for the senior engineer who knows this codebase

1. **The blog is about the wrong bottleneck for our prefill.** Moondream's "GPU bubble" is a
   *decode-step* phenomenon: tiny per-token forwards (2–10 ms) stalled by fixed CPU
   housekeeping, fixed by CUDA-graph replay + two-slot forward/sample pipelining. That is a
   real lever — **but we already apply its core mechanism to decode** (graph capture at
   `decode_a.rs:211/339`, launch at `:198`) and **prefill forwards are large, not tiny**, so
   the bubble it describes is not where our 3× lives. Apply the blog to *decode concurrency*,
   not prefill. **[verified]**

2. **"Prefill" is three different bottlenecks, and only one is addressable by software in a
   way that yields a multiple:**
   - **GDN/SSM scan (~24–38% of prefill):** *latency-bound on a serial recurrence*, NOT
     occupancy-bound. The occupancy "fix" the RFC/TFLA notes advocated was built and
     **measured to regress** (V-block split 0.34–0.71×). This is the hard one.
   - **MoE grouped GEMM (~26%, #1 at chunk≥2048):** *weight-bandwidth-bound* on GB-scale
     expert weights that don't fit L2. Batching tokens does NOT amortize. Near hardware roof.
   - **Attention (~tiny, <1%):** GDN is linear, only 10 quadratic layers → not a bottleneck
     at our context sizes. Ignore it for throughput.
   **[verified + note]**

3. **The "3× gap" headline is partly an artifact of how it was measured.** The most recent
   clean measurement **[note, 2026-06-29]** puts single-stream prefill at ~3,700 tok/s and
   shows large-context prefill is *compute-saturated by one stream, exactly like vLLM*
   (C1→C8 is 1.02× for both). Earlier "3×" and "9–13×" figures were harness bugs (tiny
   heterogeneous prompts; back-to-back probes queuing behind decode). **Before investing,
   re-establish the gap with the image binary + server-side TTFT.** If a real ~3× remains at
   7–16K, it is concentrated in the GDN serial scan + the per-kernel efficiency of the MoE
   GEMM, not in launch bubbles.

4. **The one place the blog's framing genuinely transfers to us and we are NOT fully
   exploiting it:** *prefill has no CUDA graphs and dispatches from a single thread.*
   **[verified]** At small chunks / mixed traffic, per-kernel launch overhead from one CPU
   core is a real tax (the notes already flagged this as "THE prefill CPU bottleneck"). This
   is the blog's actual contribution to *our* situation — but its upside is bounded (it helps
   small/medium chunks and co-dispatch, not the large-context compute wall).

---

## 1. What the blog actually claims (and what it doesn't)

Moondream, *"Popping the GPU Bubble"*:

- **Definition:** The bubble = idle GPU cycles where *"the GPU often sits idle, not for lack
  of work, but because the CPU hasn't told it what to do next yet."* Per decode step the CPU
  pays a **fixed** housekeeping cost (pick next requests, build metadata, sample the token,
  record it), and *"one token's worth of GPU work is small, while the CPU housekeeping is a
  fixed cost paid on every trip."*
- **Where divergence appears:** decode steady-state. They report **+6.5% (3090·1) →
  +35.4% (B200·32)** end-to-end, and explicitly: *"the win grows with GPU speed… the
  bookkeeping is GPU-speed-independent, so as the forward shrinks… the bubble is a bigger
  share of the step."* Per-step forwards are **2.45–10.24 ms**; sampling **0.14–0.26 ms**.
- **Identified bottleneck:** **CPU↔GPU synchronization / serialization**, NOT memory
  bandwidth or compute. The blocking loop *"plans and launches a forward, the GPU runs it,
  then the CPU synchronizes, waits for the results to land, commits them, and only then
  starts planning the next step."*
- **Techniques:** (1) **Ping-pong slots** with fixed buffer addresses so the step can be
  **captured once as a CUDA graph and replayed** (kills launch overhead) + a separate **copy
  stream** so D2H runs as background DMA; (2) **forward-now-sample-later** — launch forward
  t+1 before committing/sampling step t (hide CPU work behind GPU work); (3) **zombie
  refcounting** for finished-but-in-flight seqs; (4) a **unified prefill/decode pipeline**
  where *"a prefill forward can be launched into one slot while a decode step from the other
  slot is still being committed."* Result: steady-state idle **<0.05 ms/step**.

**The critical caveat for us:** every number and mechanism is about **decode**, where the
forward is small and the CPU fixed cost is comparable to GPU time. Their model is
moondream2 — a *dense* vision-language transformer. None of it addresses a **recurrent SSM
scan** or a **256-expert grouped GEMM**. The bubble shrinks to irrelevance the moment one
forward does seconds of GPU work, which is exactly our large-context prefill regime.

---

## 2. Our prefill, decomposed by bottleneck class

GB10: 48 SMs, ~273 GB/s… no — unified LPDDR5X ~**273 GB/s** class bandwidth (low for a
"GPU"), strong FP4/BF16 tensor throughput. **This bandwidth/FLOP ratio is the single most
important hardware fact** and it inverts a lot of dense-transformer intuition: on GB10 you
are bandwidth-starved far more easily than on an H100/B200.

### 2a. The GDN / SSM scan — **latency-bound serial recurrence** (the hard core)

**Verified kernel facts** (`kernels/gb10/common/gated_delta_rule_fla.cu`, dims K=V=128,
CHUNK=64, num_v_heads=32):

The scan is three FLA kernels. Two are parallel-over-chunks and already fill the machine:
- `recompute_wu` (`:136`): grid `(num_chunks, 32, 1)`, 128 thr/4 warps, ~49 KB smem, **uses
  wmma**. Parallel — not the bottleneck.
- `chunk_fwd_o` (`:1036`): grid `(num_chunks, 32, 1)`, ~98 KB smem, **uses wmma**. Parallel.

The bottleneck is the **state-carry** kernel:
- **Production dispatch = `chunk_delta_h_ksplit<SPLIT=2>`** (`:633`): grid `(32, 1, 1)`,
  **256 threads / 8 warps**, `__launch_bounds__(256,1)`, ~**99.8 KB smem** double-buffered
  cp.async, **pure scalar f32 FFMA — no tensor cores**. It computes
  `S_{c+1} = exp(gc_last)·S_c + K̃ᵀU − K̃ᵀ(WS)` **serially across all chunks** inside each CTA
  (one CTA per v-head). At batch=1 that is **32 CTAs on 48 SMs (~66% occupancy)**.

The decisive, code-resident verdict (comments at `:526`, `:629`, `:848–878`):
- SPLIT=4 (16 warps) gives **no gain** over SPLIT=2 (26.5 vs 26.3 ms) → *"8 warps already
  saturates the latency hiding… no longer occupancy-bound past that."* **[verified]**
- The V-block split — **the exact lever the croll83 RFC and the TFLA note advocated**
  (add a DV axis → 128 CTAs to fill the 48 SMs) — was **built, microtested bit-parity, and
  REGRESSES**: 0.71×/0.65×/**0.34×** at VTILES=2/4/8, monotonically worse. **[verified]**
  Verdict in the source: *"chunk_delta_h is NOT occupancy-starved at batch=1 — it is bound
  by the SERIAL CHUNK DEPENDENCY (S_{c+1} needs S_c) + per-CTA latency hiding (already
  saturated). More CTAs only add redundant W/U/K re-reads and shrink warps/CTA."*

**This directly refutes the "9.5× isolated via batch-axis + DV-block + wmma" hypothesis as a
single-stream prefill lever.** The RFC's 9.5× was at B=1,T=192 *in a different codebase* and,
crucially, **their e2e was neutral because GDN was 1–2% of their prefill.** Two of their
three levers (DV-block, more CTAs) are the ones we proved regress here; only the third (wmma
*inside the scan*) is unexplored, and the scan's serial dependency caps any wmma win.

**`tc_vblock`** (`gated_delta_rule_chunk_delta_h_tc_vblock`, `:684`, 82,952 B smem, wmma
Phase-A + DV-split) is **registered but DORMANT** — `ATLAS_GDN_TC_VBLOCK` default-off
(`ssm_gdn_a2.rs:395`), `ATLAS_GDN_BATCHED_FLA` default-off (`trait_prefill_gdn.rs:15`). The
isolated win it showed (1.4× **@batch=2**, ~1.05× @batch=4) is a *concurrency* win
(more streams → real CTAs), not a single-stream one. **[verified + note]**

**Bottleneck class: latency-bound (serial recurrence) + secondarily bandwidth (it
re-streams W/U/K per chunk).** Not compute-bound (scalar FFMA on tensor-core silicon is the
*symptom*, but you can't just MMA your way out — the dependency chain is the wall). Not
launch-bound. **This is the residual that most plausibly explains a real prefill gap vs
vLLM, because vLLM/FLA run a TFLA-style scan that keeps the state in HBM with small smem
tiles and many co-resident CTAs** — but note our own measurement says adding CTAs regressed,
so the vLLM win, if real, is per-CTA *throughput* of the inner matmuls (tensor-core scan),
not occupancy. **[inferred — this is the #1 thing to measure, §4.]**

### 2b. The MoE grouped GEMM — **weight-bandwidth-bound** (near the hardware roof)

**Verified** (`kernels/gb10/holo-3.1-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu`; dispatch
`crates/spark-model/src/layers/moe/forward_prefill_routed.rs`):
- Single fused launch, **grid.z = num_experts** (256), M_TILE=64, N_TILE 64/128, K_STEP 16,
  block `[128,1,1]`, ~3.5 KB smem, `mma.sync.m16n8k16` bf16→f32, NVFP4 E2M1→bf16 inline
  dequant. Production gate_up = FP8 fused; down has `ATLAS_MOE_PREFILL_FP8_DOWN=1` (−14% on
  the down GEMM). FP4 gate_up/down are opt-in and **e2e-neutral-to-negative** (the
  act-quant `cvt e2m1` overhead eats the microkernel win). **[verified + note]**
- **The architectural fact that governs this** (`prefill_b/batch.rs:24–51`, restated in
  notes): layer/expert weights are **GB-scale vs ~24 MB L2**, so **each request re-streams
  every weight byte; token-batching gives NO weight reuse.** The sibling `w4a16_gemm.cu`
  comments even tally it: M_TILE=64 → "16 weight re-reads → 227 MB B DRAM" vs M_TILE=128 →
  "8 re-reads → 114 MB". **[verified]**
- chunk-size ↔ MoE tile efficiency is *the* tuning knob: tokens/expert = C/32, M_TILE=64 ⇒
  **C=2048 = exactly one full tile = 100% MoE efficiency** (shipped default); C=1024 halves
  it. **[note, validated]**

**Bottleneck class: bandwidth-bound (weight streaming) with a fixed per-launch / act-quant
overhead at small M.** This is the same class the blog explicitly says it is *not* attacking.
On GB10's ~273 GB/s, MoE prefill is close to a hardware roof; the only software wins are
(a) bigger M-tiles to cut weight re-reads (M128 variant exists), (b) keeping precision low
without paying act-quant tax, (c) the CUTLASS 4.4.2 sm_121 NVFP4 **grouped** GEMM that now
exists but **is not wired in** (`forward_prefill_routed.rs` has an
`ATLAS_HOLO_MOE_GROUPED_CUTLASS` path, dormant). **[verified + note]**

### 2c. Attention prefill — **not a bottleneck**

**Verified + note:** Only 10 of 40 layers are full attention; GDN is linear. The 60K clean
profile **[note]** put flash_attn at **3.9 ms** and all q/k/v/o at ~50 ms across 10 layers —
sub-1% of a 28.6 s prefill. FlashInfer ragged prefill exists but is **opt-in
(`ATLAS_FLASHINFER_PREFILL=1`), chunk-0 only, HD=256 only** (`prefill/paged_attn.rs`).
**Do not spend effort here for throughput.** (It matters for *image/ViT* prefill — a separate
track already addressed by the GEMM-based ViT attention work in the notes.)

### 2d. The cross-cutting tax the blog DOES name: single-thread launch, no prefill graphs

**Verified:** Graph capture exists for decode/verify only (`decode_a.rs`, `verify_*.rs`);
**no `begin_capture`/`end_capture` anywhere in the prefill path.** Prefill dispatches every
kernel from one thread (`prefill_b/*` "single-stream dispatcher"). **[verified + note]** The
notes call this "THE prefill CPU bottleneck" and credit chunk=2048 partly for *halving kernel
launches*. This is the genuine Moondream-bubble analog in our engine — but it is a tax on
*small/medium* chunks and *co-dispatch*, and it shrinks toward zero as the chunk (and thus
GPU work per launch) grows. It cannot explain a 3× gap at 7–16K single-stream.

---

## 3. Prioritized levers (mechanism · upside · risk/effort · blog connection)

Ordered by expected gap-closure per unit risk, **conditioned on first re-confirming the gap
(§4)**.

### P0 — Re-measure to localize the gap (gate before any kernel work)
- **Mechanism:** `ATLAS_PROFILE=1` per-section prefill at the *production* chunk=2048 on the
  *image binary* (not a dev-box build), at the real shapes (7K/11K/16K), single-stream and
  C=4, vs the vLLM holo.service baseline at identical shapes. Pull GDN vs MoE vs proj split.
- **Upside:** Decides whether the residual is GDN (kernel rewrite) or MoE (bandwidth roof,
  largely unwinnable) or measurement. The notes already contain *contradictory* gap numbers
  (3× vs ~parity-at-large-ctx); this is unresolved and **must** be settled first.
- **Risk/effort:** Low. Half a day. **Blog tie-in:** none — but prevents chasing the bubble.

### P1 — Wire prefill into the launch-overhead escape (the blog's actual lever for us)
- **Mechanism:** Two complementary pieces. (a) **Co-dispatch reliability + default-on**:
  the +27%@C4 / 1.43–1.57× co-dispatch path is correctness-fixed (slot-collision fix,
  commit 22d54af) and gated behind `ATLAS_Q12_BATCHED_WITH_CACHE` / `ATLAS_PREFILL_CODISPATCH`
  — this is the *unified prefill/decode pipeline* the blog describes, and it fills occupancy
  gaps + overlaps cross-request kernels. (b) **Multi-stream / async copy** so prefill kernel
  launches and H2D metadata overlap instead of serializing on one thread.
- **Upside:** Real for *mixed/concurrent* and *small-chunk* traffic (the soak regime), where
  the launch tax is largest. **Bounded** for large single-stream prefill (compute-saturated).
  Realistically +20–40% aggregate under concurrency; ~0 for one big stream.
- **Risk/effort:** Medium. Co-dispatch correctness was a long bug hunt (SSM slot collisions);
  it's now fixed but needs a soak before default-on. The dedicated second-stream overlap
  (task 8) was **evaluated and rejected** (scratch aliasing `ssm_qkvz`/`logits`, GB-scale
  arena duplication) — do NOT rebuild that; co-dispatch strictly dominates it. **[note]**
- **Blog tie-in:** **Direct.** This is forward-now/pipeline + fixed-buffer graph replay
  applied to prefill. The piece we are NOT doing that the blog does: **capturing the prefill
  step as a CUDA graph.** Prefill shapes vary (chunk lengths), so a *single* graph won't do —
  but a small **cache of graphs keyed by (chunk_len bucket, batch)** is feasible and would
  cut the single-thread launch tax exactly as the blog predicts. **Highest-value blog-derived
  idea we haven't tried.**

### P2 — MoE: cut weight re-reads (bandwidth, the dominant *compute-phase* cost)
- **Mechanism:** Default the **M128 grouped variant** (8 weight re-reads vs 16 → halves B
  DRAM at large M) and ensure chunk=2048 keeps tiles full. Evaluate wiring the **CUTLASS
  4.4.2 sm_121 NVFP4 grouped GEMM** (356 TFLOP/s claim) into the dormant
  `ATLAS_HOLO_MOE_GROUPED_CUTLASS` branch — a real per-kernel-efficiency win vs the
  hand-rolled path, *if* it beats it after act-quant.
- **Upside:** MoE is ~26% of prefill; halving its weight traffic could net ~10% e2e. The
  CUTLASS grouped path is the one credible "vLLM uses better kernels" lever for MoE.
- **Risk/effort:** Medium. M128 is low-risk (kernel exists). CUTLASS grouped-GEMM wiring is
  real integration + an A/B that prior pin-bump showed was **neutral until wired** (the
  kernels sit in the image unused). FP4-everything was already proven e2e-neutral — don't
  re-chase precision alone. **[verified + note]**
- **Blog tie-in:** None (blog disclaims bandwidth). This is hardware-roof work.

### P3 — GDN scan: attack the *recurrence*, not the occupancy (the genuine hard lever)
- **Mechanism:** The only paths with headroom, per our own data, are **(a) tensor-core-ize
  the inner per-chunk matmuls of the serial scan** (W·S, K̃ᵀU) while keeping f32 state — i.e.
  finish the wmma Phase-B that `tc_vblock` left scalar (the batch≥4 plateau came from scalar
  Phase B); **(b) a TFLA-style HBM-resident state** with small smem tiles to *raise per-CTA
  throughput* (not CTA count). Or **(c) integrate FlashInfer's Blackwell GDN prefill** kernel
  (runs on GB10 via cuda-compat; ~17 TFLOP/s; needs AOT cubin export + launch-ABI replay).
- **Upside:** Potentially the largest *single-stream* prefill win **if** the residual gap is
  really in GDN — but capped by Amdahl (24–38%) and by the serial dependency. A 2× scan →
  ~+15–25% e2e at most.
- **Risk/effort:** **High.** This is the one the notes have circled for weeks and where every
  occupancy approach has regressed. wmma-Phase-B is the most surgical bet. FlashInfer-GDN is
  the "borrow vLLM's kernel" route but carries AOT-export + container-driver shipping risk and
  is **not a drop-in.** Do this **only if P0 proves GDN is the residual.** **[verified + note]**
- **Blog tie-in:** None. The blog has nothing to say about a recurrent scan; this is the part
  of our problem its framing entirely misses.

### P-skip — explicitly do NOT pursue (already settled, with evidence)
- GDN **occupancy / V-block / more-CTAs at batch=1** — regresses (§2a). **[verified]**
- **Dedicated second prefill stream** (task 8) — rejected, scratch aliasing. **[note]**
- **FP4-everything for MoE/projections as a speed play** — e2e neutral-to-negative; cuBLAS
  BF16 / FP8-blockscaled already win. **[note]**
- **cuBLAS grouped GEMM for FP4 MoE** — dead end (FP8/BF16 only; dequant cost). **[note]**
- Treating **attention** as a throughput target. **[verified]**
- Batched-prefill expecting **weight reuse** — there is none on GB10. **[verified]**

---

## 4. What to measure first (so we don't chase the bubble)

The dominant uncertainty is **§3-P0**: is the residual gap GDN, MoE, or measurement? Concrete
counters/kernels, in order:

1. **Confirm the gap exists at the production config.** Server-side TTFT (lifecycle.rs
   "Done… TTFT=") at chunk=2048, image binary + full `holo_serve.sh` flags, 7K/11K/16K,
   spaced (no decode queuing). Compare to vLLM holo.service same shapes. *If single-stream is
   already ~3,700 tok/s and within ~1.3× of vLLM at large ctx, the "3×" is stale and the real
   target is concurrency (P1), not kernels.* **[note says this is likely]**

2. **Per-section split at production chunk** (`ATLAS_PROFILE=1`): the prior clean split was
   taken at chunk=1024 (half-empty MoE tiles → MoE overstated ~2×). Re-take at 2048 to get
   the **honest GDN-vs-MoE ratio.** Expect GDN to be #1 there.

3. **GDN scan isolation:** `chunk_delta_h_ksplit` wall-time at the real per-layer shape vs
   **FlashInfer's GDN kernel at the same shape** (the bench harness exists,
   `scripts/gdn_bench_holo.py`; FI ≈ 17 TFLOP/s at 8K). This single A/B decides whether a scan
   rewrite can actually beat us — *we have never run this head-to-head at the Atlas shape* (the
   note flags it as "the deciding number"). If FI isn't materially faster per-layer, P3 is
   dead and the gap is MoE/measurement.

4. **MoE GEMM achieved bandwidth:** ncu (or the in-kernel DRAM tally) on
   `moe_w4a16_grouped_gemm` at M=2048/expert — compare achieved B-matrix GB/s to GB10's
   ~273 GB/s roof. If it's already >70% of roof, MoE is done; if not, M128 + better tiling
   (P2) has headroom.

5. **Launch-tax quantification (the blog's metric):** with `ATLAS_PROFILE`, measure CPU
   dispatch gap between kernels at chunk=2048 single-thread. If steady-state inter-kernel idle
   is large, the prefill-graph cache (P1) pays; if the GPU is already back-to-back busy (the
   note says "GPU 0% idle" at large ctx), the bubble is closed for big prefills and P1 only
   helps small/concurrent.

**Decision tree:** (2)+(3) point at GDN → P3 (wmma Phase-B first). (4) shows MoE below roof →
P2. (5) shows inter-kernel idle → P1 graph cache. (1) shows parity-at-large-ctx → declare
single-stream done, move all effort to **decode/prefill concurrency** where the blog's
pipeline actually applies and where the measured headroom (decode 1.93×@C8 → 2.09×@C12) lives.

---

## 5. Bottom line

The Moondream post is a good post about the wrong layer for our prefill problem. Its real gift
to us is the reminder that **decode-shaped, launch-bound work** should be graph-captured and
pipelined — which we do for decode and *don't yet do for prefill*, making a **bucketed prefill
CUDA-graph cache + reliable co-dispatch (P1)** the one blog-derived lever we're leaving on the
table. But our 3× — to the extent it survives an honest re-measurement (§4 item 1, which the
latest note suggests it may not at large context) — lives in two places the blog explicitly
does not address: a **serial GDN recurrence** that is latency-bound and resists every
occupancy trick we've tried, and a **256-expert grouped GEMM** that is weight-bandwidth-bound
on a 273 GB/s machine. Those are kernel-efficiency and hardware-roof problems, not bubbles.
Measure (§4) before building (§3-P3), because the prior occupancy-shaped intuitions — the RFC's
9.5×, batch-axis, DV-block — were tested in *our* tree and **regressed**, and the serial scan,
not idle SMs, is the wall.
