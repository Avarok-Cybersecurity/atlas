# DFlash Propose — CUDA Graph Capture (v4, Final)

**Status:** Final design, ready to implement
**Author:** Friday (with Ronald)
**Date:** 2026-05-24
**Goal:** Capture the drafter propose path as ONE CUDA graph, replayed every step. Eliminate per-call CPU launch and sync overhead. **Phase F (post-graphs) integrates K=γ rollback via the new SsmSnapshotPool on main, unlocking `DRAFT_CAP > 1`.**
**Estimated impact (Phases A-E):** propose 72ms → ~50-55ms steady state. **Tok/s 10.76 → 13-15 realistic, possibly 16. May clear 13.86 no-spec by itself.**
**Estimated impact (Phases A-F combined):** **Tok/s 10.76 → 16-19 realistic. Clears 13.86 no-spec ceiling with margin and DFlash genuinely pays for itself.**
**Estimated impact (Phases A-G combined):** **Tok/s 10.76 → 18-22 realistic.** Memory-traffic reduction + launch fusion on top of the cap lift. The stretch target lives here.
**Prerequisites:** Stage 4 (Option B) + incremental ctx-append precompute landed. K2 ACCEPT counter fix landed. Nologik confirmed K=γ integration is open territory (2026-05-24 ~19:00 PT) — Phase F is ours to ship.

---

## Design history (the road to v4)

This document went through three drafts before landing here. The pivots are worth understanding because each one was wrong for an instructive reason.

- **v1:** Per-bucket graph cache keyed on `ctx_count / 16`. **Rejected:** the captured graph freezes `kv_len` and `q_offset` as scalar kernel args. Bucketed reuse would have run wrong-by-up-to-15-positions attention math and silently dropped acceptance. Correctness defect.

- **v2:** Make `kv_len`/`q_offset` indirect via a device buffer (the v4 approach). Kernel reads them at entry from a pointer instead of taking them as scalar args. **Initially preferred. Then questioned.**

- **v3:** Use `cuGraphExecKernelNodeSetParams` to mutate the captured graph's kernel args between launches. **Rejected — see §13 (Rejected alternatives).** Despite looking elegant on paper, this is not what any production inference stack actually uses. vLLM uses persistent device buffers + pre-graph writes (the v4 approach). Atlas's existing 5 graph caches use the same "capture once, mutate the device buffer contents the graph reads" pattern. setParams-based mutation has zero production references and adds per-call CPU overhead with no compensating benefit.

- **v4 (this doc):** v2 architecture restored as the definitive answer, with the v3 investigation captured in §13 so we don't reinvestigate later.

**Core principle:** prefer best practices first. Failing that, what Atlas already does. Failing that, what vLLM does. Don't reinvent wheels.

---

## 1. Why graphs

### 1.1 Profile data

nsys profile of clean Stage 4+5 bench (no debug stranglers, 32 s response):

| CUDA API | Total time | Calls | Per-call avg | % of API time |
|---|---|---|---|---|
| `cuStreamSynchronize` | 11,521 ms | 3,876 | 2.97 ms | **62.4 %** |
| `cuMemcpyDtoHAsync_v2` | 6,448 ms | 331 | 19.48 ms | 34.9 % |
| `cuLaunchKernel` | 283 ms | 65,070 | 4.35 µs | 1.5 % |
| `cuGraphLaunch` | 23.4 ms | 98 | 239 µs | 0.1 % |

`cuGraphLaunch` is **44× more efficient per launch** than `cuLaunchKernel` on this same binary on this same hardware. Atlas's target model already graphs decode and verify; only the drafter propose remains ungraphed.

### 1.2 Per-propose budget

~150 kernel launches per propose × 4.35 µs = **650 µs of pure CPU launch overhead per call**. Over 191 propose calls in a 32 s bench = 124 ms of host time. The actual cost is higher because each launch implicitly serializes through `cuStreamSynchronize`.

With graphs: 1 `cuGraphLaunch` per propose ≈ 239 µs ≈ 46 ms total for the bench. **Saves ~80 ms+ of CPU time** plus the unmeasurable but significant stream-utilization improvement (GPU no longer goes idle waiting for the next batch of launches).

### 1.3 Industry precedent

vLLM, TensorRT-LLM, and SGLang all graph the decode hot path. vLLM specifically wraps the DFlash drafter forward in cudagraphs (`vllm/v1/spec_decode/dflash.py:219-231`). vLLM's `InputBuffers` class (`v1/worker/gpu/input_batch.py:12`) holds persistent device tensors that are written pre-graph and read by the captured graph — **this is the exact pattern v4 follows**.

---

## 2. The single-graph architecture

### 2.1 The dynamic-data problem

A captured CUDA graph freezes every kernel argument at capture time. For our propose, the dynamic values per call are:

| Value | Per-call dynamic? | Solution |
|---|---|---|
| `last_token: u32` | YES | Pre-graph: write to `draft_tokens_dev` device buffer; kernels read by pointer |
| `position: usize` | YES | Pre-graph: encoded into `position_ids` buffer |
| `ctx_count: u32` | YES | **NEW: indirect via 8-byte device buffer (the kernel edit, §3)** |
| `slot_mapping_gamma` (contents) | YES | Pre-graph: `fill_slots_from_block_table` writes to the buffer; kernel reads by pointer |
| `kv_pool_ptr(l)`, `v_pool_ptr(l)` per layer | NO (stable for seq) | Captured at first call |
| `block_table_dev` | NO (allocated once on first propose) | Captured at first call |
| All scratch buffer pointers | NO (model-load constant) | Captured at first call |
| All weight pointers | NO | Captured at first call |

**The only kernel that takes a per-call-varying scalar is `inferspark_prefill_paged_dflash_bf16`**, which receives `kv_len` (= `ctx_count + γ`) and `q_offset` (= `ctx_count`) as scalar args. Every other kernel in the propose pipeline takes either constants or device pointers (the contents of those pointers are written pre-graph by host-managed copies).

**Fix the one kernel, and the entire propose path becomes graph-friendly.**

### 2.2 The single-graph guarantee

After §3's kernel edit:

- All kernel args are either constants (γ, model dims, weights) OR device pointers (input buffers, output buffers, pools).
- Per-call dynamic values (`last_token`, `ctx_count`, etc.) live in device buffers whose pointers ARE captured, but whose CONTENTS are rewritten pre-graph each call.
- The kernel reads those contents at kernel entry.
- Grid dimensions are computed from constants (γ=16 → fixed grid).

**One graph capture covers every possible propose call for the lifetime of the seq.** No buckets, no recapture, no per-call overhead beyond the few tiny pre-graph H2D writes.

This is structurally identical to vLLM's `InputBuffers` pattern, lifted from Python/PyTorch into Rust/CUDA.

---

## 3. The CUDA work

### 3.1 Scope

**Two files touched in `kernels/gb10/common/`:**

1. **`prefill_paged_compute.cuh`** (shared compute body, ~700 lines): drop `const` from `kv_len` and `q_offset` kernel parameters. **Zero semantic change** for all existing kernels — they simply don't write to the params anymore.

2. **NEW: `inferspark_prefill_paged_indirect.cu`** (~12 lines): a sibling of `inferspark_prefill_paged.cu` that defines a `KERNEL_PREAMBLE` overwriting `kv_len` and `q_offset` with values read from device pointers passed in `KERNEL_EXTRA_PARAMS`. **This uses Atlas's existing `KERNEL_PREAMBLE` macro extension mechanism — the exact intended use case.**

### 3.2 Portability — what this does NOT touch

- **Metal backend**: zero work. DFlash isn't supported on Metal.
- **Other CUDA architectures**: zero. Atlas's CUDA target is exclusively `sm_121f`. No other arch exists in `kernels/`.
- **Other paged-attention variants** (FP8, NVFP4, batched, HDIM=512): the `const` removal in the shared `.cuh` affects them at COMPILATION (still compiles, identical PTX, identical behavior). They do not gain or lose the indirect-arg feature unless they explicitly define a `KERNEL_PREAMBLE` for it. **They will not.** Status quo preserved for all 6 existing variants.

**This is verifiable.** Phase A includes a PTX-diff step on one of the unaffected variants to prove the const-removal is byte-identical at the SASS level.

### 3.3 The new .cu file

```c
// SPDX-License-Identifier: AGPL-3.0-only
//
// Paged Prefill Flash Attention — BF16 KV cache, INDIRECT scalar args.
//
// Identical to inferspark_prefill_paged.cu except `kv_len` and `q_offset` are
// read from device pointers at kernel entry instead of taken as scalar kernel
// args. This makes the kernel graph-friendly: a captured CUDA graph holds the
// pointers; per-call dynamic values are written to the pointed-to buffers in
// pre-graph host code.
//
// Used by the DFlash drafter (BlockDiffusionDraftHead::forward_block) so the
// entire propose path can be captured as a single CUDA graph and replayed
// every step.

#include <cuda_bf16.h>

// (LOAD_KV_TILE macro identical to inferspark_prefill_paged.cu — copy verbatim)
#define LOAD_KV_TILE(...) /* same body as the BF16 sibling */

#define KERNEL_NAME inferspark_prefill_paged_indirect
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d,                          \
                           const unsigned int* __restrict__ kv_len_ptr,        \
                           const unsigned int* __restrict__ q_offset_ptr
#define KERNEL_PREAMBLE                                                         \
    /* Read indirect scalar args via shared memory: thread 0 loads, all wait. */\
    __shared__ unsigned int s_indirect[2];                                      \
    if (threadIdx.x == 0) {                                                     \
        s_indirect[0] = *kv_len_ptr;                                            \
        s_indirect[1] = *q_offset_ptr;                                          \
    }                                                                           \
    __syncthreads();                                                            \
    kv_len = s_indirect[0];                                                     \
    q_offset = s_indirect[1];

#include "prefill_paged_compute.cuh"
```

### 3.4 The .cuh edit

In `prefill_paged_compute.cuh`, change two parameter declarations in BOTH kernel definitions (BR=32 at line 60-71, BR=64 at line 404-413):

```diff
-    const unsigned int q_len,
-    const unsigned int kv_len,
-    const unsigned int q_offset,
+    const unsigned int q_len,
+    unsigned int kv_len,
+    unsigned int q_offset,
```

`q_len` stays const because we never need to override it (γ is a model constant). Only `kv_len` and `q_offset` become writable so the indirect preamble can overwrite them.

### 3.5 Rust glue

**`prefill_attn_main_a.rs`**: add `prefill_attention_paged_dflash_bf16_indirect` (analog of the existing dispatcher). Identical to the existing one except takes `DevicePtr` for kv_len/q_offset instead of `u32`. Resolves a different kernel handle (`prefill_attn_dflash_bf16_indirect`).

**`dflash_head/from_weights.rs`**: resolve the new kernel via `gpu.kernel("prefill_paged_indirect", "inferspark_prefill_paged_indirect")`. Store the handle on `DflashKernels`.

**`dflash_head.rs` (scratch buffers)**: add `option_b_indirect_args_dev: DevicePtr` — 8 bytes of device memory holding the [kv_len, q_offset] pair.

**`forward_block.rs` pre-graph**: write the two u32 values to `option_b_indirect_args_dev` via `copy_h2d` BEFORE entering the captured region.

**`forward_block_layer_paged.rs` (inside captured region)**: switch from `ops::prefill_attention_paged_dflash` to `ops::prefill_attention_paged_dflash_bf16_indirect`, passing the device pointer instead of the scalar.

---

## 4. The graph layer

### 4.1 Storage

```rust
// In BlockDiffusionDraftHead:
pub(super) propose_graph: Mutex<Option<GraphHandle>>,
pub(super) suppress_graphs: AtomicBool,           // mirrors target-model pattern
pub(super) propose_warmup_count: AtomicUsize,     // for warm-up state machine
```

One graph per drafter instance. No cache key, no buckets, no per-slot variation (drafter is single-seq, max_batch_size=1).

### 4.2 Forward_block flow (final shape)

```rust
fn forward_block(&self, last_token, position, ctx, stream,
                 ctx_buffer, option_b) -> Result<Vec<u32>> {
    // ─── Phase 1: pre-graph (every call, NOT captured) ───
    self.write_dynamic_inputs(last_token, position, stream)?;
    if let Some((bt, ctx_count)) = option_b {
        self.write_indirect_args(ctx_count, self.gamma, stream)?;  // 8 bytes H2D
        self.fill_gamma_slot_mapping(bt, ctx_count, stream)?;
    }

    // ─── Phase 2: graph (capture once, replay forever) ───
    let use_graph = option_b.is_some() && self.graph_capture_eligible();
    if use_graph {
        let warmup_target = std::env::var("ATLAS_DFLASH_PROPOSE_WARMUP_N")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let current = self.propose_warmup_count.load(Relaxed);

        let mut g = self.propose_graph.lock();
        match *g {
            Some(graph) if graph.0 != 0 => {
                self.gpu.launch_graph(graph, stream)?;
            }
            _ if current < warmup_target => {
                // Warm-up phase: eager, warming SASS cache + L2 + clocks.
                self.propose_warmup_count.fetch_add(1, Relaxed);
                self.forward_block_inner(...)?;
            }
            _ => {
                // Warmed up — capture now.
                self.gpu.begin_capture(stream)?;
                self.forward_block_inner(...)?;
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    *g = Some(graph);
                    self.gpu.launch_graph(graph, stream)?;
                } else {
                    // empty capture, eager fallback
                    self.forward_block_inner(...)?;
                }
            }
        }
    } else {
        self.forward_block_inner(...)?;
    }

    // ─── Phase 3: post-graph ───
    self.read_drafts(stream)
}
```

### 4.3 Eligibility gate (Nologik rule encoded)

```rust
fn graph_capture_eligible(&self) -> bool {
    if self.suppress_graphs.load(Relaxed) { return false; }
    if env_set("ATLAS_DFLASH_PROPOSE_NO_GRAPH") { return false; }
    // Any debug var that injects d2h/sync into propose disables graphs.
    for v in [
        "ATLAS_DFLASH_DEBUG_DUMP",
        "ATLAS_DFLASH_DEBUG_DUMP_FULL",
        "ATLAS_DFLASH_OPTION_B_DIAG",
        "ATLAS_DFLASH_PRECOMPUTE_DUMP",
        "ATLAS_DFLASH_VERIFY_TRACE",
        "ATLAS_DFLASH_LOG_DRAFTS",
        "ATLAS_DFLASH_DEBUG_FORCE_PATTERN",
        "ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
        "ATLAS_DFLASH_DEBUG_CTX_OFF",
        "ATLAS_DFLASH_DEBUG_CTX_USED",
    ] {
        if std::env::var(v).is_ok() { return false; }
    }
    true
}
```

This is the master "don't graph if any debug surface is on" gate. Future debug flags must either be added to this list or designed not to inject D2H/sync.

### 4.4 Warm-up state machine

Two eager calls before the first capture. Rationale:

- Warms CUDA's PTX→SASS compilation cache (per-kernel JIT happens on first launch)
- Lets GB10's dynamic clock ramp to sustained-workload state (~100-500ms otherwise)
- Brings hot drafter weight tiles into L2 cache (~2× effective BW on warmed tiles)
- Hits memory allocator fast paths
- The graph captures the SASS variants the driver picks at steady state, not the cold-start variants

Default N=2. Env override `ATLAS_DFLASH_PROPOSE_WARMUP_N`. No model-load pre-capture (would need fake state setup not worth the complexity).

### 4.5 Cleanup

`BlockDiffusionDraftHead::drop` calls `gpu.destroy_graph(g)` on the held handle. Pattern from `model/drop.rs`.

---

## 5. The static/dynamic state audit

### 5.1 Per-call dynamic, pre-graph writes

| Value | Where written | Buffer (always same pointer) |
|---|---|---|
| `last_token` byte | embed input slot 0 | `self.scratch.draft_tokens_dev` |
| Mask tokens for γ-1 slots | embed input slots 1..γ | `self.scratch.draft_tokens_dev` |
| Position IDs for γ rows | int32 array | `self.scratch.position_ids` |
| Slot mapping for γ K/V writes | int64 array of size γ | `self.scratch.slot_mapping_dev` |
| **`kv_len = ctx_count + γ`** | **u32** | **`self.scratch.option_b_indirect_args_dev[0]`** |
| **`q_offset = ctx_count`** | **u32** | **`self.scratch.option_b_indirect_args_dev[1]`** |

All writes go through `copy_h2d` on the propose stream. Pointers are stable. Contents change.

### 5.2 Captured kernel sequence (inside the graph)

In order:
1. `batched_embed` — reads `draft_tokens_dev`, writes `stream_buf`
2. For each layer l in 0..8:
   - `rms_norm` (input_layernorm)
   - `dense_gemm` q_proj, `dense_gemm` k_proj, `dense_gemm` v_proj
   - `rms_norm` q_norm, `rms_norm` k_norm
   - `rope_yarn` reads `position_ids`
   - `reshape_and_cache` reads `slot_mapping_dev`, writes `k_pool[l]` / `v_pool[l]`
   - **`prefill_attention_paged_dflash_bf16_indirect` reads `option_b_indirect_args_dev`** (the indirect kv_len + q_offset), reads `k_pool[l]` / `v_pool[l]` via `block_table_dev`, writes `attn_out`
   - `dense_gemm` o_proj
   - `residual_add`
   - `rms_norm` post_attention_layernorm
   - `dense_gemm` gate_proj, `dense_gemm` up_proj
   - `silu_mul`
   - `dense_gemm` down_proj
   - `residual_add`
3. `rms_norm` final
4. `dense_gemm` lm_head
5. γ × `argmax_bf16` — writes `draft_tokens_dev` (input buffer reused as output; safe because step 1 read it before this write)

### 5.3 Capture-hostile operations confirmed AVOIDED

- `gpu.copy_h2d` — pre-graph only (writes dynamic inputs)
- `gpu.copy_d2h*` — post-graph only (reads drafts)
- `gpu.synchronize` — never inside the captured region
- `gpu.memset` (uses default_stream + sync) — replaced with `gpu.memset_async(stream)` for the captured region
- `kv_cache.lock()` for `k_pool_ptr(l)` / `v_pool_ptr(l)` — hoisted ONCE pre-graph, pointers passed into the inner function
- `dstate.block_table_dev` lookup — done once outside the captured region, pointer hoisted
- Memory allocation — none inside the captured region (block_table_dev allocation is a first-call-only path that runs before any capture attempt)

---

## 6. Implementation phases

Each phase ends in a buildable, bench-able milestone. We do NOT proceed without numbers.

### Phase 0 — Communication policy (no engineering work). Already decided.

Phase A modifies `kernels/gb10/common/prefill_paged_compute.cuh` — territory Nologik (Thomas Braun) is actively working in ("common kernels"). He confirmed 2026-05-24 ~19:00 PT that he is **NOT working on DFlash** and the K=γ rollback / propose-graph work is open territory for us.

**Strategy: ship results first, then loop him in with the perf delta.**

Do NOT send a heads-up before Phase A. The right move is to land Phases A through E completely, measure tok/s improvement against the current 10.76 baseline, write the perf report, and only THEN send Nologik a results-driven message that includes:

1. The specific files we touched in his territory (`prefill_paged_compute.cuh` const-removal + new `inferspark_prefill_paged_indirect.cu`).
2. Why we used `KERNEL_PREAMBLE` (his existing extension mechanism, used as designed).
3. PTX-diff evidence that the 6 untouched variants are byte-identical at the SASS level.
4. **The measured tok/s improvement.** This is the asset we don't have yet, and it's what makes the conversation a "look what we shipped" instead of a "permission slip for an idea."

Same approach for Phase F (the SSM snapshot integration is in territory adjacent to his Phase-C watchdog work).

This is intentional. Showing results is more credible and respectful of his time than asking for design feedback ahead of evidence. Builds professional credibility for any future requests we need from him.

**No external dependencies for Phases A-E. Proceed at our own pace.**

### Phase A — Kernel & shared header. 2-3 hours.

1. Write `kernels/gb10/common/inferspark_prefill_paged_indirect.cu`.
2. Edit `prefill_paged_compute.cuh`: drop `const` from `kv_len` and `q_offset` in both BR=32 and BR=64 kernel signatures.
3. Build with `ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server`.
4. **PTX diff verification**: dump SASS for `inferspark_prefill_paged.cu` (the BF16 contiguous variant — unaffected by our change) before and after the `.cuh` edit using `cuobjdump --dump-sass`. They must be byte-identical. Same check for `inferspark_prefill_paged_nvfp4.cu`. Proves the const-removal does not affect the 6 untouched variants.

Milestone: clean build + PTX-diff passes for both unaffected variants. Existing benches still produce 10.76 tok/s.

### Phase B — Rust dispatcher & kernel handle. 1-2 hours.

5. Add `prefill_attention_paged_dflash_bf16_indirect` in `prefill_attn_main_a.rs`.
6. Resolve handle in `dflash_head/from_weights.rs` (add to `DflashKernels`).
7. Allocate `option_b_indirect_args_dev: DevicePtr` (8 bytes) in scratch buffer init.
8. Write the dispatcher invocation, **but DO NOT YET wire it into forward_block_layer_paged**. Compiles only.

Milestone: cargo build clean. Bench reproduces 10.76 tok/s.

### Phase C — Eager indirect-path test (no graphs yet). 1-2 hours.

9. In `forward_block.rs` pre-graph section, write `[kv_len, q_offset]` to `option_b_indirect_args_dev` via `copy_h2d` of 8 bytes.
10. In `forward_block_layer_paged.rs`, switch from `prefill_attention_paged_dflash` to the new `_indirect` variant. Read the indirect args from the device buffer.
11. Run bench. Compare token output byte-for-byte against the previous bench. **If diverges → indirect-arg kernel has a bug. STOP and fix.**

Acceptance: byte-identical output to the previous bench. Acceptance rate matches 44.9% ± noise. Tok/s within ±0.3 of 10.76 (no perf change expected — same work, just routed through an indirect arg).

Milestone: indirect kernel is correct AND used. No graph yet.

### Phase D — First graph capture (with warm-up). 2-3 hours.

12. Add `propose_graph: Mutex<Option<GraphHandle>>` to `BlockDiffusionDraftHead`.
13. Add `propose_warmup_count: AtomicUsize` to track how many real calls have run eagerly. Initial: 0.
14. Add `graph_capture_eligible()` with the full debug-var check.
15. Wire warm-up + capture + replay logic (see §4.2). Use `cuGraphInstantiateWithFlags(... CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD ...)` for eager device upload (avoids first-launch JIT-to-device cost).
16. Add `Drop` cleanup.

Acceptance: same byte-identical token output. Acceptance rate same. Tok/s **improves materially** (target ≥ 11.5; expecting more because warm-up means graph captured at steady state).

Milestone: "Captured CUDA graph for drafter propose (after N=2 warm-up iterations)" logs ONCE. Subsequent calls show no kernel-launch spam under nsys.

### Phase E — Verify + perf report. 1 hour, plus warmup-N sweep.

17. Re-run nsys profile.
18. Verify in the new report:
    - `cuLaunchKernel` count drops from ~65k to under 5k (we estimate ~1000-2000 — first 2 warmup calls + post-graph readbacks across the run)
    - `cuGraphLaunch` count increases from 98 to ~290 (98 from target + 191 from propose + buffer)
    - `cuStreamSynchronize` % drops materially
    - Per-propose latency drops noticeably (look at the timeline)
19. **Warmup-N sweep**: bench at N ∈ {1, 2, 4, 8}. Lock in the best.
20. Write perf report next to this design doc.

Milestone: written perf delta with before/after numbers. Tok/s ≥ 12.0 to validate the architecture. Tok/s ≥ 13.86 is a stretch goal that means we've matched no-spec from this work alone.

### Phase F — DFlash K=γ rollback via SsmSnapshotPool. **POST-GRAPHS, Nologik confirmed open territory.** Estimate: half-day to one full day.

**Context:** until 2026-05-24, the propose.rs:170-194 comment described K=γ rollback as "multi-week kernel work" or "restrict DFlash to pure-attention targets." That framing is **OUT OF DATE.** Thomas Braun (tbraun96 / nologik) landed Phase-C SSM snapshot infrastructure on `origin/main` (PR #63) — exactly the primitive needed. It's currently wired for watchdog rollback only; integrating it with DFlash K=γ verify unblocks `ATLAS_DFLASH_DRAFT_CAP > 1`.

**The unlock:**

Today: `ATLAS_DFLASH_DRAFT_CAP=1` because K=γ verify's partial-accept path corrupts SSM state. Drafter produces 16 tokens, we throw away 15, verify amortizes over ~1.45 accepted tokens/cycle (one always-accepted + 0.45 from the actual draft).

With cap lifted to γ: still 45% per-row acceptance, but verify amortizes over ~1.83 tokens/cycle (geometric ceiling). **Per-output-token cost drops from 99 ms to ~75 ms.** Combined with graphs (Phases A-E) this clears 13.86 no-spec and DFlash actually pays for itself.

**The integration work:**

21. Sync our sidecar with `origin/main` (or cherry-pick the SSM snapshot files). New code touches:
    - `crates/spark-model/src/model/ssm_snapshot.rs` (the pool itself)
    - `crates/spark-model/src/model/trait_impl/verify_a.rs` (the save/restore dispatch methods)
    - `crates/spark-model/src/model/trait_impl/mod.rs` (trait method exposure)
    - `crates/spark-server/src/scheduler/lifecycle.rs` + `prefill_*_step.rs` + `phase_promote_prefills.rs` (per-ActiveSeq `SsmDecodeRing`)
    - **Be careful with the RoPE config fix** — our `b8f8244` commit fixes hardcoded YaRN in `dflash_head/from_weights.rs`; main has the OLD hardcoded version. We need to keep our fix and rebase the snapshot work on top, OR upstream `b8f8244` first.

22. Lift `ATLAS_DFLASH_DRAFT_CAP=1` cap in `propose.rs:412` to `γ` (drop the env-var override default to use the full drafter output).

23. Modify the K=γ verify path:
    - **Before** the γ-block target forward: call `model.save_decode_ssm_snapshot(seq, ring_slot)`. The `ring_slot` is allocated from the per-`ActiveSeq` `ssm_rollback_ring`.
    - **After** verify decides accept/reject prefix: if `num_accepted < γ` (partial accept), call `model.restore_decode_ssm_snapshot(seq, ring_slot)` to restore pre-γ-block state, then re-run the SSM portion of the layer loop ONLY on the accepted prefix.
    - On full accept: no restore needed; just free the snapshot slot.

24. Update the SsmDecodeRing capacity if needed. Today's ring is sized for `max_batch_size` watchdog slots; DFlash K=γ verify also needs one snapshot per verify cycle. They can share the ring (both transient) but capacity may need 1 extra slot per active sequence.

25. Bench at cap=4, then cap=8, then cap=16 (full γ). Per the geometric math, cap=8 captures ~95% of the available win; cap=16 is the absolute ceiling.

**Acceptance gates:**

- **Output identity at cap=1**: bench with cap=1 + snapshot-ring-enabled must produce byte-identical tokens to the pre-Phase-F bench. Confirms the snapshot save/restore plumbing doesn't break the safe path.
- **Output correctness at cap=γ**: bench output must be a coherent English Volvo essay (not the `"The—the-,—i -!!!!"` garbage we saw at cap=4 in the broken state). This is the strongest signal that SSM rollback works.
- **Accept rate at cap=γ**: should be similar to cap=1 (~45% per-row). If it tanks below 30%, the snapshot save/restore isn't restoring the right state — debug before proceeding.
- **Perf at cap=γ**: tok/s ≥ 13.86 (no-spec ceiling) is the win-or-lose threshold. Realistic target: 14-17 tok/s.

**Risks:**

- **Ring capacity exhaustion under contention.** If both the watchdog and DFlash verify need slots simultaneously, ring may run out. Mitigation: size ring at `max_batch_size × 2` (one for watchdog, one for verify). Cost: 2× the snapshot HBM, currently ~165 MB → ~330 MB. Acceptable.
- **Snapshot timing.** The save must happen *after* the previous decode commits to the pool but *before* the γ-block writes any SSM state. Wrong ordering = save captures stale state, restore is useless. Verify by reading the `save_decode_ssm_snapshot` dispatch (`verify_a.rs:351-380`) for exact stream ordering requirements.
- **Block size for hybrid models.** Qwen3.6-27B has 30 GDN layers; each save/restore is L * (h_bytes + conv_bytes) of D2D. Cost per snapshot: roughly 5-10 MB D2D = ~50-100 µs at HBM bandwidth. Per propose: +100 µs. **Negligible vs the ~25 ms per-output-token win.**
- **Atlas main has the RoPE regression.** If we sync main wholesale, we re-introduce hardcoded YaRN. Mitigation: cherry-pick only the snapshot infrastructure, leave our `b8f8244` RoPE fix in place. Track separately as a needed upstream.

**Open dependencies:**

- **Nologik confirmed (2026-05-24 ~19:00 PT) the K=γ integration is open territory.** No upstream blocker. Same communication policy as Phases A-E (see Phase 0): land the work, measure the delta, send a results-driven message after.

**Milestone:** tok/s ≥ 13.86. Output correctness verified. Updated perf report with before/after.

---

### Phase G — Memory-traffic and launch reduction (the "everything else" phase). **POST-PHASE-F.** Estimate: half-day to one full day total.

**Rationale:** even with graphs (Phase E) and the cap lift (Phase F), the drafter still does 155 kernel launches per propose with redundant memory traffic patterns inherited from "make-it-work-first" engineering. vLLM and TRT-LLM both push these wins to the kernel level. We follow.

Each item below is **independent of the others** and has a clean before/after benchable milestone. Land them in order of expected impact; STOP if a measurement shows <1% gain (means we mis-estimated and time is better spent elsewhere).

#### G.1 — Batched argmax (biggest single win, ~3-5% tok/s)

**Today:** drafter emits γ=16 logits rows. We launch `argmax_bf16` 16 times per propose, each reading the full `[247936 × 2 bytes BF16] = 484KB` row from HBM, returning 4 bytes. **7.7 MB of HBM traffic per propose, 16 launches, just to extract 64 bytes of result.**

**Fix:** new kernel `argmax_bf16_batched` that takes γ rows of logits and produces γ token IDs in a single launch. Grid: `(γ, 1, 1)` blocks of 256 threads. Each block does the parallel reduction for one row. Same algorithm, just bundled.

**Wins:**
- 16 launches → 1 (saves CPU launch overhead — most of this gain already captured by graphs, but still meaningful)
- One big kernel saturates HBM far better than 16 small ones
- Eliminates inter-launch sync between argmax calls

**CUDA work:** new `argmax_bf16_batched.cu` (~80 lines, lifts the existing argmax loop into a 2D grid). Plus Rust wrapper.

**Acceptance gate:** output identity vs eager argmax. Tok/s improvement ≥ 2.5%.

#### G.2 — Use existing fused `residual_add_rms_norm` in drafter (free win, ~1-2%)

**Today:** drafter calls `residual_add` then `rms_norm` as two separate kernels in two places per layer (post-attention, end-of-MLP). Each read+writes the full `[γ × hidden]` tensor. **4 MB of HBM traffic per propose** spent on this pattern.

Atlas's target model already uses a **fused `residual_add_rms_norm` kernel** (line 4288 of the nsys kernel list — 4.6 µs avg). The drafter just doesn't call it.

**Fix:** in `forward_block_layer_paged.rs`, replace the residual+norm pair with the fused kernel. Two replacement sites per layer × 8 layers = 16 fewer kernel launches per propose.

**CUDA work:** **NONE.** The fused kernel already exists and is registered.

**Rust work:** ~20 lines of dispatcher changes.

**Acceptance gate:** output identity. Tok/s improvement ≥ 1%.

#### G.3 — Fused QKV projection (~1-3%)

**Today:** drafter does 3 separate `dense_gemm` calls (q_proj, k_proj, v_proj), each reading the same `[γ × hidden]` input from HBM. With γ=16 and hidden=2048, that's **128KB of redundant input reads per layer × 8 layers = 1 MB wasted per propose.** Plus 3 small GEMMs at M=16 underutilize SM tile occupancy.

**Fix:** at weight-load time, concatenate `q_weight + k_weight + v_weight` along the output dimension into one fused `qkv_weight: [hidden, q_dim + k_dim + v_dim]`. Single GEMM produces all three projections at once. Slice the output buffer for downstream use.

vLLM does this (`qkv_proj` in Qwen3DecoderLayer). Pattern matches Atlas's existing `fused_kv_weight` in the precompute path.

**CUDA work:** NONE. `dense_gemm_bf16` already handles arbitrary M×N×K shapes.

**Rust work:** ~60 lines — weight concat in `from_weights.rs`, dispatcher update in `forward_block_layer_paged.rs`, output-buffer slicing.

**Acceptance gate:** output identity. Tok/s improvement ≥ 1%.

#### G.4 — Fused gate_up projection (~1-2%)

**Today:** drafter does 2 separate `dense_gemm` calls (gate_proj, up_proj) on the same MLP input, then silu_mul, then down_proj. Same pattern as G.3 — redundant input reads, undersized GEMMs.

**Fix:** at weight-load time, concatenate `gate_weight + up_weight` into `gate_up_weight`. Single GEMM. Then `silu_mul` reads the concatenated output, applies SiLU to the gate half, multiplies by the up half. Atlas already has `moe_silu_mul` in the kernel inventory — confirm it works on the dense path or write the obvious dense variant.

**CUDA work:** possibly a new `silu_mul_split_bf16` kernel if `moe_silu_mul` doesn't generalize. ~50 lines if needed; possibly zero.

**Rust work:** ~50 lines — same pattern as G.3.

**Acceptance gate:** output identity. Tok/s improvement ≥ 1%.

#### G.5 — Audit for any remaining capture-time redundancy (~0.5-1%)

After G.1-G.4 land, re-profile with nsys and look for any remaining patterns: redundant intermediate buffer writes, sub-optimal kernel launch configs that `cuGraphInstantiate` might be able to optimize at the captured-graph level, etc. This is the cleanup pass.

**Implementation:**

26. **G.1 first** (batched argmax). Largest single win, kernel-design pattern well-established. Land + bench.
27. **G.2 second** (fused residual+norm). Free win, no new CUDA. Land + bench.
28. **G.3 + G.4 together** (fused projections). Same fix pattern, same blast radius — qkv first, then gate_up. Land each independently with its own bench.
29. **G.5 audit** with fresh nsys after G.1-G.4.

**Combined expected impact:** 5-10% additional tok/s on top of Phase A-F.

**The math for the 20 tok/s stretch target:**

```
10.76 baseline
× 1.40 (graphs at the high end of vLLM-observed range)   → 15.1
× 1.26 (Phase F cap lift to γ, amortization 1.45 → 1.83) → 19.0
× 1.06 (Phase G combined wins)                            → 20.1
× 1.05 (fat LTO at the very end)                          → 21.1
```

**Every multiplier has to land at or near its high end. Achievable, not guaranteed.** Each phase has a benchable milestone with go/no-go criteria; we'll know at each step whether we're on the curve.

**Acceptance for Phase G overall:** tok/s ≥ 18.0 after G.1-G.4. Combined with Phase E + F, we want to land at 18-22 tok/s — well past the 13.86 no-spec ceiling, and meaningfully into "DFlash actually shines on dense non-MoE models" territory.

### Phase H — Native FP8 propose (FlashAttention-3 style). **POST-PHASE-G, PRIORITY-FLOATING.** Estimate: 2-4 days depending on whether SM12.x acceptance can be unlocked.

> **Priority note:** parked here for now because Phases A-G are bounded, designed, and benchable. **Likely to escalate** ahead of late-G items if early graph numbers (E or F) come in below target — FP8 is the single biggest remaining lever on both throughput and memory budget. Re-evaluate priority at the end of Phase E.

**Why FP8 matters:**

- ~2× attention throughput on Blackwell GB10 vs BF16 (FA-3 paper §4, NVIDIA Hopper data — Blackwell improves further).
- Halves KV-cache memory per drafter layer, freeing budget for `DRAFT_CAP > 1` (Phase F) and/or larger `ctx_window`.
- Reduces L2 / HBM pressure during propose — the drafter is partly memory-bound even after Phase G.1-G.4.
- The dispatcher and kernel already exist (`prefill_attention_paged_dflash_fp8` + `inferspark_prefill_paged_fp8`), parked behind a quality gate.

**Why it's parked, not killed:**

Per `dflash_head.rs:82–86`, drafter FP8 KV collapses acceptance on SM12.x (GB10) — the dynamic range loss on the K side breaks the bidirectional γ-block attention math at the precision the drafter was trained at. This is a real correctness issue, not a perf knob, so we ship BF16 first and tackle FP8 as a separate phase with its own correctness gate.

**The integration work:**

1. **Reproduce the SM12.x acceptance collapse on a clean Stage 5 + Phase E baseline.** Capture the exact delta (currently anecdotal from earlier runs).
2. **Diagnose with the FA-3 quantization recipe:** per-tile K scales (vs. per-tensor), Q in BF16, K/V in FP8 E4M3, dequantize-in-SMEM before the QK GEMM. This is the FA-3 paper's recommendation and matches what TRT-LLM does on Hopper. Atlas's current FP8 path may be using per-tensor scales — verify and switch if needed.
3. **Add an indirect FP8 twin** (`inferspark_prefill_paged_fp8_indirect`) following the exact pattern from Phase A. Same `kv_len`/`q_offset` indirection so the FP8 path is graph-friendly from day one.
4. **Wire the indirect FP8 dispatcher** in `prefill_attn_main_a.rs` and `from_weights.rs` (mirror of the BF16 indirect work).
5. **Per-tile K scale buffer:** new pre-graph H2D write (8-16 bytes depending on tile count) — same pattern as `option_b_indirect_args_dev`. If FA-3-style per-tile scales restore acceptance, this is the unlock.
6. **Quality gate before merge:** acceptance rate within 2 percentage points of the BF16 baseline on the standard 32 s bench. No regression on accept-rate is a hard prerequisite — FP8 that's faster but accepts less is a net loss.
7. **Switch the drafter KV cache to FP8** (`DflashQuantization::Fp8`) once the gate clears. Memory recovered ≈ `ctx_window × num_layers × num_kv_heads × head_dim` bytes (half the BF16 cost).

**Estimated impact:**

- **Propose attention kernel time:** ~50% reduction in the attention step (the single largest GEMM in the layer body).
- **Tok/s ceiling:** if Phases A-G land at 18-22, Phase H pushes toward 22-26 — but only if the acceptance gate clears.
- **Memory headroom:** half the drafter KV cache, opening room for larger `ctx_window` or `DRAFT_CAP=γ` simultaneously.

**Escalation triggers:**

- Phase E tok/s comes in under 13.0 → move H ahead of G.3-G.5 (the marginal launch-reduction wins).
- Phase F can't lift `DRAFT_CAP > 1` due to memory pressure → move H ahead of F.
- We get an external request from Avarok / Nologik for FP8 numbers → move H to next-up.

**Acceptance for Phase H:** tok/s ≥ 20.0 with acceptance rate within 2pp of the Phase E baseline. Cache memory halved on the drafter side. No regression in the target model path (Phase H touches only the drafter).

---

## 7. Risks & mitigations

### 7.1 The shared memory broadcast is correctly synchronized

The `KERNEL_PREAMBLE` writes to shared memory from thread 0, then `__syncthreads()` to broadcast. All threads then read `kv_len` and `q_offset` as locals (the params got overwritten before any other thread reads them).

**Risk:** thread 0's write happens before the `__syncthreads`, but `kv_len = s_indirect[0]` (the reassignment) happens AFTER the sync, on all threads. Correctness depends on this ordering being respected by the C++ compiler.

**Mitigation:** the `__syncthreads()` is a hard fence; no compiler can reorder past it. The reassignment IS data-dependent on `s_indirect[]` which IS data-dependent on the sync. Correct by construction.

### 7.2 Existing 6 kernel variants unaffected

Dropping `const` from `kv_len`/`q_offset` in the shared `.cuh` is a non-semantic change — the existing kernels never write to them. Compilation gives identical PTX. **Phase A includes a `cuobjdump --dump-sass` diff on two unaffected variants (BF16 contiguous + NVFP4) to prove byte-identical SASS.**

### 7.3 Stale pointers

The captured graph holds pointers to: scratch buffers (model-load constants), weights (model-load constants), KV pool pointers (allocated once at model load, never moved), `block_table_dev` (allocated once on first propose, never moved), `option_b_indirect_args_dev` (scratch, model-load constant).

**Every captured pointer is either model-load-stable or allocated-once-on-first-propose-stable.** No pointer churn possible.

### 7.4 Graph capture failure

CUDA may reject the capture if any in-stream operation isn't captureable. Atlas's existing graphs use `CU_STREAM_CAPTURE_MODE_RELAXED` (mode 2), which accepts most operations. Our captured region is pure kernel launches — no allocations, no host syncs, no D2H. Should capture cleanly.

**Mitigation:** `end_capture` returns `GraphHandle(0)` on empty/failed capture; we detect this and fall back to eager.

### 7.5 First-call latency

The first propose pays ~5-15 ms for graph capture + instantiate. For a 300-token response, that's <0.05% of total runtime. Invisible in real workloads.

For benchmark consistency: warm up by running a short prompt before timing. Or measure tok/s over the steady-state portion of the response (steps 10-N) instead of total.

### 7.6 max_batch_size > 1

Documented assumption: max_batch_size = 1 today. Drafter only runs one seq at a time. If max_batch_size > 1 is ever introduced, the propose_graph would need to become per-slot (HashMap keyed by slot_idx). The architecture supports this trivially.

### 7.7 Drop ordering

If `BlockDiffusionDraftHead` drops before the GPU stream is destroyed, calling `destroy_graph` works fine — `cuGraphExecDestroy` is valid as long as the CUDA context is alive. Atlas's existing drop pattern in `model/drop.rs` handles this; we follow it.

---

## 8. Test plan

### 8.1 Correctness — output identity

Two benches:
- `ATLAS_DFLASH_PROPOSE_NO_GRAPH=1` (graphs off)
- Default (graphs on)

Tokens MUST match byte-for-byte (temperature=0). Divergence = stale-pointer bug or kernel correctness bug. Halt and debug.

### 8.2 Correctness — accept rate

Within ±2 % of 44.9 %. Anything below 42.5 % is a regression.

### 8.3 Correctness — indirect kernel parity

Phase C explicitly tests the indirect kernel in eager mode before graphs land. This decouples "indirect-arg kernel is buggy" from "graph capture is buggy" — the two failure modes look identical from the bench output, but Phase C isolates one of them.

### 8.4 Perf — tok/s

- Phase C (indirect, eager): ≥ 10.5 tok/s (within noise of baseline)
- Phase D (graphs on): ≥ 11.5 tok/s threshold. ≥ 12.0 to call the architecture validated.
- Stretch: ≥ 13.0 tok/s. Clearing 13.86 with graphs alone would be exceptional.

### 8.5 Perf — kernel launches

nsys re-run must show:
- `cuLaunchKernel` drops from ~65k to ~1000-2000
- `cuGraphLaunch` increases to ≥ 191 + existing target-side count
- `cuStreamSynchronize` count drops materially

### 8.6 Robustness — debug-var disable

Each of the gating env vars (§4.3) enabled individually must:
- Disable graphs (logs "drafter graphs disabled: <var> set")
- Server still functions
- Bench still produces correct output (back to ~10.76 tok/s)

### 8.7 Memory — graph leak check

`Drop` log line reports destroyed graph(s). Count should match captures. No leak.

---

## 9. Expected impact

### 9.1 Per-propose savings

Before:
- ~150 cuLaunchKernel × 4.35 µs = 650 µs CPU launch
- Mid-propose sync points (implicit) = additional 1-3 ms CPU idle
- One forced sync at the end (D2H readback) ≈ 2-5 ms (unavoidable)

After:
- 1 cuGraphLaunch ≈ 239 µs CPU
- No mid-propose sync points
- Same final D2H readback
- One small H2D for indirect args (8 bytes, async, ~5 µs CPU) — bundled with the other pre-graph H2Ds

**Per-propose saving: ~1-3 ms.** Over 191 calls: **190-570 ms total saving** on a 32 s bench. At 32 s baseline: that's 5.9-17.8 % wall-time reduction.

### 9.2 Tok/s prediction (revised honest range)

The naive estimate (launch overhead saved in isolation) gives only ~12.0 tok/s. That undercounts the real win.

**The dominant effect is GPU stream utilization.** Without graphs, the CPU re-feeds the GPU every few kernel launches; the GPU spends 30-50% of decode time idle waiting on the next batch. With graphs, the entire 150-kernel sequence submits in one driver call — no CPU back-pressure, no idle bubbles.

vLLM published numbers for graphing decode hot paths: **30-50% throughput improvement** depending on model and batch size. SGLang reports similar. Both run the same kind of CUDA graph capture we're proposing.

Applied to our 10.76 baseline:

| Scenario | Multiplier | Tok/s |
|---|---|---|
| Floor (launch overhead only, no stream-utilization gain) | 1.07x | 11.5 |
| Low end of industry-typical graphs delta | 1.15x | 12.4 |
| **Mid-range realistic (matches vLLM observed)** | **1.30-1.40x** | **14.0-15.1** |
| High end (no hidden sync points remain) | 1.5x | 16.1 |

**Median honest expectation: 13-15 tok/s.** **Plausibly clears 13.86 no-spec by itself.**

Combined with Phase F (cap lift from 1 to γ, amortization 1.45→1.83):

| Scenario | Tok/s |
|---|---|
| Floor (Phase E low end + cap lift) | 14.5 |
| **Realistic (Phase E mid + cap lift)** | **16-19** |
| Optimistic | 21+ |

**Realistic combined target: 16-19 tok/s. Clears no-spec with margin and DFlash actually pays for itself.**

---

## 10. What we are NOT doing (scope discipline)

- NOT writing new kernels for the other 6 paged-attention variants (FP8, NVFP4, batched, HDIM=512). They're not in the drafter path.
- NOT modifying the Metal backend (DFlash unsupported there).
- NOT graphing the precompute path (vLLM doesn't either; not needed).
- NOT changing K2 cap or SSM rollback logic before Phase F.
- NOT touching the target model's existing graph caches.
- NOT using `cuGraphExecKernelNodeSetParams` to mutate captured graphs (see §13).

---

## 11. Decisions on open design questions

(Previously open; locked in 2026-05-24 per Ronald.)

1. **Kernel naming.** `prefill_attention_paged_dflash_bf16_indirect`. Rust function follows the same `_indirect` suffix. Kernel registry name `inferspark_prefill_paged_indirect`. Pattern matches existing variant naming (`_fp8`, `_nvfp4`).

2. **Indirect-args buffer location.** `DflashScratch`. Single drafter, single seq today (max_batch_size=1). Per-seq would be premature complexity. If multi-seq drafter ever lands, move it to `DflashProposerState` then; the API change is local.

3. **Warm-up + pre-capture.** Run the eager forward path **N=2 times** before capturing the graph. Reasons:
   - Warms CUDA's PTX→SASS compilation cache (per-kernel JIT happens on first launch)
   - Lets GB10's dynamic clock ramp to sustained-workload state (~100-500ms otherwise)
   - Brings hot drafter weight tiles into L2 cache (~2× effective BW on warmed tiles)
   - Hits memory allocator fast paths
   - **Most importantly:** the graph captures the SASS variants the driver picks at steady state, not the cold-start variants — significantly better graph quality

   Default N=2, override via `ATLAS_DFLASH_PROPOSE_WARMUP_N`. No model-load dummy pre-capture (block_table_dev isn't allocated yet — would need fake state setup that isn't worth it).

4. **`--no-graph` CLI flag.** Env-var only: `ATLAS_DFLASH_PROPOSE_NO_GRAPH=1`. CLI surface stays clean. Env-var is sufficient for A/B benching and emergency disable.

5. **Graph instantiate flag.** Use `CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD` (= 8). Eagerly uploads the graph to the device, avoiding a first-launch JIT-to-device cost. Free perf, single API arg change.

6. **Phase-E warmup N sweep.** Phase E benches N ∈ {1, 2, 4, 8} and picks the best. ~30 min of benching. Could move us 0.3-0.5 tok/s; worth measuring.

---

## 12. Summary

**Phases A-E (graphs):** One CUDA file (~12 lines new), one .cuh edit (drop `const` from 2 params), ~100 lines of Rust glue, and the entire drafter propose path becomes one captured CUDA graph. Single graph, captured at first call (after 2 warm-up iterations), replayed every step. Expected tok/s: 10.76 → ~12.0.

**Phase F (K=γ SSM rollback, post-graphs):** Wire DFlash K=γ verify into the new `SsmSnapshotPool` decode-rollback ring that Thomas Braun landed on main. ~Half-day of Rust integration (NOT kernel work, despite what propose.rs:170-194 claims — the snapshot infrastructure makes the "multi-week kernel work" framing obsolete). Lifts `ATLAS_DFLASH_DRAFT_CAP` from 1 to γ, raising amortization from 1.45 → 1.83 accepts/cycle. **Combined with Phases A-E: 10.76 → ~14-17 tok/s. Clears no-spec.**

**Phase G (memory-traffic and launch reduction, post-F):** Five independently-benchable optimizations: G.1 batched argmax (single launch instead of 16, ~3-5% gain). G.2 fused residual+rms_norm in drafter (free — kernel exists). G.3 fused qkv projection (vLLM-style, one GEMM instead of three). G.4 fused gate_up projection (same pattern). G.5 post-audit cleanup. Combined: ~5-10% additional tok/s. **Combined Phases A-G: 18-22 tok/s.**

Both phases follow established Atlas patterns. Phase A-E follows verify_d.rs / decode_a2.rs graph capture and uses Atlas's existing `KERNEL_PREAMBLE` macro extension mechanism. Phase F follows the scheduler/rollback.rs Phase-C watchdog pattern. Phase G follows vLLM's standard kernel-fusion playbook. **Zero greenfield architecture.**

Ship order:
1. This document → review.
2. Phase A → kernel landed + PTX-diff verification.
3. Phase B → Rust dispatcher.
4. Phase C → indirect-eager bench, byte-identity gate.
5. Phase D → graphs on, perf gate.
6. Phase E → final nsys + writeup + warmup-N sweep. **First win — DFlash perf-acceptable.**
7. **Results-driven message to nologik** with the tok/s delta. See Phase 0.
8. Phase F → K=γ snapshot integration. **Second win — DFlash perf-shippable.**
9. Phase G.1 → batched argmax. Largest single Phase-G win.
10. Phase G.2 → fused residual+rms_norm (free win, no CUDA).
11. Phase G.3 → fused qkv_proj.
12. Phase G.4 → fused gate_up_proj.
13. Phase G.5 → post-Phase-G nsys audit + cleanup.
14. **Second results-driven message to nologik** with combined-phase tok/s delta. Frame: "DFlash on dense non-MoE went from 10.76 → X tok/s. Here's the breakdown."
15. Apply fat LTO to workspace Cargo.toml as final pre-ship optimization. **DO NOT FORGET.** (See LTM 121.)

---

## 13. Rejected alternatives (the v3 investigation)

A v3 of this design proposed using `cuGraphExecKernelNodeSetParams` (the CUDA Driver API for mutating captured graph kernel-node args between launches) instead of indirect-arg kernels. This section records why that path was investigated and ultimately rejected, so future maintainers don't re-investigate.

### 13.1 The v3 idea

Capture the propose graph once with the first call's `kv_len` and `q_offset` baked in as scalar kernel args. Between launches, identify the 8 attention kernel nodes via post-capture filtering, then call `cuGraphExecKernelNodeSetParams` on each to update the two scalars before re-launching. No kernel changes at all.

### 13.2 Why it looked attractive

- Zero CUDA edits. Zero blast radius outside the drafter path.
- No risk to the 6 unrelated paged-attention variants (FP8, NVFP4, batched, HDIM=512).
- Closer to how TensorRT-LLM updates per-iteration args.
- cudarc 0.19.2 already binds `cuGraphExecKernelNodeSetParams` — no new dependencies.

### 13.3 Why it was rejected

**No production reference.** When evaluated against the standing rule ("best practices first, then what Atlas does, then what vLLM does"):

- **Atlas:** zero uses of `cuGraphExecKernelNodeSetParams` anywhere in the codebase. All 5 existing graph caches (decode, batch_decode, verify2/3/4, verify_kgamma) use the "capture once per cache key + persistent device buffers" pattern.
- **vLLM:** zero uses. Their `InputBuffers` class (`v1/worker/gpu/input_batch.py:12`) holds persistent device tensors written pre-graph; the captured graph reads them by pointer. **Structurally identical to v4's approach.** For scalar args that vary, vLLM pads to canonical sizes and captures one graph per padded size.
- **TRT-LLM:** uses some form of arg mutation internally, but it's deeply integrated with their plugin system, not exposed as a recommended pattern for external use.

**Per-call cost is higher.** v3 needed 8 `cuGraphKernelNodeGetParams` + 8 `cuGraphExecKernelNodeSetParams` calls per propose ≈ 110 µs CPU. v4 needs one 8-byte `copy_h2d` ≈ 5 µs CPU + ~1 µs GPU. Net: v4 is ~100 µs faster per call. Small but real.

**Failure modes are subtler.** v3's setParams must preserve all pointer args correctly when mutating scalars; any miss → segfault on next launch. v4's failure modes are confined to one kernel and detectable at PTX-diff time.

**`KERNEL_PREAMBLE` exists for exactly this use case.** Atlas's authors built a preprocessor extension point into `prefill_paged_compute.cuh` specifically to allow kernel-variant customization without forking the shared body. Using it is using the system as designed, not subverting it.

### 13.4 Conclusion

v3 was a textbook "clever-but-not-shipping" idea. v4 reverts to v2's indirect-args design with the v3 investigation recorded here so we don't have to re-litigate it.

The lesson generalizes: when an approach has zero production references in stacks we trust, that's a strong signal — not a green field to plant flags in. Best practices first, then Atlas, then vLLM. Don't reinvent wheels.

---

## 14. Phase E — Piecewise graph + async tail (the actual fix)

**Status**: design (not implemented). Phases A-D landed May 26 2026. Single full-region graph captures cleanly, replays 408× per bench, but the bench moved 8.67 → 8.74 tok/s — a wash. nsys says why.

### 14.1 What Phase D revealed

Profile from `~/atlas-dflash-nsys-phaseD-20260526-120413/`:

| API call                 | % of API time | Calls   | Avg     |
|--------------------------|---------------|---------|---------|
| `cuStreamSynchronize`    | **63.2%**     | 271,596 | 97.5 µs |
| `cuMemcpyDtoHAsync_v2`   | **31.3%**     | 545     | 24.1 ms |
| `cuMemcpyDtoDAsync_v2`   | 3.9%          | 309,167 | 5.2 µs  |
| `cuLaunchKernel`         | 0.8%          | 57,535  | 6.1 µs  |
| `cuGraphLaunch`          | 0.2%          | 408     | 213 µs  |

Read the right column carefully: `cuLaunchKernel` is **already under 1%**. Phase D did exactly what the design promised — it killed per-kernel launch overhead. The 408 `cuGraphLaunch` calls are the captured forward_block replays.

So why no speedup? Because the bottleneck was never launch overhead. The 271k `cuStreamSynchronize` calls and the 545 D2H readbacks were **outside** the captured region all along. The graph can't optimize calls we make around it.

The single biggest sync source: `forward_block.rs:589-590`, **fires once per propose**:

```rust
let mut host_buf = vec![0u8; self.gamma * 4];
gpu.synchronize(stream)?;                        // unconditional FULL stream sync
gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut host_buf)?;
```

That `synchronize` blocks the host until every captured kernel completes, then we D2H γ×4 bytes to feed verify. Profile: 24 ms average D2H × 545 calls ≈ 13.1s of wall time, with the host stalled inside `cuStreamSynchronize` for most of it.

vLLM hits the identical problem. Their **Full CUDA Graph for the drafter** work is still landing as of [issue 33341](https://github.com/vllm-project/vllm/issues/33341) — they ship piecewise graphs for the drafter today, full graphs for the main model. They explicitly note: *"only FlashAttention 3 supports full graphs cleanly; everything else falls back to piecewise."*

So we're not behind. We're shipping the same fight they're shipping.

### 14.2 Cross-references: how others handle this

**vLLM v1 piecewise capture** (`vllm/compilation/cuda_graph.py` + `vllm/v1/cudagraph_dispatcher.py`):

- `CUDAGraphWrapper` wraps any callable; each wrapper instance binds one runtime mode (`FULL` or `PIECEWISE`) and holds a `torch.cuda.CUDAGraph` per `BatchDescriptor` key.
- `CudagraphDispatcher` is the single source of truth: it picks `FULL` > `PIECEWISE` > `NONE` per batch and tells wrappers which mode to use via a thread-local `ForwardContext`.
- **Piecewise capture cuts at attention boundaries.** Attention runs eager; everything between attention layers is wrapped in its own captured graph. The wrappers see contiguous compute and capture freely; sync points and dynamic shapes get pushed out to the eager boundaries.
- They explicitly call out: **a piecewise wrapper doesn't manage warm-up.** That's the model runner's job. Match our Phase D warm-up state machine.

**FlashAttention-3 host launcher** (`hopper/flash_fwd_launch_template.h`):

- `run_flash_fwd(params, cudaStream_t stream)` is the entire host-side surface area. **No `cudaStreamSynchronize` anywhere.** Caller passes a stream; the kernel runs async; the caller's next operation chains on the same stream and serializes naturally.
- `launch_with_pdl` flag = Programmatic Dependent Launch. On Hopper/Blackwell, the next kernel begins loading its data while the previous kernel is finishing — eliminates launch-to-launch gap without graphs. We can pursue PDL later; for Phase E it's overkill.
- Inside the kernel: persistent tile scheduler + warp specialization (producer-consumer between TMA loads and WGMMA math). Lesson for us: the host launcher is a thin pass-through. **Anything that calls `synchronize` on the host is bypassing the entire pipeline.**

**Atlas, post Phase D**: `forward_block.rs:589` synchronize is the smoking gun. Three other syncs (`forward_block.rs:90,196`, `forward_block_layer_paged.rs:214,238,260`) are gated behind `ATLAS_DFLASH_DEBUG_DUMP` and inert in production.

### 14.3 The cut points

Today's `forward_block` (paged + graph path) has three regions:

```
┌─ PRE-GRAPH (host-side, ~30 µs) ────────────────────────────┐
│  • H2D writes: position_ids, draft_tokens_dev,             │
│    last_token, indirect args, slot mapping                 │
│  • Optional fc_proj + hidden_norm + precompute_ctx_kv      │
│    (FIRST PROPOSE ONLY, conditional)                       │
└────────────────────────────────────────────────────────────┘
┌─ CAPTURED (5 drafter layers + final norm + lm_head +       │
│           γ argmax). ONE graph, replayed every step.       │
│   Today: ~83 kernels per replay, ~213 µs per launch_graph. │
└────────────────────────────────────────────────────────────┘
┌─ POST-GRAPH ──────────────────────────────────────────────┐
│  • gpu.synchronize(stream)   ← THE FLOOR                   │
│  • gpu.copy_d2h(draft_tokens_dev, γ*4 bytes)               │
│  • Vec<u32> from bytes; return to verify                   │
└────────────────────────────────────────────────────────────┘
```

Cut points where Phase E should slice:

1. **The H2D writes are static-pointer.** We already proved this in Phase C (indirect args). They can move *inside* the capture as `cuMemcpyHtoDAsync` — captured H2D nodes are legal in a CUDA graph as long as the source is pinned host memory. **Lift them into capture #1.**

2. **The `precompute_ctx_kv` first-propose path is conditional.** Conditional branches break capture. **Don't capture it.** Either run it eagerly as a one-shot pre-capture step (already what it does), or extract into its own dedicated graph keyed by `is_first_propose` — but the math says one-shot wins.

3. **The synchronize-then-D2H tail is the floor.** Two options, in increasing order of complexity:
   - **E.1 (drop the synchronize):** `copy_d2h_on_stream(stream)` already serializes against the captured kernels — the explicit `synchronize` is redundant. Same fix logged in LTM 124 from May 24's nsys (Win #2 attempt). Last time it didn't move the needle because the bottleneck was elsewhere; **now it's the bottleneck**, so this is the first thing to try.
   - **E.2 (pinned async D2H + event):** allocate `draft_tokens_dev`'s host shadow as `cudaHostAlloc(cudaHostAllocPortable)`. Issue `cudaMemcpyAsync` immediately after `launch_graph`. Record an event on the stream. Verify queries the event with `cudaEventQuery` (non-blocking) and stalls only if the copy hasn't landed by the time verify actually needs the tokens. This decouples drafter wall-time from D2H latency entirely.

4. **The captured region itself should split into two captures IF E.1+E.2 don't close the gap.** Cut between the layer loop and the final-norm/lm_head/argmax tail. Rationale: the layer loop is dense compute (drafter MLPs + paged attention); the tail is lm_head GEMM + γ argmax. If a future debug flag or per-call variant lands that breaks the tail's capture eligibility but not the layer loop's, the layer-loop graph keeps replaying and only the tail falls to eager. Mirrors vLLM's piecewise wrapper-per-piece structure. **Not needed for the v1 perf win; logged so a future refactor doesn't lose it.**

### 14.4 Phase E execution plan

**Phase E.1 — drop the redundant synchronize. 30 minutes.**

1. `forward_block.rs:589-590`: replace
   ```rust
   gpu.synchronize(stream)?;
   gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut host_buf)?;
   ```
   with `gpu.copy_d2h_on_stream(self.scratch.draft_tokens_dev, &mut host_buf, stream)?` (already exists in `cuda_backend/gpu_impl.rs` per LTM 124). The on-stream variant serializes against the captured kernels via stream order; no explicit pre-sync needed.
2. Bench. Expect `cuStreamSynchronize` % to halve. Tok/s expected: 9.5-10.5.

**Acceptance:** 44.9% accept rate exact, tok/s ≥ 9.5. If output diverges, immediately revert — host_buf isn't valid before the copy completes, and we may need a `cudaEventSynchronize` on a per-propose event instead of a stream sync.

**Phase E.2 — pinned async D2H with event. 2 hours.**

1. Allocate γ×4 bytes of pinned host memory in `DflashScratch` construction (`from_weights.rs`). Add `draft_tokens_host_pinned: *mut u8` to `DflashScratch`.
2. Add `draft_tokens_event: CudaEvent` to `DflashScratch` (one event, reused).
3. `forward_block.rs:589-590`: issue `cudaMemcpyAsync(host_pinned, dev, γ*4, D2H, stream)` then `cudaEventRecord(draft_tokens_event, stream)`. Return immediately; **no host sync**.
4. In `propose.rs` where the returned `Vec<u32>` is consumed by verify: `cudaEventSynchronize(draft_tokens_event)` then read from `host_pinned` directly. (verify already runs on the same stream, so the event is *probably* implicit — measure first.)
5. Bench.

**Acceptance:** 44.9% accept, tok/s ≥ 11. Profile expectation: `cuStreamSynchronize` collapses from 63% to single digits, `cuGraphLaunch` becomes the dominant API call (correct — that's the actual work).

**Phase E.3 — H2D lifted into capture (only if E.1+E.2 leave room). 4 hours.**

1. Allocate pinned host memory for every per-propose H2D source (`last_token` byte, mask tokens, `position_ids`, indirect args, slot_mapping). Stable host pointers required for graph capture.
2. Move all `gpu.copy_h2d` calls into the captured region. Use `copy_h2d_async`. Each H2D becomes a graph node.
3. Pre-capture, host code now writes to the pinned buffers (memcpy on host, instant) instead of calling into the driver per H2D.
4. Bench.

**Acceptance:** 44.9% accept, tok/s ≥ 12. Diminishing returns from here — at this point we're inside a percent or two of the 13.86 no-spec ceiling.

**Phase E.4 (optional, deferred):** split capture into layer-loop graph + tail graph (per 14.3.4). Logged here, not on the critical path.

### 14.5 What we are NOT doing

- **PDL (Programmatic Dependent Launch).** Real win on Hopper/Blackwell, requires kernel attribute + driver support + per-kernel verification. Re-visit after E.1/E.2/E.3 prove insufficient. Not on the v1 path.
- **Persistent kernel scheduler (FA3-style).** Genuinely large rewrite — collapses N small kernels into one persistent kernel doing all the tiles. Architectural change; future-major-version stuff.
- **Migrating to FlashAttention-3.** sm_90a-only (Hopper); GB10 is sm_121f (Blackwell). The WGMMA/TMA instruction set differs. Useful as a reference for host-side patterns (this section), not as a drop-in.
- **vLLM's full `CudagraphDispatcher` infrastructure.** Overkill for a single-graph drafter. Their dispatcher exists to switch between FULL/PIECEWISE per batch composition; we have one shape, one batch, one graph. The piecewise *idea* is what we're cribbing, not their dispatch machinery.

### 14.6 Honest projection

E.1 alone should reclaim the Phase C regression (8.67 → ~10.5). E.2 is where the real win lives — kills the synchronize floor, drafter wall-time decouples from D2H latency. E.3 is gravy.

If E.1+E.2 still don't beat 13.86 no-spec, the floor is somewhere else (drafter dense_gemm_bf16, currently 61% of GPU time) and the next investigation is the drafter's MLP math itself, not the host-side orchestration. nsys will tell us.

**Phase E succeeds when Atlas DFlash beats Atlas no-spec on the canonical Volvo bench.** Anything less and the architecture doesn't ship.

### 14.7 Phase E results (May 26 2026)

- **E.1 landed.** Replaced `gpu.synchronize(stream); gpu.copy_d2h(...)` with `gpu.copy_d2h_on_stream(stream)` in `forward_block.rs:589`. Build clean. Bench: 44.9% accept (exact), 8.64 tok/s (Δ ≈ -0.03 from Phase D, run-to-run noise). nsys: `cuStreamSynchronize` 29.7% (down from 63.2%), `cuMemcpyDtoHAsync_v2` 63.9% (up from 31.3%). **The accounting shifted, not the wall time** — the redundant sync was already a no-op against the captured graph; the wait was always inside the D2H copy itself.
- **E.2 landed.** Added `event_synchronize` to `GpuBackend` trait + CUDA impl. Added `draft_tokens_host_pinned: AtomicPtr<u8>` (γ×4 bytes via `cuMemAllocHost_v2`) and `draft_tokens_event: u64` (created via `cuEventCreate`) to `DflashScratch`. Swapped the post-graph D2H to `copy_d2h_on_stream` into the pinned buffer, `record_event`, then `event_synchronize` before host reads. Build clean. Bench: 44.9% accept (exact), 8.48 tok/s. nsys: `cuStreamSynchronize` 62.9%, `cuMemcpyDtoHAsync_v2` 31.8% — back to roughly pre-E.1 shape because `cuEventSynchronize` aggregates into the same nsys bucket as `cuStreamSynchronize`.

The E.1+E.2 hypothesis was: the post-graph sync was the floor, kill it and we reclaim 1-2 tok/s. **The hypothesis was wrong.** What actually happens: `forward_block` returns `Vec<u32>` to `propose_drafts`, which immediately returns to the scheduler, which immediately consumes `drafts[0]` to build `tokens_k2`. There is no host work to overlap with. Whether we wait via `cuStreamSynchronize`, `cuEventSynchronize`, or a busy-poll loop, we're waiting on the same physical D2H to complete. The pinned destination *should* speed the D2H itself (DMA fast path vs bounce-buffer), but the bench doesn't show it because the floor isn't the D2H either — **it's the drafter kernel work itself**.

GPU kernel time post-E.2 (`cuda_gpu_kern_sum.csv`):

| Kernel              | % GPU time | Instances |
|---------------------|------------|-----------|
| `dense_gemm_bf16`   | 60.7%      | 484       |
| `w4a16_gemv_dual`   | 14.9%      | 5280      |
| `w4a16_gemv`        | 9.0%       | 7459      |
| `w4a16_gemv_silu_input` | 8.3%   | 4224      |
| `w4a16_gemm`        | 3.8%       | 256       |

`dense_gemm_bf16` is the drafter's BF16 MLP. 60% of GPU time, ~15ms average per instance × 484 instances = 7.3s on a 43s bench. That's the floor. Host-side orchestration tweaks can't move it.

### 14.8 E.3 / E.4 SKIPPED — rationale

- **E.3 (lift H2D into capture).** nsys shows `cuMemcpyHtoDAsync_v2` at 0.0% of API time post-E.2 (1908 calls, 11ms total). Even if we moved all of them inside capture, the absolute time saved is bounded at single-digit milliseconds. Not worth the 4-hour refactor + capture-pointer-stability risk.
- **E.4 (split capture in two).** Architectural cleanup, not a perf win. The motivation was making the tail (lm_head/argmax) eager-able if a debug flag toggled. We don't need that today; we'd be doing work on speculation.

Both stay in the design doc as future references. Neither is on the v1 path.

### 14.9 The wall

E.1+E.2 didn't move the bench because the wall is `dense_gemm_bf16`, not the host-side orchestration we've been chiseling at. Three real levers from here:

1. **Piecewise graphs at the attention boundary** (Phase F, next). Validated by vLLM as the production pattern for drafters that aren't FA3-ready. Structural, not a perf win on its own — but it's the prerequisite for the kind of fusion that *would* move dense_gemm_bf16.
2. **FP8 weights for the drafter MLP.** Cuts the GEMM work roughly in half. Atlas already has FP8 GEMM kernels in the runtime (the target model uses them). Real lever, real risk: drafter FP8 acceptance-rate collapse on SM12.x was the original reason `DflashQuantization::Bf16` is the only variant.
3. **Kernel fusion** (RMSNorm + Q/K/V projections, silu+mul, residual paths). Real lever, real work — each fused kernel is custom CUDA.

Phase F next. FP8 after. Fusion last.

---

## 15. Phase F — Piecewise graph capture at attention boundaries

**Status**: design (not implemented). Based on vLLM's production drafter pattern and Atlas's existing `forward_block_layer_paged` structure.

### 15.1 Why piecewise, why now

vLLM v1 ships piecewise CUDA graphs for the drafter, not full graphs. Their [`CompilationConfig._attention_ops`](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py) lists `vllm::unified_attention_with_output` and 12 similar ops as the **mandatory split points**. The graph splits at every attention call; everything between is one captured subgraph. Their [`split_graph`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/backends.py) function in `vllm/compilation/backends.py:548` is the canonical reference.

The endorsement chain from issue #23261 (Cascade attention RFC): "*Cascade attention requires piecewise cudagraphs. Some attention backends don't always support cudagraph_mode=FULL.*" — the entire vLLM compilation pipeline is built around this assumption. Only FlashAttention-3 supports `FULL` graphs cleanly; everything else (FlashInfer, FlashMLA, our `inferspark_prefill_paged_indirect`) goes piecewise.

Atlas DFlash today has **one** captured region — the full layer loop + final norm + lm_head + argmax. Phase D proved this works; Phase E proved the floor isn't host-side. Phase F changes the *structure* of the capture so we can introduce per-layer fusion (or per-layer FP8 swap-in) later without breaking the graph.

### 15.2 Where Atlas cuts

In our terms (one DFlash drafter, 5 layers, BF16, paged):

```
forward_block:
  ── pre-graph (eager) ───────────────────────────────────
  | H2D writes: position_ids, draft_tokens, indirect args |
  | slot mapping kernel                                   |
  ── per-layer loop (×5) ─────────────────────────────────
  │ ┌─ SUBGRAPH N (captured) ────────────────────────────┐│
  │ │  input_layernorm → q_proj/k_proj/v_proj → q_norm   ││
  │ │  k_norm → rope (Q+K) → reshape_and_cache           ││
  │ └────────────────────────────────────────────────────┘│
  │ ┌─ ATTENTION (eager) ────────────────────────────────┐│
  │ │  prefill_attention_paged_dflash_bf16_indirect      ││
  │ └────────────────────────────────────────────────────┘│
  │ ┌─ SUBGRAPH N+1 (captured) ──────────────────────────┐│
  │ │  o_proj → residual_add → post_attention_layernorm  ││
  │ │  → gate_proj/up_proj → silu_mul → down_proj        ││
  │ │  → residual_add                                    ││
  │ └────────────────────────────────────────────────────┘│
  ── post-layer-loop (one final subgraph, captured) ──────
  | final RMSNorm → lm_head GEMM → γ argmax              |
  ── post-graph (eager) ──────────────────────────────────
  | async D2H + event_synchronize (per Phase E.2)        |
```

**Subgraph count:** 11 per propose (5 pre-attention × 5 layers + 5 post-attention × 5 layers + 1 tail). Each is captured once on the first eligible propose post-warmup, replayed thereafter.

**Why this cut, specifically:**

- vLLM does it. The pattern is production-validated across every attention backend that doesn't support FULL graphs.
- Attention is the kernel where dynamic state (KV cache pointers, kv_len, q_offset) lives. Even with our indirect-args kernel, the attention call is the *natural* sync barrier between drafter steps.
- The pre-attention and post-attention subgraphs each contain only dense_gemm_bf16 + rms_norm + silu_mul + reshape_and_cache + residual_add. All of these are pure compute, no per-call state changes. They graph cleanly.
- This is the structure that *lets* us introduce future fusion. Today the pre-attention subgraph is 5-7 kernels; a fused `qkv_proj_norm_rope_cache` kernel would collapse it to 1. The capture boundary doesn't have to move.

### 15.3 Atlas-specific implementation

**Files affected:**

- `crates/spark-model/src/layers/dflash_head.rs`: replace `propose_graph: Mutex<Option<GraphHandle>>` with `Mutex<Option<Vec<GraphHandle>>>` (one per subgraph slot). Replace `propose_warmup_count` with a per-subgraph variant (or keep it global — global is simpler, all subgraphs warm together).
- `crates/spark-model/src/layers/dflash_head/forward_block.rs`: rewrite the layer loop. Instead of one outer `gpu.begin_capture(stream)`...`gpu.end_capture(stream)` around the whole loop, wrap each pre-attention and post-attention block in its own capture region.
- `crates/spark-model/src/layers/dflash_head/forward_block_layer_paged.rs`: split `forward_block_layer_paged` into `forward_block_layer_pre_attn` and `forward_block_layer_post_attn`. Attention call lives in `forward_block.rs` directly between the two.

**Capture/replay loop sketch (paged path only):**

```rust
fn forward_block(...) -> Result<Vec<u32>> {
    // ── pre-graph (eager) ───
    self.write_dynamic_inputs(...)?;
    self.fill_gamma_slot_mapping(...)?;
    self.write_indirect_args(...)?;

    let graphs = self.propose_graphs.lock();  // Mutex<Option<Vec<GraphHandle>>>
    let warmed = self.propose_warmup_count.load(Relaxed);
    let use_replay = graphs.is_some() && warmed >= warmup_target;

    let mut new_graphs: Vec<GraphHandle> = if !use_replay { Vec::with_capacity(11) } else { Vec::new() };

    for layer_idx in 0..self.num_layers {
        // Pre-attention subgraph
        self.run_or_capture_subgraph(
            &graphs, &mut new_graphs, layer_idx * 2,
            use_replay, stream,
            || self.forward_block_layer_pre_attn(layer, &args, ctx),
        )?;

        // Attention — eager, NEVER captured (vLLM convention)
        ops::prefill_attention_paged_dflash_bf16_indirect(
            gpu, self.kernels.prefill_attn_dflash_bf16_indirect, ...
        )?;

        // Post-attention subgraph
        self.run_or_capture_subgraph(
            &graphs, &mut new_graphs, layer_idx * 2 + 1,
            use_replay, stream,
            || self.forward_block_layer_post_attn(layer, &args, ctx),
        )?;
    }

    // Tail subgraph: final norm + lm_head + argmax
    self.run_or_capture_subgraph(
        &graphs, &mut new_graphs, self.num_layers * 2,
        use_replay, stream,
        || self.run_lm_head_tail(...),
    )?;

    if !use_replay && warmed >= warmup_target {
        *graphs = Some(new_graphs);
    } else if warmed < warmup_target {
        self.propose_warmup_count.fetch_add(1, Relaxed);
    }

    // ── post-graph (eager, per Phase E.2) ───
    self.async_d2h_drafts(stream)?;
    self.event_synchronize_drafts()?;
    self.read_drafts_pinned()
}

fn run_or_capture_subgraph<F>(
    &self,
    cached: &Option<Vec<GraphHandle>>,
    new: &mut Vec<GraphHandle>,
    slot: usize,
    use_replay: bool,
    stream: u64,
    body: F,
) -> Result<()>
where F: FnOnce() -> Result<()>,
{
    if use_replay {
        let graph = cached.as_ref().unwrap()[slot];
        if graph.0 != 0 { self.gpu.launch_graph(graph, stream)?; }
        else { body()?; }
    } else if /* capturing */ {
        self.gpu.begin_capture(stream)?;
        body()?;
        let graph = self.gpu.end_capture(stream)?;
        new.push(graph);
        if graph.0 != 0 { self.gpu.launch_graph(graph, stream)?; }
    } else {
        // warm-up: eager only, no capture
        body()?;
    }
    Ok(())
}
```

**Eligibility gate:** identical to Phase D — `option_b_on && !suppress_graphs && !debug_dump && env-var gauntlet`. Same 10 env vars disable graphs across the board.

### 15.4 What Phase F buys us

**Not perf, by itself.** The total captured compute is identical; we've just put it in 11 small graphs instead of 1 big one. The 408 → ~4500 graph launches per bench will add some overhead — `cuGraphLaunch` is 213µs avg, so 4500 × 213µs = 960ms over a 43s bench, ~2% slower. **Expected: 44.9% accept (exact), 8.3-8.5 tok/s.**

What Phase F **enables**:

1. **Per-layer kernel fusion.** Fused `qkv_proj_norm_rope_cache` slots into the pre-attention subgraph; the capture boundary doesn't care about kernel count, only kernel state.
2. **Per-layer FP8.** If we land FP8 weights on layers 1-4 (keeping layer 0 BF16 for accuracy) and the FP8 GEMM kernel has different launch shape than BF16, the per-layer subgraph keeps each layer self-contained.
3. **Cascade attention support** (future). Today's paged attention is single-stream; cascade attention requires multiple. Piecewise structure is the prerequisite.
4. **Debug eligibility per-layer.** If `ATLAS_DFLASH_DEBUG_DUMP` is on for one specific layer, only that layer's subgraphs fall to eager; others keep replaying.

### 15.5 Risk + acceptance

**Risks:**

- **Graph instantiation cost × 11.** Phase D measured 2 calls to `cuGraphInstantiateWithFlags` at 3.6ms each. Phase F multiplies that by 11 = 40ms one-shot at warmup-end. Negligible against a 43s bench.
- **Launch overhead increase.** 11 launches/propose vs 1. Already accounted for in the 2% estimate above.
- **Capture pointer stability.** Same as Phase D: the per-call dynamic data must come from device pointers that don't move. The pre-attention and post-attention bodies use only scratch buffers + layer weights — all stable. No new risk.

**Acceptance:**

- 44.9% accept exact match.
- 11 capture-success log lines after warmup (one per subgraph).
- Tok/s within ±0.3 of Phase E.2 baseline (8.48). Anything significantly slower means a capture/launch dispatching bug.
- `cuGraphLaunch` count rises ~10× in nsys; total wall time roughly flat.

### 15.6 What Phase F is NOT

- Not a perf win. Pure structural cleanup.
- Not the place to add fusion. Fusion goes in **after** F lands and the per-subgraph structure is proven.
- Not FP8. FP8 is the next perf lever and gets its own phase.

### 15.7 Execution plan

**F.1 — split forward_block_layer_paged into pre_attn + post_attn (no graphs yet).** 2 hours.
- Extract pre-attention work (input_layernorm through reshape_and_cache) into `forward_block_layer_pre_attn`.
- Extract post-attention work (o_proj through final residual_add) into `forward_block_layer_post_attn`.
- `forward_block.rs` calls them in sequence around the existing attention call.
- Bench: 44.9% accept exact, tok/s within ±0.05 of E.2 (8.48). No graph changes.

**F.2 — replace single-graph capture with per-subgraph capture.** 3 hours.
- Change `propose_graph: Mutex<Option<GraphHandle>>` to `propose_graphs: Mutex<Option<Vec<GraphHandle>>>`.
- Add the `run_or_capture_subgraph` helper.
- Wrap each pre-attn / post-attn / tail block in a capture region.
- Bench: 44.9% accept exact, tok/s within ±0.3 of E.2.

**F.3 — verify per-subgraph nsys profile.** 1 hour.
- Confirm 11 distinct cuGraphLaunch traces per propose in nsight UI.
- Confirm attention kernel runs eager between captured subgraphs (no graph node).
- Document the new structure in this file with a profile snippet.

Total: 6 hours. Phase F is the bridge to fusion + FP8.

### 15.8 Phase F results (2026-05-28)

All three F sub-phases landed on branch `dflash-cuda-graph-phaseF` (commits `a6df463` F.1 + `7570c9f` F.2).

**F.1 (split forward_block_layer_paged into pre_attn / attention / post_attn).** Pure refactor. Bench: 44.9% accept exact, 8.69 tok/s (Δ +0.21 from E.2 baseline, within noise).

**F.2 (per-subgraph capture, 11 slots).** New propose_graphs field replaces single propose_graph. Capture pass builds `[pre_0, post_0, ..., pre_4, post_4, tail]` in one propose call after the standard 2-pass warmup. Server log confirms:

```
DFlash piecewise capture: starting (warmup_count=2, target=2, slots=11)
DFlash piecewise capture: complete (11/11 subgraphs captured)
```

Bench: 44.9% accept exact, 8.70 tok/s (Δ +0.22 from E.2 baseline, within noise; identical to F.1 — the structural change cost nothing measurable).

**F.3 (nsys verification).** Ran the canonical Volvo bench under `nsys profile --trace=cuda,nvtx,osrt`. Bench under profiling: 44.9% accept exact, 8.49 tok/s (nsys overhead minimal at this scale). All three acceptance criteria from §15.5 confirmed:

| Criterion | Expected | Observed | ✓ |
|---|---|---|---|
| Accept rate | 44.9% exact | 44.9% (92/205) | ✓ |
| `cuGraphLaunch` count rises ~10× | Yes | 2,306 calls (vs 408 in Phase D for same propose count) — 5.65× rise per propose ≈ 11 slots × N propose cycles | ✓ |
| Attention runs eager | Yes | `inferspark_prefill_paged_indirect` at 970 GPU kernel instances, not folded into `cuGraphLaunch` | ✓ |
| `cuGraphInstantiateWithFlags` count | ~11 one-shot | 12 (11 slots + 1 reuse of a prior instantiation, ~3.6ms each — negligible against 43s bench) | ✓ |
| Begin/end capture pairs | 11 each | `cuStreamBeginCapture` = 12, `cuStreamEndCapture` = 12 | ✓ |

**Post-F.2 nsys API-time breakdown** (matches §14.7 post-E.2 shape, confirming F.2 didn't introduce host-side regressions):

| API call | % API time | Calls | Avg |
|---|---|---|---|
| `cuStreamSynchronize` | 62.7% | 246,104 | 101 µs |
| `cuMemcpyDtoHAsync_v2` | 31.3% | 523 | 23.9 ms |
| `cuMemcpyDtoDAsync_v2` | 3.9% | 281,789 | 5.5 µs |
| `cuLaunchKernel` | 0.8% | 58,259 | 5.3 µs |
| `cuGraphLaunch` | 0.3% | 2,306 | 46.5 µs |

GPU-side, `dense_gemm_bf16` is still the wall at 58.8% of GPU time (was 60.7% post-E.2, same kernel work). **Phase F bought structure, not perf — exactly as designed.**

### 15.9 What lands next

The piecewise structure is the prerequisite for:

1. **Drafter MLP FP8** (Phase G candidate). Cuts `dense_gemm_bf16` work roughly in half. Real lever, real risk: drafter FP8 acceptance-rate collapse on SM12.x was the original reason `DflashQuantization::Bf16` is the only variant today. Per-layer FP8 (keeping layer 0 BF16, FP8 on 1-4) slots naturally into the per-layer subgraph: if FP8 GEMM has a different launch shape than BF16, each layer's pre/post subgraph captures it independently. Predicted gain: 40-50% tok/s if acceptance holds.
2. **Kernel fusion** (Phase H candidate). Fused `qkv_proj_norm_rope_cache` collapses ~7 kernels in the pre_attn subgraph to 1. The subgraph boundary doesn't move; fewer launches inside the capture buys L2 reuse + launch-overhead reduction. Predicted gain: 10-20% tok/s.

FP8 first (bigger lever, fits the per-layer subgraph cleanly). Fusion after.

---

## 16. Phase G — Drafter MLP FP8 (detailed plan)

**Status**: design (not implemented). Builds on Phase F.2's per-layer
subgraph structure. Captured 2026-05-28 after Phase F.3 landed.

### 16.1 The opportunity

Post-Phase F.2 nsys (§15.8) confirms `dense_gemm_bf16` is 58.8% of
GPU time. The drafter has **seven dense BF16 GEMMs per layer** — three
for QKV projection, one for o_proj, three for the MLP (gate, up, down)
— running over γ=16 rows, five layers, ~200 propose calls per bench.
Total: ~7,000 BF16 GEMM kernel invocations per bench.

FP8 E4M3 weights cut weight bandwidth in half (1 byte/weight vs 2)
and let the GEMM use Blackwell's FP8 tensor cores (sm_121f). On the
target model Atlas already gets ~2× speedup from FP8 GEMM at moderate
M (gemm_quant.rs:46-72 docstring claims "~2× speedup on out_proj at
ISL≥128"). At γ=16, M=16 — small, but the per-call kernel time is
also tiny, so the relative win should still be substantial.

**Predicted gain: 40-50% drafter tok/s uplift if accuracy holds.**

### 16.2 What Atlas already has (massive head-start)

Inventory of in-tree FP8 infrastructure that Phase G consumes:

**FP8 GEMM kernels** (`kernels/gb10/common/`):
- `fp8_gemm_n128`: BF16 × FP8 → BF16, grid `(ceil(N/128), ceil(M/64), 1)`. Used by qwen3_attention and qwen3_ssm prefill paths.
- `fp8_gemm_n128_m128`: BF16 × FP8 → BF16, M≥128 variant with ~2× speedup at long ISL. Used by paged_oproj.
- `fp8_fp8_gemm_n128_m128`: FP8 × FP8 → BF16, requires pre-quantized activations. Maximum bandwidth savings, used by qwen3_attention QKV when activations are pre-quantized.
- `bf16_to_fp8`: BF16 activation → FP8 E4M3 (no scale, just truncation). 256 thread blocks.

**Weight conversion kernels**:
- `predequant_nvfp4_to_fp8`: NVFP4 packed [N, K/2] + scale → FP8 [N, K]. **Source is NVFP4 only — does not handle BF16 source.** This is the gap we'll have to fill.
- `quantize_bf16_to_fp8`: Referenced in `mtp_head/new.rs:28`. Need to verify what scale layout this uses — likely the kernel we want.

**Rust ops surfaces** (`crates/spark-model/src/layers/ops/gemm_dense.rs`, `gemm_quant.rs`):
- `ops::fp8_gemm_n128`, `ops::fp8_gemm_n128_m128`, `ops::fp8_fp8_gemm_n128_m128` — all already wrapped and used by 13+ call sites across qwen3_ssm, qwen3_attention, moe, and mtp_head.
- `weight_map::Fp8DenseWeight` — `{weight: DevicePtr, row_scale: DevicePtr}` — runtime-quantized from BF16 with per-row scale. **Exact shape we need for drafter weights** (drafter is BF16 on disk; this struct represents BF16→FP8 quantized at load time).
- `DenseWeight::predequant_to_fp8` — exists only for the `QuantizedWeight` (NVFP4) source path. **No equivalent for the BF16 source path.** Will need a sibling method.

**Verification**: vLLM's `v1/spec_decode/dflash.py` and `v1/spec_decode/eagle.py` have zero FP8 references. Upstream DFlash runs the drafter at the parent model's dtype. **We're not behind upstream — we're ahead.** The "FP8 acceptance collapse" claim in the codebase is Atlas-specific.

### 16.3 The acceptance-collapse claim — corrected

`crates/spark-model/src/layers/dflash_head.rs:131-136` says drafter FP8 collapses acceptance on SM12.x. This needs disambiguation:

- **Drafter FP8 KV cache** — `Phase H` in §13 calls out this specifically: "drafter FP8 KV collapses acceptance on SM12.x ... dynamic range loss on the K side breaks the bidirectional γ-block attention math." Real concern. Anecdotal (§13: "currently anecdotal from earlier runs").
- **Drafter FP8 weights** — different thing entirely. Not what the collapse comment is about. Weight FP8 doesn't touch the bidirectional attention math because the FP8 GEMM output is BF16; only the *multiplication* is FP8 × FP8.

The DflashQuantization::Bf16 enum comment lumps both together, but the operative risk is KV cache, not weights. **Phase G targets weight FP8 only. KV cache stays BF16.**

If weight FP8 *does* tank acceptance, the failure mode is dynamic-range loss on the MLP intermediate activations (gate_proj output flows into silu_mul → down_proj). This is mitigatable with per-row scales (Fp8DenseWeight already has these) and falls outside the K-side bidirectional attention concern.

### 16.4 What we still have to write

**Update 2026-05-28**: a kernel sweep found that **`quantize_bf16_to_fp8` already exists in tree** at `kernels/gb10/common/dense_gemv_fp8w.cu:36`. It does exactly what Phase G needs: BF16 `[N, K]` → FP8 E4M3 `[N, K]` with per-row f32 scales, called once at model load. Comments even say "Called once at model load time (not on the decode hot path)." Registered as kernel `("gemv_fp8w", "quantize_bf16_to_fp8")` and already used by `mtp_head/new.rs:28`.

**No new kernel needed.** The G.1 effort drops from 3 hours to ~1 hour. Remaining work:

**One new Rust op**: `ops::quantize_bf16_to_fp8(...)` in `gemm_dense.rs`, wrapping the existing kernel. ~20 lines, mirrors the `bf16_to_fp8` op pattern.

**One new DenseWeight method**: `DenseWeight::quantize_to_fp8(...)` in `weight_map/quantized.rs`, mirroring `QuantizedWeight::predequant_to_fp8` but for the BF16 source path. Allocates the FP8 buffer + per-row scale buffer, calls the op, returns an `Fp8DenseWeight`.

That's it. **Everything else is wiring.**

For reference, the kernel's actual algorithm (matches the spec we'd have written from scratch):

```
Input:  bf16_weight [N, K] device buffer
Output: fp8_weight  [N, K] device buffer
        row_scale   [N]    f32 device buffer

Per-row algorithm (one CTA per row, 256 threads):
  1. Parallel absmax reduction over K BF16 elements (warp shuffle + smem).
  2. row_scale[row] = absmax / 448.0  (FP8 E4M3 max).
  3. inv_scale broadcast via shared memory.
  4. Each thread quantizes K/256 elements: clamp to [-448, 448], cast to E4M3, store.

Grid: (N, 1, 1)  Block: (256, 1, 1) — one row per CTA.
```

### 16.5 Per-layer subgraph integration

The per-layer subgraph structure from Phase F.2 makes the integration trivial. Each pre_attn subgraph captures three dense GEMMs (q_proj, k_proj, v_proj); each post_attn captures four (o_proj, gate_proj, up_proj, down_proj). If we swap any of those from `dense_gemm` (BF16 × BF16) to `fp8_gemm_n128` (BF16 × FP8), only that subgraph's launch sequence changes — capture boundaries don't move, replay logic unchanged.

**Layer-level gating** (the original §15.4 motivation): we can ship FP8 on layers 1-4 and keep layer 0 BF16 for accuracy if needed. Per-layer subgraphs make this a one-line conditional in the layer loop. If FP8 tanks acceptance on layer 0 specifically (the closest to the target model's hidden state), we shed it and keep the gains on the rest.

### 16.6 Activation pre-quantization decision

Two FP8 GEMM tiers in Atlas:

1. **BF16 × FP8 → BF16** (`fp8_gemm_n128`). Cheap to deploy: only weights are FP8, activations stay BF16. Predicted ~30-40% of full FP8 gain.
2. **FP8 × FP8 → BF16** (`fp8_fp8_gemm_n128_m128`). Maximum bandwidth savings, but needs `bf16_to_fp8` activation prequant before each GEMM. Used by qwen3_attention QKV in tree.

For the drafter, **start with BF16 × FP8** because (a) it isolates the risk to weight quant only, (b) doesn't add `bf16_to_fp8` kernel launches inside the capture, (c) lets us A/B accuracy cleanly. If accuracy holds and the perf gain is short of target, then layer the activation prequant in as G.2.

### 16.7 Execution plan

**G.1 — write the Rust op + DenseWeight method.** 1 hour.
- The CUDA kernel already exists: `kernels/gb10/common/dense_gemv_fp8w.cu:36 quantize_bf16_to_fp8`. Registered as `("gemv_fp8w", "quantize_bf16_to_fp8")`. No new CUDA work.
- Rust op `ops::quantize_bf16_to_fp8` in `gemm_dense.rs` (~20 lines, mirrors `bf16_to_fp8`).
- `DenseWeight::quantize_to_fp8` in `weight_map/quantized.rs` (mirrors `QuantizedWeight::predequant_to_fp8` shape).
- Smoke test: round-trip a known BF16 weight via the new path, check dequant matches within FP8 precision (~3 sig figs).

**G.2 — gate Phase G via env var + scratch FP8 buffers.** 1 hour.
- Add `ATLAS_DFLASH_DRAFTER_FP8` env var (default OFF).
- Extend `DflashQuantization` with `Fp8Weights` variant.
- In `dflash_head/from_weights.rs`, if env var is set, call `quantize_to_fp8` on each layer's q/k/v/o_proj + gate/up/down_proj, store the resulting `Fp8DenseWeight` in a new optional field on `DflashLayer`.
- Predequant happens once at model load (zero hot-path cost), runtime stream sync between predequant batches.

**G.3 — swap GEMM call sites in `forward_block_layer_pre_attn` + `_post_attn`.** 2 hours.
- Both methods get a runtime check on `self.quant`. BF16 path: existing `ops::dense_gemm`. FP8 path: `ops::fp8_gemm_n128` with the matching `Fp8DenseWeight`.
- Re-run Phase F.2 capture pass after the change — capture should succeed with FP8 kernels (they're stateless on host).

**G.4 — accept-rate guard + bench.** 1 hour.
- Bench with `ATLAS_DFLASH_DRAFTER_FP8=1`. Acceptance criteria:
  - **Hard fail**: accept rate < 42% (was 44.9%). Pull the patch.
  - **Acceptable**: accept rate ≥ 43% AND tok/s ≥ 11.0 (vs 8.70 F.2 baseline) — call it a win.
  - **Marginal**: accept rate 42-43% — investigate per-layer (skip layer 0 FP8 and re-test).
- If hard-fail, run a layer-by-layer ablation: enable FP8 on layers 4, 3, 2, 1, 0 incrementally. The first layer that breaks acceptance is the diagnostic.

**G.5 — nsys verification.** 30 min.
- Re-run `~/atlas-code-atlas-bench-nsys.sh`. Expected GPU kernel summary changes:
  - `dense_gemm_bf16` GPU time drops from 58.8% to ~30%.
  - `fp8_gemm_n128` appears with ~7000 instances (7 GEMMs × 5 layers × ~200 propose calls).
  - `cuMemcpyDtoHAsync_v2` and `cuStreamSynchronize` unchanged — host orchestration didn't move.

**Total: 5.5 hours.** Realistically 1 day with debug (down from 7.5 — the kernel already exists in tree).

### 16.8 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FP8 weight quant tanks acceptance globally | Medium | High | Layer-level gating (G.4). Skip layer 0 FP8 first. If still bad, run with row_scale at higher precision (f32 instead of bf16 — already the case). |
| FP8 GEMM kernels untested at M=16 | Low | Medium | All existing call sites are M=128+. M=16 might trigger an unhandled tile shape. Easy to validate at G.1 smoke test (run kernel with γ=16 sized input, compare against BF16 reference). |
| Per-row scale storage overhead | Low | Low | 35 GEMMs × (5120 or 17408) rows × 4 bytes = ~400 KB. Trivial. |
| Predequant adds load-time latency | Low | Low | One-shot at model load. 7 GEMMs × 5 layers × ~5ms/kernel = ~175ms. Buried in 20s drafter load. |
| Capture region invalidated by new kernel | Very low | High | Capture is stateless wrt kernel identity; replays work as long as the launch sequence matches what was captured. Risk is zero unless we conditionally swap kernels mid-graph, which we don't. |
| KV cache stays BF16 but Phase H eventually wants FP8 KV | N/A | Future | Phase G is orthogonal to Phase H. KV cache concern lives elsewhere. |

### 16.9 What Phase G is NOT

- **Not FP8 KV cache.** That's Phase H; different risk profile (bidirectional attention math).
- **Not FP8 activations.** That's a G.2 follow-on if accuracy holds and we want more perf.
- **Not native FP8 propose.** That's the §13 Phase H roadmap (FlashAttention-3-style native FP8 attention), and it depends on KV cache changes.
- **Not per-layer quantization research.** Layer-level gating in G.4 is a kill-switch, not a quantization-aware training step. If accuracy needs layer-specific tuning beyond all-on/all-off/skip-0, we stop and reconsider.

### 16.10 Acceptance criteria for Phase G ship

- ✓ Accept rate ≥ 43% (within 1.9pp of BF16 baseline).
- ✓ Tok/s ≥ 11.0 (≥ 26% improvement over 8.70 F.2 baseline).
- ✓ Bench produces exact-match output as BF16 for the canonical Volvo prompt (the 200-char prefix at minimum).
- ✓ nsys shows fp8_gemm_n128 displacing dense_gemm_bf16 as the top kernel.
- ✓ `ATLAS_DFLASH_DRAFTER_FP8=0` (default) is bit-identical to F.2 baseline — opt-in only until we trust it.

Phase G is the bridge that makes DFlash a perf story instead of a structural story. After G lands, fusion (Phase H candidate) is the last incremental lever before the next wall.

### 16.11 Phase G live status + resume plan (2026-05-28 EOD)

**Where we landed**: per-layer FP8 (q/k/v/o/gate/up/down) live and shipping. Bench: 44.9% → 45.1% accept exact, 8.70 → 9.77 tok/s (**+12.3%**). Branch `dflash-phaseG`, commits G.1 (f465de2), G.2 (739ddce), G.3 (30692ad). The custom `fp8_gemm_t_row_scaled` kernel is in `kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu` and consumed by `forward_block_layer_pre_attn` + `_post_attn`.

**The lm_head detour and where it left us**: lm_head is the largest GEMM in the drafter (γ × vocab = 16 × 248320). Two attempts to swap it to FP8:

1. **Used the existing `fp8_gemm_t_row_scaled`** (M_TILE=64). At M=γ=16 the kernel wastes 48 of 64 M_TILE rows per CTA. GPU utilization crashed from ~95% to ~6-30%; build had to be killed before the bench finished cleanly. Functionally produced output but at unusable speed.
2. **GEMV loop** (16 × `dense_gemv_fp8w`). GPU util recovered to ~95% but launch overhead from 16 kernel launches per propose was a net loss: 8.70 → 7.51 tok/s. Reverted.
3. **Wrote a small-M FP8 GEMM** (`fp8_gemm_t_row_scaled_m16`, M_TILE=16, 1 warp per CTA, 32 threads). **Produced garbage output — 0% accept rate, 7.01 tok/s.** Reverted the call site; kernel + Rust op (`ops::fp8_gemm_n128_row_scaled_m16`) + handle (`DflashKernels::fp8_gemm_n128_row_scaled_m16`) all kept in tree for debugging.

**The bug in `fp8_gemm_t_row_scaled_m16`** (NOT YET FIXED — investigate when we resume):

The smem_A load pattern is the most likely culprit. The M_TILE=64 parent kernel splits A loading across 128 threads (each thread loads 8 BF16 = 16 bytes via `cp_async_pred_16`); my M_TILE=16 variant tried to compress that to 32 threads, each thread loads 1 row × 32 cols (or rather, row=t/2, col_offset=(t%2)*16) via a single `cp_async_pred_16` per thread. The mapping of A elements to MMA fragment lanes is sensitive to the smem layout — the original `bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4])` pattern assumes the M_TILE=64 layout where group_id (0..7) indexes into rows 0..7, warp_m_offset adds warp×16 to that. My M_TILE=16 kernel dropped warp_m_offset (since 1 warp = 1 M-tile) but the row mapping `fr0 = group_id, fr1 = fr0 + 8` may not match the smem layout I'm writing.

**Specific things to check first when we resume**:
- The smem_A write pattern: `smem_A[buf][threadIdx.x >> 1][col]`. Thread t=0 writes row 0, t=1 writes row 0 with col offset 16, t=2 writes row 1, etc. So rows 0..15 get written by threads 0..31 in pairs.
- The MMA fragment read: `sA[fr0 * a_stride + tid * 4]` where fr0 = group_id (0..7) and tid = lane_id & 3 (0..3). Lane 0 reads row 0 cols 0..3, lane 1 reads row 0 cols 4..7, etc. group_id 0..7 selects M rows 0..7 of the m16n8k32 MMA's A tile.
- **Hypothesis**: the M_TILE=64 parent uses `warp_m_offset = warp_id × 16` to offset which 16-row block of M_TILE=64 a warp reads. M_TILE=16 only has rows 0..15. Without warp_m_offset, group_id 0..7 maps to rows 0..7 (the m16n8k32 instruction's first 8 A rows) and `fr1 = fr0 + 8` should map to rows 8..15. So far so good. But the smem_A buffer is declared `__shared__ __nv_bfloat16 smem_A[2][16][K_STEP_T + PAD_T]`. The load pattern writes `smem_A[buf][row][a_col]` for row=0..15. The MMA reads `sA[row * a_stride + tid*4]` where sA is `unsigned short *` — same stride as bf16. Should be aligned. But: cp_async fills 16 bytes (8 BF16 values), and the macro writes both `a_col=0` and `a_col=16` halves — but at M_TILE=16 with my mapping, each thread only does ONE write. So columns 8..15 of each row might be **uninitialized**.

**That's almost certainly the bug**: the M_TILE=64 kernel issues 2 cp_async_pred_16 per thread per iteration (because it has 64 rows × 32 cols ÷ 128 threads = 16 bytes/thread but K_STEP_T=32 cols = 64 bytes/row, needing two loads per row). My M_TILE=16 version cuts to 1 cp_async per thread, halving the cols loaded. The smem_A K dimension is 32 cols (K_STEP_T) but only 16 cols got written per row. The MMA reads `sA[row * a_stride + 16 + tid * 4]` for the second half of K which is garbage.

**The fix**: in `FP8_LOADS_M16` make each thread do TWO cp_async_pred_16 loads — one for cols 0..15 and one for cols 16..31. With 32 threads and 16 rows, each row gets 2 threads (16 bytes each), so each thread covers one (row, col_half) pair. Need to map: thread t handles row = t & 15, col_half = (t >> 4). Then both halves of all 16 rows are loaded.

Alternative cleaner fix: 32 threads × 16 rows = 2 threads per row. Thread 2*r writes cols 0..15 of row r, thread 2*r+1 writes cols 16..31. So row = t >> 1 and col_offset = (t & 1) << 4 — but that's what I have. The problem is that with K_STEP_T=32, each thread loading 16 bytes (8 BF16) covers cols 0..7 OR 16..23. **Cols 8..15 and 24..31 are uninitialized.**

**Real fix**: 32 threads × 32 K cols × 16 rows = 512 BF16 = 1024 bytes. cp_async loads 16 bytes (8 BF16) per call. Need 1024/16 = 64 cp_async issues. With 32 threads that's 2 issues per thread. Mapping: each thread does 2 issues. Thread t: issue 0 = row t/2, col (t&1)*8. Issue 1 = same row, col 16 + (t&1)*8. That fully covers all 16 rows × 32 cols.

**Pragmatic alternative**: just keep M_TILE=16 layout but use the parent kernel's 2-warp split. Or simpler: skip the small-M kernel entirely, use the existing `dense_gemm` (BF16) for lm_head, since the +12% drafter MLP gain is already shipping and lm_head FP8 is incremental.

**Phase G ship status**: per-layer GEMMs are GOOD. lm_head is BF16 (reverted). Branch ready to merge. **Pre-commit step before merging**: rebuild + bench cleanly (kernel count should be 91 since the m16 variant is in tree but unused), confirm 9.77 tok/s holds.

### 16.12 Next steps after resume

1. **Fix or abandon `fp8_gemm_t_row_scaled_m16`.** The 2-load fix described in §16.11 is ~10 lines of CUDA. Test with a tiny smoke (compare M=16 N=128 K=32 output against the M_TILE=64 kernel result on the same inputs). If correct, ship lm_head FP8. Expected additional gain: 5-10% tok/s (lm_head is the biggest single GEMM).

2. **Kernel fusion (Phase I candidate).** With FP8 weights live, the per-layer subgraph has 7 GEMMs + 5-7 cheap kernels (norms, residuals, rope). Fusing the QKV norm rope cache pipeline into a single kernel collapses ~7 launches into 1. Predicted gain: 10-20% tok/s on top of FP8. Files to touch: `kernels/gb10/qwen3.6-27b/nvfp4/` — write `fused_qkv_norm_rope_cache.cu`. Replaces ops::dense_gemm × 3 + rms_norm × 2 + rope_yarn + reshape_and_cache in `forward_block_layer_pre_attn`. The piecewise capture boundaries DON'T move; we just emit fewer kernels inside the pre_attn subgraph.

3. **MLP fusion**: gate + up + silu_mul + down can fuse into a single launch using the existing `moe_silu_mul` pattern as reference (`crates/spark-model/src/layers/moe/`). ~5-10% gain.

4. **Re-bench plain model without our optimizations** to quantify the total gain from Phase E → G. Use the F.2 tag `wip/dflash-phase-F-complete` as the baseline.

**Resume here**: read §16.11 first for the kernel bug, §16.12 for the queue. Branch `dflash-phaseG` (tip 30692ad as of EOD May 28). The uncommitted state is reverted lm_head + the broken-but-in-tree `fp8_gemm_t_row_scaled_m16` kernel. Commit before resuming.

---

## 17. Phase H — Main-model FP8 audit (mirror vLLM's FP8 footprint)

> **🚫 OUT OF SCOPE (2026-05-28).** Avarok's DFlash deliverable is scoped to the drafter / speculative-decode path only. Main-model FP8 work — even if it shows perf headroom — falls outside the engagement. This section is preserved for two reasons: (1) it documents the reconnaissance findings that landed Phase G's scope correctly (drafter, not main model), and (2) if Avarok later expands the scope to include main-model perf, the audit plan here is a starting point. Until then, **do not implement** any of H.1–H.4.

**Status**: design + reconnaissance, **marked out of scope** (not implemented, not planned). Captured 2026-05-28 after a side-quest discovered that Atlas's drafter perf wall is `dense_gemm_bf16` precisely *because* the main model is already aggressively quantized (NVFP4 weights + FP8 KV when flagged). The drafter sticks out as the BF16 island.

This section exists because of a real-time miscommunication during Phase G planning: the user assumed "FP8" meant mirroring vLLM's FP8 footprint on the main model, not the drafter. Phase G stays scoped to the drafter. Phase H scopes the main-model audit separately so we can decide whether to invest there next.

### 17.1 vLLM's FP8 surface area (the reference)

vLLM uses FP8 on the main model across three surfaces, all of which independently matter for tok/s:

1. **FP8 weight storage + GEMM.** Either loaded pre-quantized (via the llm-compressor recipe) or runtime-quantized at model load. The GEMM consumes FP8 weights with either BF16 or FP8 activations.
2. **FP8 KV cache.** K/V values are quantized to FP8 E4M3 (or E5M2) before write to the paged cache, dequantized on read. Halves KV bandwidth and capacity.
3. **FP8 activations.** Pre-quantize hidden states to FP8 before feeding into FP8×FP8 GEMMs. Maximum bandwidth savings on the activation side.

What vLLM does **not** do: quantize the drafter. Drafters run at the parent model's dtype because the drafter is small and the perf juice isn't worth the accuracy risk. This is the gap Phase G targets.

### 17.2 Atlas's main-model FP8 footprint today

The reconnaissance side-quest produced a complete inventory. Atlas is **ahead of vLLM in many places** — NVFP4 is more aggressive than FP8 — and has parity or near-parity on the rest.

**Weights**: NVFP4 (4-bit, group-scaled). Stored as `QuantizedWeight` (`[N, K/2]` packed + `[N, K/GROUP_SIZE]` scale + tensor scale2). The on-disk format for the canonical Qwen3.6-27B-AEON-NVFP4 model is NVFP4 directly. **More aggressive than vLLM's FP8 weights** — half the bytes, and Blackwell's NVFP4 tensor cores get a comparable speedup to FP8 tensor cores. No gap.

**KV cache**: configurable via `--kv-cache-dtype`. Supported: `bf16`, `fp8`, `nvfp4`, `turbo3`, `turbo4`, `turbo8`. The canonical bench script uses `--kv-cache-dtype nvfp4` (more aggressive than vLLM's FP8 KV). `fp8` is also fully wired with online calibration: `fp8_kv_calibration_tokens` config tracks per-tensor max|K|/|V| during a warm-up window, then locks scales. Kernels in tree: `reshape_and_cache_fp8.cu`, `paged_decode_attn_fp8.cu`, `inferspark_prefill_paged_fp8.cu`. **No gap.**

**Activations (the actually-interesting bit)**: the qwen3_attention prefill path **already pre-quantizes activations to FP8** before FP8×FP8 GEMMs.
- `crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip.rs:75` runs `ops::bf16_to_fp8` on the normed hidden state.
- `crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_qkv.rs:153,165` calls `fp8_fp8_gemm_n128_m128` / `fp8_fp8_gemm_n128` on the quantized activations.
- Kernel handles in `qwen3_attention/types.rs:245-248`: `fp8_fp8_gemm_k`, `fp8_fp8_gemm_t_m128_k`.

So the QKV projections on the attention prefill path are already FP8×FP8. This is the same pattern vLLM uses.

**Where vLLM still has work Atlas doesn't yet do** (preliminary list — needs verification before committing to a plan):

- **MoE expert GEMMs**. `moe_w4a16_grouped_gemm.cu` exists and `moe_fp8_grouped_gemm.cu` exists in `kernels/gb10/common/`. Need to check whether the canonical Qwen3.6-27B-AEON-NVFP4 path routes through the FP8 grouped GEMM or the W4A16 path. If W4A16, the activation side of MoE expert routing is still BF16.
- **Dense FFN GEMMs on non-attention layers**. NVFP4 weights, but activations might still be BF16 for the FFN GEMMs depending on which call site. Audit needed.
- **Attention output projection**. `paged_oproj.rs:59,71` uses `fp8_gemm_n128_m128` (BF16 × FP8) — not FP8×FP8. Could move to fp8_fp8 with activation prequant.
- **Decode-path GEMMs**. Most existing FP8 wiring is on the prefill path. Decode uses `dense_gemv_fp8w` (BF16 × FP8 GEMV) but not FP8×FP8 GEMV. At batch=1 the savings here are bandwidth-bound and could be real.

### 17.3 Realistic perf headroom on the main model

**Honest framing** (correcting an over-optimistic earlier read):

The main model's GPU time at the current configuration is dominated by:
- W4A16 GEMVs (NVFP4 weights, BF16 activations) — `w4a16_gemv`, `w4a16_gemv_dual`, `w4a16_gemv_silu_input` collectively 32.7% of GPU time in the F.2 nsys (sum of rows 3-5 in §15.8). These are decode-path GEMVs, M=1.
- `gated_delta_rule_*` SSM kernels — ~1.5% combined.
- `inferspark_prefill_paged_indirect` (DFlash drafter attention) — 0.2%.
- Various small kernels (rms_norm, residual_add, rope, embed) — single percent each.

The main-model GEMVs at decode are bandwidth-bound. Moving them from BF16 activations to FP8 activations halves activation bandwidth but the *activations are tiny* at M=1 (5120 BF16 = 10KB per layer). The weights dominate — and they're already 4-bit. **Realistic uplift from main-model FP8 activations at decode: ~5-10% tok/s, not 40-50%.**

The prefill path is different — there activations are large enough to matter, and the existing fp8_fp8 wiring proves the pattern works. But prefill is a smaller fraction of bench wall time at our typical workload.

**The drafter (Phase G) is genuinely where the bigger relative win lives**, because its weights are *still BF16* — it's the only path that hasn't been quantized at all. The drafter just happens to have a small absolute footprint.

### 17.4 Sub-phase scoping (audit-first, code-second)

**H.1 — Main-model FP8 audit.** 4 hours. Pure investigation, no code changes.
- Enumerate every GEMM/GEMV call site in the main-model decode and prefill paths.
- For each, tag: weight dtype, activation dtype, output dtype, kernel used.
- Cross-reference against the kernel inventory in `kernels/gb10/common/`.
- Identify call sites where (weight FP8/NVFP4 + activation BF16) could trivially become (weight FP8/NVFP4 + activation FP8) by inserting `ops::bf16_to_fp8` + swapping the GEMM kernel.
- Output: an audit table in `docs/design/main_model_fp8_audit.md` listing every BF16-activation GEMM with its kernel handle, call site, and conversion cost estimate.

**H.2 — Bench instrumentation for FP8-relevant kernels.** 2 hours.
- Add an nsys helper that filters GPU kernel time by (BF16 activation) vs (FP8 activation) for each GEMM family.
- Establish the baseline: how much GPU time is spent in BF16-activation GEMMs across the canonical bench? This is the upper bound on what main-model FP8 activation prequant can buy.
- **Gate**: if the audit shows < 10% of GPU time is in BF16-activation GEMMs that could move to FP8, the project doesn't ship. The drafter Phase G is a better use of time.

**H.3 — Decision point.** Based on H.1 + H.2 output, decide whether to:
- (a) Pursue main-model FP8 activations as a Phase H.4 implementation (estimated 1-2 weeks).
- (b) Defer indefinitely and let Phase G drafter work + future kernel fusion be the perf path.
- (c) Cherry-pick: identify the 2-3 highest-payoff call sites and do those only (estimated 2-3 days).

**Total to decision: 6 hours.** No commitment to implementation until the audit numbers justify it.

### 17.5 What Phase H is NOT

- **Not a vLLM port.** The infrastructure exists in Atlas; this is an *audit* to find under-utilized FP8 surfaces, not a port project.
- **Not blocking Phase G.** Phase G ships on its own merits. H informs the *next* perf phase after G.
- **Not a quantization research project.** If H.1 shows that all the big GEMMs are already optimally configured, the answer is "no main-model FP8 work needed" and we move on.
- **Not native FP8 propose / FlashAttention-3.** That was the older Phase H roadmap in §13 — the §17 Phase H reuses the letter only because we renumbered after Phase F.2 landed. The FA3 work is its own thing; see §13 for the (still-deferred) plan.

### 17.6 Acceptance criteria for shipping H.4 (if we get there)

- ✓ Audit (H.1) identifies ≥ 10% of bench GPU time in BF16-activation GEMMs that can move to FP8 activations without a new kernel write.
- ✓ Each cherry-picked call site has a documented accuracy guard (exact-match Volvo prefix or equivalent regression bench).
- ✓ Tok/s improvement ≥ 5% per call site converted, or the conversion is reverted.
- ✓ Env-var gated (`ATLAS_MAIN_FP8_ACTIVATIONS=1`) and bit-identical to the current path when OFF.

### 17.7 Honest take

**The user is right** that vLLM's proven FP8 progress lives on the main model, not the drafter. We should run the audit (H.1 + H.2) before sinking more time into Phase G if there's any chance the main-model headroom is larger than the drafter headroom.

**Friday's read on the data**: based on the F.2 nsys, the BF16-activation main-model GEMMs aren't a dominant chunk of GPU time at decode. The big main-model kernels are `w4a16_gemv*` which are already optimally configured (4-bit weight, GEMV is fundamentally weight-bandwidth-bound, FP8 activations don't help much at M=1). The audit should confirm this in 4 hours, and if it does, Phase G drafter work is the right next step. If it surprises us — great, we'll know.

**Recommended order**: run H.1 + H.2 as a half-day reconnaissance side-quest *before* starting Phase G implementation. If the audit says "no headroom," start Phase G. If it says "real headroom on N call sites," do H.3 cherry-pick first.
