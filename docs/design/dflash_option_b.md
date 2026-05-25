# DFlash Drafter — Option B: vLLM-Architecture Parity

**Status:** design draft for review (Ronald, Friday/Stark voice)
**Goal:** kill the 32× compute amplification in `BlockDiffusionDraftHead::forward_block` so the drafter actually pays for itself.
**Bottom line up front:** Atlas already has every kernel we need. This is ~600 LoC of Rust plumbing, not a Triton/CUDA rewrite. Estimated 2–3 engineer-days for the rewrite, plus one day of correctness shake-out.

---

## 1. Current state (Atlas) — what we're paying for

**Per-propose setup** (`forward_block.rs:82–291`, runs once):

| Step | Kernel | Rows / shape | Notes |
|------|--------|--------------|-------|
| 0    | `dense_gemv` × `eff_ctx` (16) | 1×10240→2048 per call | `fc_proj` of stacked target hiddens. **GEMV loop, not batched** — 16 launches. |
| 0b   | `rms_norm` | 16 × 2048 | `hidden_norm` over the 16 ctx rows. |
| 1    | host build pos_ids + `copy_h2d` | 32 × 4 B | — |
| 2    | `memset` + `batched_embed` + `memset` | 32 rows | embed of `[last_token, mask × 15]` over γ slots; ctx slots re-zeroed. |

**Per-drafter-layer body** (`forward_block_layer.rs:77–434`), with `n_attn = eff_ctx + γ = 16 + 16 = 32` rows, `h=2048`, `q_dim=4096`, `kv_dim=512`, `inter=6144`:

| Step | Kernel | Rows | FLOPs (≈) | File:line |
|------|--------|------|-----------|-----------|
| 3a   | `rms_norm` | **32** | 32·h | `forward_block_layer.rs:78` |
| 3b.q | `dense_gemm` (q_proj) | **32** | 32·h·q_dim = 268 M | `:91` |
| 3b.k | `dense_gemm` (k_proj) | **32** | 32·h·kv_dim = 33 M | `:102` |
| 3b.v | `dense_gemm` (v_proj) | **32** | 32·h·kv_dim = 33 M | `:113` |
| 3b'.k | `dense_gemm` (k_proj re-do over ctx) | **16** | 16 M | `:128` *(overwrites K rows 0..15)* |
| 3b'.v | `dense_gemm` (v_proj re-do over ctx) | **16** | 16 M | `:139` |
| 3b'.Qz | `memset` ctx Q → 0 | 16·q_dim | trivial | `:152` |
| 3c.qn | `rms_norm` (per-head Q) | 32·num_q_heads=1024 | 1024·head_dim | `:173` |
| 3c.kn | `rms_norm` (per-head K) | 32·num_kv_heads=128 | 128·head_dim | `:184` |
| 3d   | `rope_yarn` | **32** | 32·(q_dim+kv_dim)/2 | `:212` |
| 3e   | `prefill_attention` (BF16, non-causal, HDIM=128) | **q=32, kv=32** | O(n²) = 1024 dot products × num_q_heads = ~134 M | `:244` |
| 3f   | `dense_gemm` (o_proj) | **32** | 32·h·q_dim = 268 M | `:300` |
| 3g   | `residual_add` | 32·h | 65 K | `:338` |
| 3h   | `rms_norm` | **32** | 32·h | `:356` |
| 3i.g | `dense_gemm` (gate_proj) | **32** | 32·h·inter = 403 M | `:369` |
| 3i.u | `dense_gemm` (up_proj) | **32** | 32·h·inter = 403 M | `:380` |
| 3j   | `silu_mul` | 32·inter | 197 K | `:393` |
| 3k   | `dense_gemm` (down_proj) | **32** | 32·h·inter = 403 M | `:404` |
| 3l   | `residual_add` | 32·h | 65 K | `:417` |

**Per-layer kernel launch count: ~17** (the dump branches are debug-gated). Total per-layer FLOPs ≈ **1.97 GFLOP** dominated by the MLP triplet (gate+up+down = 1.21 GFLOP, 61%) and attention QKV+O projections (602 MFLOP, 30%).

**× 5 layers** (Qwen3.6-DFlash drafter has 5 — check `num_hidden_layers`, the struct comment says "8" but `target_layer_ids` carries `[1,10,19,28,37]` which is 5 capture points; per-task spec says 5 layers):
- **~85 kernel launches** in the layer loop alone
- **~9.85 GFLOP** of compute on n_attn=32 rows

**Post-loop** (`forward_block.rs:323–377`): `rms_norm` (γ=16) + `dense_gemm` lm_head (γ × vocab × h = 16 GFLOP — actually the **dominant** single op) + γ `argmax_bf16` launches + D2H. ~20 launches.

**Total per-propose:** ≈100 kernel launches, ~26 GFLOP, ~165 ms measured.

The damning bit: with `ctx_window=64` (n_attn=80) latency was *identical* to `ctx_window=512` (n_attn=528). That smells like a **kernel-launch-floor problem**, not a compute problem — confirmed by the row-scaling math below.

---

## 2. Target state (vLLM-equivalent)

### 2a. ONCE per propose, OUTSIDE layer loop — ctx K/V derivation

For the most recent ctx token (vLLM appends one slot per accepted step, not γ at a time — see `qwen3_dflash.py:342–434`):

1. `rms_norm` × 1 — `hidden_norm` over `[1, h]` (the fc-projected current target hidden).
2. **One fused** `dense_gemm` — projects `[1, h]` through stacked `_fused_kv_weight` of shape `[L × 2 × kv_dim, h]` → `[1, L × 2 × kv_dim]`. **Single GEMM for all 5 layers' K and V.** (`qwen3_dflash.py:381–389`).
3. Per-layer `rms_norm` × 5 — k_norm on each layer's K slice.
4. **Single fused** `rope_yarn` over `[L × 1, kv_dim]` view — one launch, not L (`qwen3_dflash.py:411–418`).
5. Per-layer `reshape_and_cache` × 5 — write K/V into each drafter layer's paged cache slot at the appropriate `slot_mapping`.

**Per-step ctx setup launches: 1 + 1 + 5 + 1 + 5 = 13** (independent of γ).

On **first** propose after prompt prefill, this runs over the WHOLE accumulated prefix length (e.g. 200 tokens) — but only once. Subsequent steps append 1 (or `num_accepted+1`) slot.

### 2b. PER LAYER over γ=16 rows only

| Step | Kernel | Rows | Notes |
|------|--------|------|-------|
| a | `rms_norm` (input_layernorm) | **16** | down from 32 |
| b | `dense_gemm` qkv_proj (fused or 3 separate) | **16** | 3 GEMMs over 16 rows |
| c | per-head Q-`rms_norm`, per-head K-`rms_norm` | 16·nq + 16·nkv | down from 32· |
| d | `rope_yarn` over Q and the γ-K only | **16** | down from 32 |
| e | **`reshape_and_cache`** for the γ K/V (so future ctx attends to them too, AND so noise-rows attend to each other via cache) | 16 | NEW (current path has no cache writes) |
| f | **`prefill_attention_paged`** (BF16) — q_len=16, kv_len = ctx_len + γ (e.g. 32 or 200+16), reads K/V from layer's paged cache | q=16, kv=variable | replaces non-causal `prefill_attention` |
| g | `dense_gemm` o_proj | **16** | |
| h | `residual_add` | 16·h | |
| i | `rms_norm` post_attn | **16** | |
| j | `dense_gemm` gate + up + `silu_mul` + down | **16** | **all MLP rows go from 32 → 16** |
| k | `residual_add` | 16·h | |

**Per-layer kernel launches: ~13**, every kernel runs on **γ=16 rows** regardless of ctx_window.

### 2c. Attention kernel difference

| Aspect | Current Atlas | Option B |
|--------|---------------|----------|
| Kernel | `inferspark_prefill_h128` (full self-attn) | `inferspark_prefill_paged` (paged K/V cache walk) |
| Q rows | 32 | 16 |
| K/V layout | contiguous `[32, kv_dim]` BF16 | paged, walked via `block_table` |
| Causality | `causal=false`, no mask | bidirectional in γ-block, "older" prefix positions unmasked (use `prefill_attention_paged_fp8_dflash`-style `causal_mask_enabled=0` dispatcher) |
| Cost | O(32²) per head per layer | O(16·(ctx+16)) per head per layer — same Big-O but **2× lower at ctx=16, dramatically lower per-row** |

The win isn't theoretical attention FLOPs (those scale similarly). The win is **every other op in the layer body drops from 32 rows to 16 rows**, AND the kernel launch count drops slightly per layer.

---

## 3. File-by-file delta

### `crates/spark-model/src/layers/dflash_head/forward_block_layer.rs`
- **Stays:** the function skeleton, `LayerArgs` struct (with field changes), debug dump helpers.
- **Rip out:** lines **125–153** (ctx K/V override + Q-zero memset — moves to the new precompute), lines **150–152** ctx-Q memset, the `n_attn`-based row counts throughout (replace all `n_attn` with `g`).
- **Replace** line `:244` `ops::prefill_attention` with `ops::prefill_attention_paged` (BF16 variant — exists at `prefill_attn_main_a.rs:152`) called with `q_len=γ`, `kv_len=ctx_len+γ`, `q_offset=ctx_len`, `block_table=self.kv_cache.lock().block_table_for_layer(layer_idx)`. Needs a `kv_cache` lock + per-layer pool pointer.
- **Add:** before attention, a `reshape_and_cache` call (BF16 variant, `kv_cache.rs:56`) writing the γ rows' K/V into the layer's paged cache at `slot_mapping = [ctx_len .. ctx_len+γ]`. Slot mapping is built once in `forward_block.rs` and reused across layers.
- **LOC:** ~80 deleted, ~60 added → net **−20** (file gets smaller, hallelujah).
- **Risk:** **LOW**. Every kernel call exists. Only state plumbing changes. The paged-attn kernel expects `q_offset` for the causal compare; we pass it and use the *non-causal* DFlash dispatcher.

### `crates/spark-model/src/layers/dflash_head/forward_block.rs`
- **Stays:** debug dump helpers, position_ids construction (simplified — only γ positions now, ctx positions live in the cache), embed step.
- **Rip out:** lines **87–193** (Step 0: fc_proj + hidden_norm loop) move to a new method `precompute_ctx_kv`, called as the FIRST thing inside `forward_block`. Lines **216–266** (memset ctx slots in stream_buf, batched_embed with ctx-zero pad) — simplify to γ-only embed. Lines **200–203** position_ids: drop ctx positions, only emit γ.
- **Add:** call to new `self.precompute_ctx_kv(...)` that does the once-per-step fused KV derivation + cache insert (mirrors vLLM `precompute_and_store_context_kv`).
- **LOC:** ~80 deleted from this file, ~30 added (the heavy lifting moves to the new precompute module).
- **Risk:** **LOW**. Position bookkeeping is the tricky bit — see §6.

### `crates/spark-model/src/layers/dflash_head/precompute_ctx_kv.rs` (NEW)
- The "once per propose" ctx K/V derivation. Implements `BlockDiffusionDraftHead::precompute_ctx_kv(&self, new_ctx_count, dstate, stream)`:
  1. project the *new* ctx tokens through `fc` (batched `dense_gemm` over `new_ctx_count`, not a GEMV loop — current loop at `forward_block.rs:160–173` is wasteful regardless),
  2. `rms_norm` (`hidden_norm`),
  3. fused KV GEMM through `self.fused_kv_weight` (NEW field, see §5),
  4. per-layer k_norm + RoPE-on-K (fused view across L layers),
  5. per-layer `reshape_and_cache` writing into `self.kv_cache`.
- **LOC:** ~200 (this is the real new code).
- **Risk:** **MEDIUM**. Two specific risks:
  - **fused_kv_weight build at load time** — we have to concatenate 5 layers' `[2·kv_dim, h]` weights into `[5·2·kv_dim, h]` on device. Trivial `memcpy` job in `from_weights.rs` but the SOURCE weights currently land as 5 separate `DenseWeight`s; we either (a) build the fused tensor at load and KEEP the per-layer weights for the γ-path qkv_proj, OR (b) keep both. Both cost ~6 MB extra VRAM per layer × 2 (K and V) = 60 MB total. Negligible.
  - **k_norm fused application**: vLLM loops `for i in range(L): ops.rms_norm(...)` (line 395–401) — that's 5 sequential launches, not fused. Match it; don't get fancy yet.

### `crates/spark-model/src/layers/dflash_head/from_weights.rs`
- **Stays:** everything except KV cache config + new fused weight build.
- **Change** lines **74**: `dtype: KvCacheDtype::Fp8` → `KvCacheDtype::Bf16`. Drafter is BF16 weights; an FP8 cache here is a future optimization. Use BF16 first to land correctness. (The struct comment at `dflash_head.rs:215–220` says FP8; ignore — Phase 1 reality is the cache is unused.)
- **Add:** `fused_kv_weight` allocation + concatenation loop (~30 LoC). Optional `fused_kv_bias` — Qwen3 attention has `bias=False` for q/k/v_proj in modern checkpoints; verify against the drafter's safetensors but expect `None`.
- **Add** kernel-handle resolution for `prefill_attention_paged` (BF16) — currently the struct resolves only `prefill_attn_dflash_fp8`. Module name: `prefill_paged` (verify against existing `qwen3_attention/trait_impl.rs` references).
- **LOC:** ~50 added.
- **Risk:** **LOW**, modulo verifying the BF16 paged kernel module is compiled for the drafter target. The qwen3_attention layer uses it — it's compiled.

### `crates/spark-model/src/layers/dflash_head.rs`
- **Add** fields to `BlockDiffusionDraftHead`: `fused_kv_weight: DevicePtr`, `fused_kv_bias: Option<DevicePtr>`, plus a per-step `slot_mapping_dev: DevicePtr` of size `[γ × i32]` allocated once in scratch.
- **Add** fields to `DflashProposerState`: `ctx_token_count: usize` (how many ctx tokens are currently in the cache — distinct from `ctx_len` which counts target hidden slots in the accumulator). On rollback we shrink this back.
- **Modify** `after_verify`: implement the KV trim. For `num_accepted < γ`, mark `(γ - num_accepted)` slots as free. Atlas's `PagedKvCache` exposes free-block APIs (see `kv_cache/paged_impl.rs`).
- **LOC:** ~40 added.
- **Risk:** **MEDIUM** — `after_verify` rollback semantics. See §6.

### `crates/spark-model/src/layers/dflash_head/propose.rs`
- **Stays:** state-management plumbing (the SKIP_ROW0 patch, target_hidden_stack capture path).
- **Modify:** before `forward_block` call, run `precompute_ctx_kv` over the NEW ctx tokens since last propose (delta against `dstate.ctx_token_count`). After the forward, bump `ctx_token_count` by γ (drafter's own γ outputs become future ctx — but ONLY for the prefix that survives verify; cleaned up in `after_verify`).
- **LOC:** ~30 added.
- **Risk:** **LOW**.

### Total LOC change
- Deleted: ~160
- Added: ~410
- Net: **+250 LoC**, mostly in one new file.

---

## 4. Kernel signature audit

**Answer: (a) Yes, direct reuse, no new kernel work.**

The kernel Atlas needs for Option B's per-layer attention is **`ops::prefill_attention_paged`** at `crates/spark-model/src/layers/ops/prefill_attn_main_a.rs:152`. Signature:

```rust
pub fn prefill_attention_paged(
    gpu, kernel,
    q: DevicePtr,           // [q_len, num_q_heads, head_dim] BF16
    k_cache: DevicePtr,     // layer pool base — kv_cache.k_pool_ptr(layer_idx)
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr, // [num_blocks] u32 — layer's pages
    q_len: u32,             // = γ = 16
    kv_len: u32,            // = ctx_token_count + γ
    q_offset: u32,          // = ctx_token_count (where γ starts in logical seq)
    num_q_heads, num_kv_heads, head_dim,
    cache_block_size: u32,  // = 16 (matches drafter cache alloc)
    sliding_window: u32,    // = 0 for now (drafter not windowed)
    inv_sqrt_d: f32,
    stream,
)
```

Used in production today by `Qwen3AttentionLayer::prefill_attention_paged_attn` (`prefill/paged_attn.rs:281`). HDIM=128, BF16 KV, non-FP8. **Exactly the kernel we need.**

For the bidirectional (non-causal) case inside the γ-block, the existing kernel passes `causal_mask_enabled = 1`. We have two choices:
- **(i) Easy:** add a `prefill_attention_paged_dflash` wrapper in `prefill_attn_main_a.rs` that passes `0` instead of `1`. The BF16 kernel binary needs to be re-checked: does the .cu source honor that flag? The FP8 sibling does (see `prefill_attention_paged_fp8_dflash` at `:259`, comment lines `:188–190` reference a "dflash" dispatcher). **15 LoC wrapper if the .cu honors the flag.**
- **(ii) Fallback if the BF16 kernel ignores the flag:** use the FP8 dflash kernel that already exists, and switch the drafter KV cache to FP8. Currently the cache is allocated as `KvCacheDtype::Fp8` at `from_weights.rs:74` — so this path is *already wired*, we just need to plumb the kernel through and use it. **RISKY** because FP8 KV on the drafter is known to collapse acceptance (`dflash_head.rs:82–86`). Strict preference: option (i).

**Reshape-and-cache:** `ops::reshape_and_cache` (BF16) at `crates/spark-model/src/layers/ops/kv_cache.rs:56`. Direct reuse.

**Slot mapping:** `ops::fill_slots_from_block_table` at `kv_cache.rs:20` builds the `[γ]` slot indices on-device from the block table. Direct reuse.

**Verdict: ZERO new CUDA. ZERO new Triton (we have none anyway). All Rust glue.**

---

## 5. New struct fields / allocations

| Field | Type | Size at L=5, γ=16, h=2048, kv_dim=512, ctx_max=512 |
|-------|------|----------------------------------------------------|
| `fused_kv_weight` | `DevicePtr` BF16 | 5 × 2 × kv_dim × h × 2 B = 5 × 2 × 512 × 2048 × 2 = **20 MB** |
| `fused_kv_bias` | `Option<DevicePtr>` | typically `None` (Qwen3); if present, 5 × 2 × 512 × 2 = 10 KB |
| `slot_mapping_dev` (scratch) | `DevicePtr` i32 | γ × 4 = **64 B** |
| `ctx_token_count` (state) | `usize` | 8 B per sequence |

**Paged KV cache for the drafter (already allocated, just unused today):**
- `from_weights.rs:80` allocates `PagedKvCache` sized for `max_seq_len + γ + 1` positions, block_size=16, L=5, num_kv_heads=4, head_dim=128.
- At max_seq_len=16384: `num_blocks ≈ 1025`, per-block size = `block_size × num_kv_heads × head_dim × 2 (K+V) × 2 B = 16 × 4 × 128 × 4 = 32 KB` × 5 layers = 160 KB/block.
- Total: **~165 MB per sequence at max_seq_len=16384**. (At FP8 it'd be ~83 MB; defer FP8 to a follow-up.)
- **This is already allocated today** — Option B doesn't pay it twice.

**Net new VRAM:** ~**20 MB** (fused_kv_weight). Trivial.

**Scratch buffers shrink:** every `n_attn = γ + ctx_window` allocation at `from_weights.rs:177–189` becomes `γ` (16). At ctx_window=512: `stream_buf` drops from 2.1 MB → 64 KB. `mlp_intermediate`: 6.3 MB → 192 KB. `logits`: 257 MB → 8 MB. **Net scratch savings: ~270 MB per head.** That's a fringe benefit that wasn't in the task spec but matters at higher batch.

---

## 6. Risk & sequencing

### Hardest part — **KV-cache slot mapping + position bookkeeping.**

Three pitfalls, in descending pain:

1. **RoPE position alignment across cache and γ-block — RISKY.**
   Context K rows are RoPE-rotated at positions `[position - ctx_count .. position)` when they enter the cache. The γ-block Q and K are rotated at `[position .. position + γ)`. The attention kernel computes `softmax(Q·Kᵀ)` over the concatenated key set; it has no concept of which key came from where. If the position arithmetic is off by one anywhere (ctx insert site, γ-block site, or the position argument passed to `precompute_ctx_kv`), every dot product gets a wrong rotation angle, the softmax collapses to garbage, and **acceptance silently drops to ~0**. No crash, no error log — just bad drafts.

   *Mitigation:* the existing `atlas_dflash_pyref.py` harness already produces per-layer K/V reference dumps. Phase 2's milestone is a bit-exact diff of `precompute_ctx_kv` output against that reference before we wire it in. That eliminates the position-bookkeeping bug class up front.

2. **Rejected-draft K/V cleanup after partial verify — MIDDLE.**
   When verify accepts `n < γ` drafts, the `γ - n` "wrong" K/V rows we wrote into the drafter cache must be invalidated before the next propose, or the next γ-block's attention will see ghost keys from a future that never happened.

   *Decision (locked):* **write all γ K/V before attention, free the unaccepted suffix in `after_verify`.** Reason: `prefill_attention_paged` reads K/V exclusively from the paged cache. The γ-block self-attends across its own K/V, so those rows must be in the cache before the attention call. The alternative — keeping γ K/V in a scratch tensor and writing only the accepted prefix later — would require either a separate scratch attention path (defeating the architectural fix) or a write-after-verify reordering that breaks per-step timing.

   *Implementation:* ~10–15 LoC in `propose.rs::after_verify` calling `PagedKvCache::free_blocks` for the `γ - n` trailing blocks. Verified against the same PyRef harness as #1.

3. **First-propose prefill — TRIVIAL.**
   On the first propose call, the ctx cache is empty and the drafter needs K/V for every token in the prompt prefix. vLLM's `precompute_and_store_context_kv` handles this by running the precompute path over the full prefix in one shot.

   Atlas already accumulates all prefill hidden states into `dstate.ctx_hidden_acc` (`dflash_head.rs:129–145`), so the data is sitting there. On first propose we call `precompute_ctx_kv` once over `dstate.ctx_len` tokens. No new state, no new bookkeeping. ~5 LoC at the propose entry point.

### Suggested implementation order (5 phases)

**Phase 1 — Land the BF16 paged-attn dispatcher (no behavior change).** 30 min.
- Add `ops::prefill_attention_paged_dflash` (BF16, causal=0) wrapper in `prefill_attn_main_a.rs`.
- Verify the .cu kernel reads the `causal_mask_enabled` flag (read `kernels/<hw>/common/inferspark_prefill_paged.cu`).
- If it doesn't, add the flag handling (CUDA edit, ~5 lines). One unit test.
- **Milestone:** `cargo test` green, no runtime behavior change yet.

**Phase 2 — Build fused_kv_weight + precompute_ctx_kv, but don't wire it in.** Half day.
- New file `precompute_ctx_kv.rs`. Builds fused weight at construction. Implements the precompute path. Can be called from a test harness that compares against PyTorch reference dumps.
- **Milestone:** dump K/V for ctx_count=1 from precompute_ctx_kv, compare bit-exact (or within BF16 rounding) against PyTorch. Use existing `atlas_dflash_pyref.py`.

**Phase 3 — Switch forward_block_layer to paged attention, γ-only rows.** Half day.
- Hoist the precompute, rip out the ctx K/V override in the layer body, change every `n_attn` to `γ`, swap `prefill_attention` for `prefill_attention_paged_dflash`.
- Keep ctx_token_count=0 path working as a sanity-check escape hatch (`ATLAS_DFLASH_DEBUG_CTX_OFF=1` already exists).
- **Milestone:** with ctx=0, output tokens identical to current Atlas (no behavior change). With ctx>0, acceptance ≥ current 77–81% K=2.

**Phase 4 — after_verify rollback.** 2 hours.
- Wire `PagedKvCache::free_blocks` for the unaccepted suffix.
- **Milestone:** run a 200-token generation, verify acceptance rate matches Phase 3 single-step measurements.

**Phase 5 — Perf measurement + tuning.** 2 hours.
- Run the canonical bench (whatever Ronald ran for the 9.7 tok/s number).
- Check: did we hit ≥15 tok/s? If not, profile launch overhead — maybe fuse k_norm across L layers (`qwen3_dflash.py:395–401` already loops, so fusing isn't urgent).
- **Milestone:** ≥15 tok/s, otherwise file follow-ups.

### Minimum testable correctness milestone
End of **Phase 3**, with `ctx_token_count=0`: should produce byte-identical drafts to current Atlas (we've only removed ctx, which the current code already supports via `ATLAS_DFLASH_DEBUG_CTX_OFF=1`). That gates moving forward.

### Estimated total
- **Code:** ~+250 LoC net.
- **Human-hours:** 16–24h for an engineer fluent in the codebase. Ronald is Rust-learning — add 50% buffer → **24–36h**. Two to three working days.

---

## 7. Expected perf math

**Current measured:** 165 ms propose, 9.7–11.6 tok/s end-to-end.

**Per-propose kernel launches:**
- Current (ctx_window=16, n_attn=32): 100+ launches × ~5 μs = ~500 μs launch overhead. Compute dominates.
- Option B (ctx=variable but most rows are γ=16):
  - Once-per-step precompute: ~13 launches (only grows with `new_ctx_count`).
  - Per-layer γ-only: ~13 launches × 5 layers = 65.
  - Post: ~20 (rms_norm + lm_head + γ argmax + D2H).
  - **Total: ~98 launches** — not dramatically fewer. **The win is compute, not launch count.**

**Per-propose compute** (rows × layer FLOPs):
- Current: 5 layers × (32 rows × ~2 GFLOP/row equiv) ≈ 10 GFLOP layer-body + 16 GFLOP lm_head ≈ **26 GFLOP**.
- Option B: 5 layers × (16 rows × ~2 GFLOP/row) ≈ 5 GFLOP layer-body + 16 GFLOP lm_head + 1 GFLOP precompute ≈ **22 GFLOP**.

That's only a 1.18× compute reduction — **NOT enough** to explain a 15-ms-class improvement. Hmm.

But wait: the empirical data showed `ctx_window=64 → 9.76 tok/s` and `ctx_window=16 → 11.62 tok/s`. The delta between those is **n_attn=80 → 32**, a 2.5× row drop. That delta produced only ~20% latency improvement. **Strong evidence the bottleneck is the lm_head GEMM** (which runs over γ=16 rows in BOTH cases — n_attn doesn't affect it) **and per-layer launch overhead**, not the row count on layer-body kernels.

If lm_head is the floor: it's 16 × 248320 × 2048 = 16 GFLOP. On GB10's ~30 TFLOPS BF16 dense throughput that's ~530 μs of math. There's no way that's 100ms by itself — kernel launch overhead must be the real floor.

**Realistic estimate**:
- If Option B drops 50% of layer-body launches (from 85 to 65): savings = 20 × 5 μs = 100 μs. **Negligible.**
- If Option B drops 50% of layer-body FLOPs: savings = (5 layer-body GFLOP) / (30 TFLOPS) ≈ 170 μs. **Negligible too.**
- The 165 ms current number means **something else is broken** — probably synchronization or D2H. The `gpu.synchronize(stream)` calls in the debug dump paths shouldn't be active in production, but it's worth verifying.

**RISKY** projection: Option B alone gets us to maybe **130–140 ms propose, 13–14 tok/s**. That's *better than 15.7 baseline IF* the verify step is also faster. If the goal is "≥15.7 tok/s with spec," Option B is necessary but **likely not sufficient**. We may also need to investigate the 165 ms floor (graph capture? CUDA-graph reuse? sync points?).

**Honest read:** Option B is the right rewrite — vLLM's architecture is correct and ours isn't — but if we ship it and STILL aren't beating baseline, the next investigation is `nsys profile` on the 165 ms call to find the actual stall.

---

## 8. Open questions for Ronald

1. **Drafter layer count.** Struct comment at `dflash_head.rs:6` says "8 layers" but task spec and `target_layer_ids = [1,10,19,28,37]` (5 entries) suggest 5. `num_hidden_layers` comes from drafter config.json. **Which is it for Qwen3.6-27B-DFlash?** All cost estimates above used 5; if it's 8 multiply layer-loop numbers by 1.6×.

2. **BF16 paged-attn kernel `causal_mask_enabled` flag honoring.** I confirmed the *Rust* dispatcher passes `1` (line `prefill_attn_main_a.rs:190`). Need to read `kernels/<hw>/common/inferspark_prefill_paged.cu` to confirm the kernel actually branches on it — and if not, add the conditional (5 lines of CUDA). Should I cite a specific file path? Not in the repo I scanned — please confirm the .cu file location.

3. **Where to land the 165 ms floor investigation.** If Option B lands and we're at 14 tok/s instead of 15.7, do we (a) merge anyway as a step forward, (b) hold the PR and chase the floor first, or (c) ship Option B with FP8 KV cache (more risk, more upside)? My vote: (a). The arch is wrong today, the arch is right after Option B, ship the arch fix and chase the floor as a follow-up.

---

## Appendix — Friday's one-line take

> "We're running 32 rows through 5 layers of MLP when vLLM runs 16. We allocated the paged KV cache eight months ago and forgot to plug it in. Three days of Rust glue, zero new CUDA, and we get the arch we said we'd build. Approve it."
