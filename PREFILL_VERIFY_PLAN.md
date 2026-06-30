# Prefill-mode Attention in DFlash Verify Path

**Goal**: Replace `decode_multi_seq(k)` with a single prefill-mode causal attention pass
for all K verify tokens, reducing `verify_multiplier` from ~12.44× to ~1.4-1.5×
and targeting ~37-50 tok/s on Qwen3.6-27B-NVFP4 + DFlash K=16.

**Reference**: SGLang `ForwardMode.TARGET_VERIFY` — one prefill pass over K tokens
with `q_offset = seq_len` so token `t` sees positions `0..seq_len+t+1` via causal mask.

---

## Kernel Chain Verification (read before implementing)

Полный путь для Qwen3.6-27B-NVFP4, K=16, подтверждён чтением кода:

```
layer.prefill(seq_len_start=seq_len, num_tokens=K)        [trait_impl.rs:90]
  → Qwen3AttentionLayer::prefill  ← OVERRIDE (не дефолтный fallback!)
  → prefill_inner(seq_len_start>0, batched_meta=None)     [prefill_inner.rs:117]
  → prefill_attention_paged(...)                           [paged.rs:131]
  → prefill_attention_paged_attn(...)                      [paged_attn.rs:429]
  → prefill_attention_paged_nvfp4(                         [prefill_attn_main_b.rs:18]
        q_len   = K                  -- число verify-токенов
        kv_len  = seq_len + K        -- полная длина KV (paged.rs:473)
        q_offset= seq_len            -- = seq_len_start as u32
        causal_mask_enabled = 1      -- hardcoded, строка 56
    )
```

**Важно**: дефолтный `TransformerLayer::prefill` (transformer_layer.rs:84) делает K
sequential decode вызовов (fallback для non-attention layers). `Qwen3AttentionLayer`
переопределяет этот метод (строка 90) и идёт через `prefill_inner`. Без override
мы получили бы то же что сейчас.

**KV write**: внутри `prefill_attention_paged` step 7 (paged.rs:416) происходит
запись K новых токенов в NVFP4 paged cache через `write_kv_cache` — до вызова
attention kernel. То есть порядок правильный: сначала K/V записывается, потом
attention читает `kv_len = seq_len + K` позиций.

**block_table**: kernel читает `meta.block_table` из `ctx.attn_metadata.block_table`
(paged_attn.rs:396). Это DevicePtr, который мы строим в scratch буфере в verify_d.rs.
Нужна одна строка (не K копий).

---

## Root Cause

`verify_d.rs` already does one forward pass, but attention is wrong:

```
decode_multi_seq(K=16) per attention layer
  → 16 separate paged-decode kernels
  → each reads full KV cache (seq_len tokens)
  → total: 16 × seq_len KV reads per attention layer
```

With prefill-mode:

```
prefill_attention_paged(q_len=K, kv_len=seq_len+K, q_offset=seq_len)
  → 1 kernel reads KV cache once for all K tokens
  → causal mask: token[t] sees 0..seq_len+t+1 automatically
```

No CUDA kernel changes needed. `prefill_attention_paged` already supports
this via `q_offset = seq_len` + `causal_mask_enabled = 1`.

---

## Key Files

| File | Role |
|------|------|
| `crates/spark-model/src/model/trait_impl/verify_d.rs` | Main change: metadata + call site |
| `crates/spark-model/src/layers/qwen3_attention/prefill/paged.rs` | `prefill_attention_paged` — no changes |
| `crates/spark-model/src/layers/qwen3_attention/trait_impl/prefill_inner.rs` | `prefill_inner` — no changes |
| `crates/spark-model/src/layer/transformer_layer.rs` | `TransformerLayer::prefill` trait — verify signature |

---

## Implementation Plan

### Iteration 1 — Eager Path (env-gated, no CUDA graphs)

Gate behind `ATLAS_VERIFY_PREFILL_ATTN=1` so the existing decode path stays intact
as fallback. Test with `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` for exact CUDA errors.

#### Step 1: Change `AttnMetadataDev` assembly in `verify_d.rs`

Current layout (lines 93–139): K-row decode format.

```
meta_base+0:    positions[K×4]         u32 — K positions
meta_base+256:  slots[K×8]             i64 — K KV-write slots
meta_base+512:  seq_lens[K×4]          i32 — K lengths (one per pseudo-seq)
meta_base+768:  block_table[K×max_blk×4]  — K identical rows
num_seqs = K
```

New prefill layout (same scratch offsets, different semantics):

```
meta_base+0:    positions[K×4]         u32 — K positions (unchanged)
meta_base+256:  slots[K×8]             i64 — K KV-write slots (unchanged)
meta_base+512:  seq_len[4]             u32 — ONE value: (seq_len + K) as u32
meta_base+768:  block_table[max_blk×4] u32 — ONE row (not K copies)
num_seqs = 1
```

`AttnMetadataDev` fields:
- `seq_len = meta_base.offset(512)` — points to single `(seq.seq_len + k) as u32`
- `block_table = meta_base.offset(768)` — single block table row
- `max_blocks_per_seq = seq.block_table.len() as u32`
- `num_seqs = 1`

#### Step 2: Replace attention call in the layer loop

In the `if layer_type == LayerType::FullAttention && !hss_engaged` branch,
replace `decode_multi_seq` with `layer.prefill`:

```rust
let use_prefill_attn = std::env::var("ATLAS_VERIFY_PREFILL_ATTN")
    .ok().as_deref() == Some("1");

if use_prefill_attn {
    layer.prefill(
        hidden,
        residual,
        k,                     // num_tokens = K verify tokens
        seq.layer_states[layer_idx].as_mut(),
        &mut kv_cache,
        seq.seq_len,           // seq_len_start = q_offset into KV cache
        &mut seq.block_table,
        &mut seq.disk_block_ids,
        &mut seq.disk_last_offloaded_per_layer,
        0,                     // kv_write_start = 0 (write all K slots)
        &ctx,
        stream,
    )?;
} else {
    // existing decode_multi_seq path
    let mut dummy_states = ...;
    layer.decode_multi_seq(...)?;
}
```

`ForwardContext` uses the prefill-format `attn_metadata` built in Step 1.
`prefill_inner` routes to `prefill_attention_paged` when `seq_len_start > 0`.

#### Step 3: Validate correctness

Run with both flags:
```bash
ATLAS_DFLASH_DEBUG_NO_GRAPH=1 ATLAS_VERIFY_PREFILL_ATTN=1 \
  ATLAS_DFLASH_DRAFT_CAP=15 spark serve ...
```

Check: acceptance rate ≥ what decode_multi_seq achieves. Any token mismatch
means metadata misalignment (positions, slots, or block_table layout).

---

### Iteration 2 — CUDA Graph Support

CUDA graph cache is keyed by `(seq.slot_idx, k)` (line 173 in verify_d.rs).
After changing the attention path, old cached graphs for the same key are wrong.

**Fix**: add a version byte to the key.

```rust
// Before
let cache_key = (seq.slot_idx, k);

// After
const VERIFY_ATTN_VERSION: u8 = 1; // bump when graph structure changes
let cache_key = (seq.slot_idx, k, VERIFY_ATTN_VERSION);
// HashMap type: HashMap<(usize, usize, u8), GraphHandle>
```

Also update `verify_kgamma_graph` field type in `TransformerModel`.

Remove the `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` flag and test graph capture/replay
with `ATLAS_VERIFY_PREFILL_ATTN=1` alone.

---

### Iteration 3 — Performance Measurement

Measure `verify_multiplier` before vs after:

```bash
# Before (decode_multi_seq)
ATLAS_DFLASH_DRAFT_CAP=15 spark serve ...
# send ~100 prompts, note tok/s and verify time from logs

# After (prefill-mode)
ATLAS_VERIFY_PREFILL_ATTN=1 ATLAS_DFLASH_DRAFT_CAP=15 spark serve ...
# same prompts
```

Expected results:

| Metric | Before | After |
|--------|--------|-------|
| verify_multiplier K=16 | ~12.44× | ~1.4–1.5× |
| verify time K=16 | ~957ms | ~110–130ms |
| throughput (accept_len≈6) | ~12.5 tok/s | ~37–50 tok/s |

If multiplier is still high, profile `ATLAS_DFLASH_PROF=1` to find the bottleneck.

---

### Iteration 4 — Edge Cases and Cleanup

1. **Remove env gate** once correctness and performance are confirmed. Make prefill
   the default path for non-HSS attention in `verify_d.rs`.

2. **MLA layers** (DeepSeek, Mistral-Small): `prefill_attention_paged_mla` is a
   separate kernel path. Add a guard:
   ```rust
   let has_mla = /* check model config */;
   if use_prefill_attn && !has_mla { ... }
   ```
   Fallback to `decode_multi_seq` for MLA until verified.

3. **seq_len == 0 guard**: physically impossible for verify (no context = no draft),
   but add assertion:
   ```rust
   debug_assert!(seq.seq_len > 0, "prefill-verify requires non-empty context");
   ```

4. **Buffer size check**: `BufferArena::qkv_output()` must hold
   `K × (n_q + 2×n_kv) × head_dim × 2` bytes. For K=16 this is far below
   normal prefill allocations, so no resize needed — but verify at init time.

---

## Risks

### R0 — Default `TransformerLayer::prefill` fallback (NOT a risk, but worth knowing)

**What**: `TransformerLayer::prefill` дефолтная реализация (transformer_layer.rs:84)
делает K sequential decode вызовов — это было бы хуже чем decode_multi_seq.

**Status**: НЕ риск. `Qwen3AttentionLayer` переопределяет `prefill` (trait_impl.rs:90)
и идёт через `prefill_inner`. SSM layers используют свой override (qwen3_ssm/trait_prefill.rs).
Дефолтный fallback используется только для слоёв без override — не наш случай.

---

### R1 — `prefill_inner` routes to wrong branch at `seq_len_start == 0`

**What**: `prefill_inner` takes the contiguous Flash Attention path (reads K/V from
`hidden`, not paged cache) when `seq_len_start == 0`. This silently reads garbage
for verify.

**Likelihood**: Low — verify with empty context is impossible in practice.

**Mitigation**: `debug_assert!(seq.seq_len > 0)` + manual test with seq_len=1.

---

### R2 — `AttnMetadataDev` layout mismatch

**What**: `prefill_attention_paged` reads metadata offsets by fixed convention
(`+0` = positions, `+256` = slots, `+512` = seq_len, `+768` = block_table).
If any offset diverges from what the kernel expects, attention output is garbage
with no error — just wrong tokens and collapsed acceptance rate.

**Likelihood**: Medium. The decode layout already uses these same offsets, but
with `num_seqs=K`. Switching to `num_seqs=1` changes how the kernel indexes them.

**Mitigation**: Test eager path first (`ATLAS_DFLASH_DEBUG_NO_GRAPH=1`) on a
short prompt with `CUDA_LAUNCH_BLOCKING=1` to get exact kernel errors. Compare
metadata dumps vs a regular prefill call for the same sequence.

---

### R3 — CUDA graph captures wrong metadata snapshot

**What**: Phase 1 (pre-graph) writes metadata to `meta_base`. If graph capture
begins before metadata upload completes, the captured graph has stale data.

**Likelihood**: Low — current code already has this pattern and it works for
decode format. Stream ordering ensures H2D copies complete before graph capture.

**Mitigation**: No change needed beyond the version byte on the cache key (Iter 2).

---

### R4 — MLA attention path breaks silently — ✅ CLOSED (pre-impl)

**What**: Models with MLA (e.g. DeepSeek) go through `prefill_attention_paged_mla`,
which has different metadata expectations. Calling `layer.prefill` with the
standard layout will produce wrong Q/K/V projections.

**Status**: Closed for Qwen3.6-27B-NVFP4. Confirmed by reading config.json:
`num_key_value_heads: 4`, no `kv_lora_rank` field, no MLA. Standard GQA.
For other models (DeepSeek etc.): guard still needed in Iteration 4.

---

### R5 — SSM state corruption on partial verify with new attention

**What**: SSM layers still use `decode_batched`. If the attention path produces
different hidden states than before (e.g. from a metadata bug), downstream SSM
layers accumulate incorrect `h_state`. The corruption is invisible until
acceptance collapses after a few hundred tokens.

**Likelihood**: Consequence of R2 — if metadata is right, this doesn't happen.

**Mitigation**: Run long-context benchmarks (>2K tokens) after fixing attention,
not just short prompts.

---

### R7 — `block_table` format: prefill kernel reads 1D, decode kernel reads 2D — ✅ CLOSED (pre-impl)

**What**: Decode kernel indexes block_table as `[seq_idx * max_blocks + block_idx]`
— K rows. Prefill kernel (`inferspark_prefill_paged_nvfp4`) indexes as
`block_table[block_idx]` — 1D, single row.

**Confirmed by reading `prefill_paged_compute.cuh` line 35:**
```c
unsigned int _lb = _pos / cache_block_size;   // block index
unsigned int _pb = (unsigned int)(bt)[_lb];   // bt[block_idx] — pure 1D
```
No `seq_idx * max_blocks` stride. Single flat array.

**Causal mask confirmed (lines 283-290):**
```c
unsigned int qr0 = q_offset + q_start + row0;  // absolute query position
if (kv_start + c0 > qr0) acc_s = -inf;         // mask future positions
```
With `q_offset = seq_len`: token `t` sees positions `0..seq_len+t`. ✓ Exact
verify semantics.

**Passing K identical rows**: kernel reads only row 0, result correct but wastes
scratch. Not a correctness risk but wastes ~K×max_blocks×4 bytes.

**Mitigation**: Upload 1 row (as planned in Step 1). Already confirmed safe
by scratch sizing (`bt_rows=32` in sizes.rs covers this case).

---

### R8 — `meta.seq_len` / `meta.num_seqs` not read by prefill kernel (non-risk, confirmed)

**What**: Plan originally said to change `num_seqs=K` to `num_seqs=1` and upload
a single `seq_len` value. In practice, `prefill_attention_paged` and
`prefill_attention_paged_attn` do NOT read `meta.seq_len` or `meta.num_seqs`.
The kernel receives `kv_len` directly as an argument computed in paged.rs line 473:
`kv_len = (seq_len_start + num_tokens) as u32`. The `num_seqs` field is not
passed to any kernel in the prefill path.

**Status**: Not a risk. But it means the scratch layout change for `seq_len`
(from K int32s to 1 uint32) and `num_seqs` field value don't affect kernel
correctness — only `positions[K]`, `slot[K]`, and `block_table[max_blocks]`
matter. Simplifies the implementation.

---

### R9 — SSM path unchanged, but `ctx.buffers` shared with new prefill path

**What**: After calling `layer.prefill()` for an attention layer, the buffers
`qkv_output`, `ssm_qkvz`, `ssm_deinterleaved`, `attn_output` hold intermediate
attention results. When the next SSM layer calls `decode_batched`, it reads from
`hidden` — not from these attention-specific buffers. `prefill_inner` writes the
final result back to `hidden` via residual add (prefill_inner.rs:409-417).

**Status**: Not a risk — each layer reads `hidden` on entry and writes `hidden`
on exit. Intermediate buffers are only live within a single layer's call.
Confirmed by tracing residual_add at end of prefill_inner.

---

### R6 — Throughput gain smaller than expected

**What**: `verify_multiplier` drops but not to ~1.4×. SSM `decode_batched` is
also O(K) sequential kernel launches and may dominate at low K or on GB10.

**Likelihood**: Medium. SGLang's ~1.4× was measured on A100 with a Mamba ratio
different from Qwen3.6-27B (28 attention + 8 SSM layers).

**Expected outcome**: Profile attention vs SSM separately via `ATLAS_DFLASH_PROF=1`
after Iteration 1. If SSM dominates, the next step is SSM parallelization
(fundamentally harder — requires breaking causal recurrence).

---

## Build Command (always)

```bash
ATLAS_TARGET_MODEL=qwen3.6-27b \
ATLAS_TARGET_QUANT=nvfp4 \
ATLAS_TARGET_HW=gb10 \
cargo build --release -p spark-server
```
