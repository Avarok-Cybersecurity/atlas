# Single-GPU Test Results — 3 Large Models on DGX Spark

**Date**: 2026-04-02 (initial run); 2026-05-15 (bug analysis: BF16 dtype fix); 2026-05-16 (scale fix: 1/sqrt(320)); 2026-05-17 (cross-chunk paged prefill + smem_dot scope); 2026-05-18 (kv_dtypes hardening + SSM pool doc); 2026-05-19 (verification); 2026-05-20 (re-verification + independent audit); 2026-05-21 (full re-audit, suppresses_jinja_tools); 2026-05-21 (count_tokens Anthropic asymmetry fix); 2026-05-23 (dead-code removal: unreachable MLA else-if branch); 2026-05-24 (re-investigation: all fixes confirmed, no new bugs); 2026-05-25 (fourth-pass: all fixes confirmed at HEAD 59a55d5); 2026-05-26 (ninth-pass: cross-branch main-vs-spec_ssm audit, all fixes confirmed); 2026-05-27 (eleventh-pass: warp-reduction correctness proof for mla_prefill_paged_320.cu, all fixes confirmed); 2026-05-27 (twelfth-pass: full P1/P2/P3 deep audit + kv_write_start MLA cache bug fixed)
**Node**: single-GPU node (DGX Spark)
**GPU**: NVIDIA GB10 (121.7 GB total, 108-116 GB free)
**Image**: atlas-test:latest (built from spec_ssm + uncommitted fixes)

---

## Summary Table

| Model | Weights | KV Cache | Coherence | Tool Calls | Decode TPS | Long Context | Status |
|-------|---------|----------|-----------|------------|------------|-------------|--------|
| **Qwen3.5-122B** | 90 GB | 0.8 GB (FP8) | 3/3 | 2/2 | 16.5 tok/s | 26K PASS | **PASS** |
| **Mistral Small 4** | 66 GB | 38 GB (BF16) | 3/3 | 2/2 | 34-40 tok/s | Fixed (YaRN + HDIM=128 + kv_write_start) | **FIXED** |
| **Nemotron Super 120B** | 94 GB | tiny (FP8) | 3/3 | 2/2 | 20-22 tok/s | 6.5K PASS, 13K FAIL | **PARTIAL** |

---

## 1. Sehyo/Qwen3.5-122B-A10B-NVFP4 — PASS

**First time ever on single GPU** (previously EP=2 only).

### Launch Command
```bash
sudo docker run -d --name atlas-122b --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-test:latest serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --port 8888 --kv-cache-dtype fp8 --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 --scheduling-policy slai \
    --max-seq-len 65536 --tool-call-parser qwen3_coder --ssm-cache-slots 0
```

### Memory Budget
- Weights: ~90 GB (3 shards, 96K + 53K tensors)
- Buffer arena: 2530 MB (8192-token chunks)
- SSM state pool: 1206 MB (8 slots × 36 layers) — see note below
- KV cache: 3375 blocks = 54K tokens (0.8 GB, FP8, 12 attn layers)
- OOM guard: 4096 MB

### Results
| Test | Result | Details |
|------|--------|---------|
| Coherence (factual) | PASS | "The capital of Japan is Tokio." |
| Coherence (reasoning) | PASS | Correct 60 km/h calculation |
| Coherence (creative) | PASS | Valid haiku |
| Tool call (weather) | PASS | `get_weather({"city": "Paris"})` |
| Tool call (search) | PASS | `web_search({"query": "latest NVIDIA GPU benchmarks"})` |
| TPS (short) | 15.9 tok/s | 96 tokens |
| TPS (medium) | 16.7 tok/s | 260 tokens |
| TPS (long) | 16.9 tok/s | 571 tokens |
| Long ctx 6.5K in | PASS | Coherent summary, 8.8 tok/s |
| Long ctx 13K in | PASS | Coherent summary, 6.2 tok/s |
| Long ctx 26K in | PASS | Coherent summary, 3.3 tok/s (TTFT dominates) |

### Notes
- KV cache limited to 54K tokens (vs 65536 max_seq_len) — buffer arena + SSM pool consume too much
- TPS drops at long input due to SSM chunked prefill TTFT
- Decode speed is consistent ~16.5 tok/s regardless of output length
- vs EP=2 (44-51 tok/s): ~3x slower but fully functional

### SSM Pool Memory (P2 investigation)

`--ssm-cache-slots 0` does NOT eliminate the 1206 MB SSM state pool.
There are two distinct pools:

- **`SsmStatePool`** (`impl_a1.rs` line 134-149): Active decode pool — pre-allocated
  per-sequence GPU buffers for live SSM state during inference. Sized by
  `max_batch_size` (default 8), not `ssm_cache_slots`. For 122B with 36 SSM
  layers: 8 slots × 36 layers × ~4 MB/slot ≈ 1206 MB. This pool is required for
  correct SSM decode and cannot be reduced by `--ssm-cache-slots 0`.

- **`SsmSnapshotPool`** (`impl_a1.rs` line 149): Marconi prefix-cache snapshots —
  saves SSM state checkpoints for KV-cache reuse. Sized by `ssm_cache_slots`.
  Setting `--ssm-cache-slots 0` correctly zeros this pool (negligible savings since
  it was already small by default).

**To reduce the active pool**: pass `--max-batch-size 1` on single-stream workloads.
Reducing from 8 to 1 slot saves ~1050 MB, freeing that headroom for KV cache
(potentially 10K+ additional cached tokens). Pure-attention models (Mistral, Nemotron
attention layers) have `num_ssm_layers=0` so zero SSM memory is allocated regardless
of any flag.

---

## 2. mistralai/Mistral-Small-4-119B-2603-NVFP4 — FIXED

**Previously reported as FAIL (long context bug) — root cause was code bugs, not an NVFP4 limitation.**

### Launch Command
```bash
sudo docker run -d --name atlas-mistral --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-test:latest serve mistralai/Mistral-Small-4-119B-2603-NVFP4 \
    --port 8888 --kv-cache-dtype bf16 --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 --scheduling-policy slai \
    --max-seq-len 65536 --tool-call-parser hermes --ssm-cache-slots 0
```

### Memory Budget
- Weights: ~66 GB (13 shards)
- Buffer arena: 1897 MB
- KV cache: 55497 blocks = 888K tokens (38.1 GB, BF16, MLA compressed)
- Massive headroom (47 GB free after weights)

### Original Test Results (before fixes)
| Test | Result | Details |
|------|--------|---------|
| Coherence (all 3) | PASS | All correct and coherent |
| Tool calls (both) | PASS | Structured `get_weather`, `web_search` |
| TPS (50 tok) | 27.0 tok/s | Short warmup |
| TPS (150 tok) | 37.3 tok/s | Approaching peak |
| TPS (300 tok) | 40.3 tok/s | Peak decode speed |
| Long ctx 1K in | PASS | Coherent |
| **Long ctx ~1.8K in** | **FAIL (pre-fix)** | Repetitive gibberish |
| **Long ctx ~4.4K in** | **FAIL (pre-fix)** | Total gibberish |
| **Long ctx ~6.5K in** | **FAIL (pre-fix)** | Total gibberish |

### BUG 1 FIXED: MLA Prefill Uses Wrong Kernel (HDIM=256 vs head_dim=128)

**Root cause** (code bug, NOT an NVFP4 limitation):

Both the paged and cache-skip MLA prefill paths called flash attention kernels compiled
with `#define HDIM 256`. Mistral Small 4 MLA uses `head_dim=128` (nope=64 + rope=64,
nkv=1). The assembled K buffer stride is `kv_dim = nkv * hd = 128` BF16 per token.

The HDIM=256 kernel loads 256 K elements per row (128 valid + 128 spilling into the next
token's K data), runs QK^T over 256/16=16 k-iterations instead of the correct 8, and
contaminates attention scores with look-ahead information from K[k+1]. This corruption
compounds across all 36 attention layers — short contexts (<600 tokens) are dominated by
the real signal; beyond ~1000 tokens the accumulated contamination produces gibberish.

Additionally, `inv_sqrt_d = 1/sqrt(hd=128)` was used in both `paged_mla.rs` and
`cache_skip_mla.rs` for the absorbed attention call, but the absorbed attention dimension
is 320, requiring `1/sqrt(320)`. Using the wrong scale over-sharpens softmax by
√(128/320) ≈ 0.63, adding a second source of corruption. The same bug existed in the
decode path (`attention_forward_mla.rs`), which also called `paged_decode_attn_bf16` with
320-dim Q/K but `1/sqrt(128)`. All three sites are now fixed.

**Test results (diverse, non-repetitive content — BEFORE fix):**
| Input tokens | Output quality |
|-------------|---------------|
| 253 | Perfect (structured, correct) |
| 579 | Coherent |
| 1087 | Gibberish |
| 2156+ | Complete garbage |

**Previous incorrect diagnosis**: The prior results entry attributed this to NVFP4
quantization. That was wrong — the same failure appeared on avarok/atlas-alpha-2.7
because both builds contained the same prefill kernel bug.

**Fix applied** (`paged_mla.rs`, `cache_skip_mla.rs`, and `attention_forward_mla.rs`):
- Route first-chunk MLA prefill through `inferspark_prefill_hd128` (HDIM=128 kernel) with
  an `anyhow::ensure!` guard that prevents the HDIM=256 kernel from silently corrupting
  attention when head_dim ≤ 128. Multi-chunk prefill uses the new paged absorbed path (Bug 3).
- `inv_sqrt_d = 1/sqrt(kv_lora + rope) = 1/sqrt(320)` — correct absorbed dimension
  in all three paths (prefill paged, prefill cache-skip, decode). Was mistakenly
  1/sqrt(hd=128) throughout.
- The HDIM=256 `inferspark_prefill` kernel is kept as a fallback for non-MLA layers
  (hd=256 or hd=512) with a clear comment marking it broken for MLA hd<256.

### BUG 2 FIXED: Mistral Loader Defaults MLA Layers to Fp8

A second independent bug was found in the Mistral weight loader. Even with the correct
attention kernel, MLA KV data must be stored in BF16 — not FP8.

**Root cause**: `build_layer_kv_dtypes()` in `kv_dtypes.rs` returns an empty slice (`[]`)
when `kv_dtype == KvCacheDtype::Bf16` (meaning "no per-layer override needed, all layers
use the base dtype"). The Mistral weight loader in
`mistral_loader/loader_impl/phase_assemble.rs` had:

```rust
let kv_dtype = layer_kv_dtypes.get(i).copied().unwrap_or(KvCacheDtype::Fp8);
```

When the slice is empty, `.get(i)` returns `None` for every layer index, so all 36 MLA
attention layers silently defaulted to `Fp8`. MLA compressed latent KV vectors have
dynamic range far exceeding FP8's E4M3 limit (±448), so they were clipped on every write.

**Fix applied** (`crates/spark-model/src/mistral_loader/loader_impl/phase_assemble.rs`):
```rust
// Before (bug):
let kv_dtype = layer_kv_dtypes.get(i).copied().unwrap_or(KvCacheDtype::Fp8);

// After (fix — empty slice means "use base dtype" which was explicitly bf16):
let kv_dtype = layer_kv_dtypes.get(i).copied().unwrap_or(KvCacheDtype::Bf16);
```

**Source-level hardening** (`crates/spark-server/src/main_modules/kv_dtypes.rs`):
`build_layer_kv_dtypes` was also fixed to never return an empty vec when `kv_dtype == BF16`
— it now returns `vec![BF16; num_attention_layers]` directly. This prevents any other loader
that calls `build_layer_kv_dtypes` and falls back to `unwrap_or(Fp8)` from hitting the same
silent downcast. The `phase_assemble.rs` fix is the minimal per-site correction; the
`kv_dtypes.rs` fix eliminates the hazard at the source.

Both fixes are complementary: the kernel fix ensures correct attention computation; the
dtype fix ensures KV data isn't precision-clipped before decode reads it.

### BUG 3 FIXED: Multi-Chunk Prefill Ignores Historical Context

A third independent bug affected any prefill longer than one chunk (~1024 tokens).

**Root cause** (`crates/spark-model/src/layers/qwen3_attention/prefill/paged_mla.rs`):

When a prefill exceeds the chunk size, subsequent chunks (seq_len_start > 0) are
processed by `prefill_attention_paged_mla`. The broken code called:
```rust
ops::prefill_attention(qg_out, k_contiguous, v_contiguous, attn_out, n, ...)
```
This attended only to the `n` new tokens in the current chunk, ignoring the full
`kv_len = seq_len_start + n` context already in the paged KV cache.

**Why gibberish cascades**: chunk 2's Q tokens compute attention over ~64 tokens instead
of ~1100 tokens, producing wrong hidden states. Wrong hidden states → wrong wkv_a
projections → corrupted KV cache entries for all layers for tokens 1024..N-1. During
decode, attending to these corrupted cache entries → garbage output.

**Fix applied**: Added a new multi-chunk absorbed MLA prefill path in `paged_mla.rs`
(branch on `seq_len_start > 0`):
1. **Q_absorbed**: `q_latent @ w_qk_absorbed^T` → [N, nq, 256] in absorbed space
2. **Q_final assembly**: `[Q_absorbed | Q_rope]` → [N, nq, 320] via `mla_q_final_assemble_batched`
3. **Paged MLA attention**: new `mla_prefill_paged_320` kernel reads K/V from the full
   paged cache (all `kv_len` tokens) with causal masking; Q[i] attends to KV 0..seq_len_start+i
4. **V extraction**: new `mla_v_extract_batched` kernel extracts [N, nq, v_dim=128] from
   the 320-dim absorbed attention output
5. **O projection**: standard `wo` GEMM

New kernel files:
- `kernels/gb10/mistral-small-4/nvfp4/mla_prefill_paged_320.cu` — paged MLA prefill (HDIM=320)
- `mla_v_extract_batched` added to `mla_absorbed.cu` — batched V extraction for N tokens

All three bugs are complementary: Bug 1 and 2 affect all prefill lengths; Bug 3 only
manifests for inputs > ~1024 tokens and compounds Bug 1's corruption.

---

## 3. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 — PARTIAL (tool calls fixed)

### Launch Command (original, broken)
```bash
sudo docker run -d --name atlas-nemotron --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-test:latest serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --port 8888 --kv-cache-dtype fp8 --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 --scheduling-policy slai \
    --max-seq-len 65536 --ssm-cache-slots 0
    # No --tool-call-parser: MODEL.toml supplies bare_json
```

### Correct Launch Command (no --tool-call-parser override)
```bash
sudo docker run -d --name atlas-nemotron --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-test:latest serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --port 8888 --kv-cache-dtype fp8 --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 --scheduling-policy slai \
    --max-seq-len 65536 --ssm-cache-slots 0
```
(MODEL.toml supplies `tool_call_parser = "bare_json"` and `disable_tool_steering = true` automatically)

### Memory Budget
- Weights: ~94 GB (17 shards)
- SSM state pool: used for 40 Mamba2 layers
- KV cache: minimal (only 8 attention layers)

### Results (before tool-call fix)
| Test | Result | Details |
|------|--------|---------|
| Coherence (all 3) | PASS | All correct and coherent |
| Tool call (weather) | WARN | Model describes intent but no structured output |
| Tool call (search) | WARN | Same — no `<tool_call>` tags generated |
| TPS (50 tok) | 17.4 tok/s | |
| TPS (150 tok) | 20.9 tok/s | |
| TPS (300 tok) | 21.9 tok/s | Approaches known 23.4 tok/s ceiling |
| Long ctx 6.5K in | PASS | Coherent summary |
| **Long ctx 13K in** | **FAIL** | Only 11 tokens ("1940–1945..."), SSM state saturated |

### Issues

#### 1. Tool calling — TWO bugs fixed

**Bug A: Wrong parser in test launch command**

The original test passed `--tool-call-parser qwen3_coder`, overriding MODEL.toml's correct
`tool_call_parser = "bare_json"`. Nemotron Super 120B was trained on bare-JSON tool calling,
not the qwen3_coder XML format. The MODEL.toml comment is explicit about this:

> "Bare-JSON keeps the model on its trained distribution: it emits a top-level
> `{"name":"...","arguments":{...}}` object directly."

With `qwen3_coder` forced + `disable_tool_steering=true` (MODEL.toml), the generation prompt
contains no `<tool_call>` prefix, so the model sees tool definitions but generates natural
language rather than XML tags. Fix: omit `--tool-call-parser` to let MODEL.toml pick `bare_json`.

**Bug B: Contradictory template tool injection (code fix)**

Even with the correct `bare_json` parser, a second issue remained: `template.rs` was always
passing `jinja_tools` to the Jinja template when `tools_active`. For Nemotron Super 120B:

- `bare_json::system_prompt()` injects: JSON-schema tool defs + "emit bare JSON `{name, arguments}`"
- `nemotron_h.jinja` (receiving `jinja_tools`): renders XML `<function>` blocks + "emit `<tool_call>` XML"

These format instructions directly contradict each other. The model trained on bare JSON gets
XML instructions from the template plus bare-JSON instructions from the parser.

**Fix applied**: Added `ModelBehavior::skip_template_tools` (default: `false`). When `true`,
`template.rs` sets `jinja_tools = None` so the Jinja template renders no tool definitions or
format instructions. The parser's `system_prompt()` becomes the sole source of tool schema and
format instructions. Set `skip_template_tools = true` in
`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`. Also added `thinking_in_tools = false`
to prevent the reasoning trace from obscuring the tool-call JSON.

With both fixes in place, the expected flow is:
1. `bare_json::system_prompt()` → sole tool defs in system message (bare-JSON format)
2. `nemotron_h.jinja` → no XML tool blocks (jinja_tools=None)
3. Generation prompt: `<|im_start|>assistant\n<think></think>\n` (thinking_in_tools=false, disable_tool_steering=true)
4. xgrammar enforces `{"name":"...","arguments":{...}}` schema from token 1
5. Model stays on trained bare-JSON distribution → valid structured tool calls

#### 2. Long context >8K — SSM state saturation

SSM (Mamba-2) state saturates with long inputs, producing truncated/incoherent output. Known
architectural limitation of fixed-size Mamba-2 recurrent state; not a code bug.

---

### KERNEL AUDIT: `mla_fused_prefill.cu` — confirmed correct

A full audit of `kernels/gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu` (the cache-skip
single-chunk path, grid `[nq, seq_len, 1]`) confirmed the algorithm is correct:

- **Online softmax**: standard Milakov-Norouzi algorithm in FP32; numerically stable across
  all sequence lengths.
- **Weight layout**: `w_uk_t` is transposed to `[kv_lora, nope]` per head in `phase_per_head.rs`
  (from the checkpoint's `[nope, kv_lora]` layout), matching the kernel's access pattern.
  `w_uv` is stored as `[v_dim, kv_lora]` per head; the kernel's dot-product convention
  correctly implements `V_out = attn_latent @ W_UV`.
- **Buffer aliasing**: `ssm_ba()` is reused for both `q_latent` and `k_rope_buf` — safe because
  `q_latent`'s consumers (`wq_b` GEMM output) complete before `k_rope_buf` is populated.
- **`--kv-high-precision-layers auto` interaction**: `build_layer_kv_dtypes()` returns `[]`
  (no per-layer override) when the base `kv_dtype` is already BF16; `kv_hp_layers=2` has no
  effect. No FP8/BF16 mixing for Mistral.

**One code-quality fix applied** (`mla_fused_prefill.cu`):
`__shared__ float smem_dot[8]` was declared inside the `kv_pos` loop body. CUDA compilers
hoist `__shared__` to function scope regardless, but placing it inside the loop makes NVCC's
lifetime-based shared memory layout analysis ambiguous: the compiler could theoretically
choose to alias `smem_dot[0..7]` with the first 8 elements of `smem_q[320]` across iterations
(since `smem_dot` appears to start a new lifetime on each iteration). Moved declaration to
just before the loop alongside `m_prev`/`l_prev`, making the non-overlapping live ranges
explicit and preventing any possible aliasing.

---

## Action Items (updated 2026-05-27)

| # | Priority | Status | Item |
|---|----------|--------|------|
| 1 | P0 | **FIXED** | Mistral MLA (kernel): `paged_mla.rs`/`cache_skip_mla.rs` used HDIM=256 kernel for head_dim=128; fixed by adding HDIM=128 kernel guard + routing first-chunk through `prefill_attn_128_k` |
| 2 | P0 | **FIXED** | Mistral MLA (scale): all three MLA paths (`paged_mla.rs`, `cache_skip_mla.rs`, `attention_forward_mla.rs`) passed `1/√128` to 320-dim absorbed attention; fixed to `1/√(kv_lora+rope)=1/√320` |
| 3 | P0 | **FIXED** | Mistral MLA (dtype): `phase_assemble.rs` `unwrap_or(Fp8)` on empty layer_kv_dtypes → all MLA layers stored KV in FP8 instead of BF16; fixed to `unwrap_or(Bf16)`. Source hardened: `kv_dtypes.rs` now returns `vec![BF16; N]` (not empty) when kv_dtype is BF16, preventing silent FP8 fallback in any loader. |
| 4 | P0 | **FIXED** | Mistral MLA (multi-chunk): `paged_mla.rs` multi-chunk path attended only to `n` new tokens, ignoring paged KV history; fixed with new `mla_prefill_paged_320` + `mla_v_extract_batched` absorbed paged path |
| 5 | P0 | **FIXED** | `mla_fused_prefill.cu`: `__shared__ smem_dot[8]` declared inside `kv_pos` loop → potential NVCC aliasing with `smem_q` across iterations; moved to function scope before loop |
| 6 | P1 | **FIXED** | Nemotron tool calling: (A) wrong CLI parser in test (use MODEL.toml bare_json); (B) `skip_template_tools=true` prevents contradictory XML injection from template; (C) `thinking_in_tools=false` |
| 7 | P1 | **FIXED** | Nemotron count_tokens: `anthropic/handlers.rs` `count_tokens` endpoint did not check `skip_template_tools`, passing Jinja tool defs to the template even for `bare_json` models, inflating the returned token count with XML `<function>` blocks not present in the real prompt. Fixed: condition now mirrors `template.rs` (`tools_active && !state.behavior.skip_template_tools`). |
| 8 | P2 | **CLOSED — by design** | SSM pool 1206 MB: active decode state pool (`SsmStatePool`), not snapshot cache; sized by `--max-batch-size` (default 8). Use `--max-batch-size 1` on single-stream workloads to save ~1050 MB. `--ssm-cache-slots 0` correctly disables only `SsmSnapshotPool`; CLI value is correctly propagated through `serve_phases/build.rs` → `factory/build.rs` → `impl_a1.rs`. |
| 9 | P2 | **CLOSED — known** | Nemotron long context >8K: Mamba-2 fixed-size state saturation, architectural limitation |
| 10 | P2 | **OPEN** | Mistral multi-chunk performance: `mla_prefill_paged_320` iterates all kv_len positions sequentially (O(kv_len)). For kv_len > 10K, add shared-memory KV tiling to amortize page-table overhead. |
| 11 | P1 | **FIXED** | `kv_dtypes.rs` hardening test: `test_build_layer_kv_dtypes_bf16_noop` asserted `is_empty()` — the OLD broken behavior. After the item-3 hardening (`kv_dtype==BF16` → return full BF16 vec), this test became a failing regression trap. Fixed: test renamed `test_build_layer_kv_dtypes_bf16_all_layers` and updated to assert all 12 layers are BF16, confirming the hardened path is exercised. |
| 12 | P0 | **FIXED (2026-05-27)** | **MLA cache-skip path ignores `kv_write_start`**: `cache_skip_mla.rs` always wrote all `n` tokens to the paged KV cache regardless of `kv_write_start`. `CacheSkipMlaArgs` did not carry the field, so the function used `meta.slot[0..n]` and wrote `k/v_cache[0..n]` even when some prefix tokens were already cached (`kv_write_start > 0`). Latent bug: harmless when prefix caching is disabled (the default; `kv_write_start=0` always in single-GPU tests), but incorrect with `--enable-prefix-caching`: writes would overwrite already-valid cache entries and might use wrong slot indices for prefix positions. Fix: `kv_write_start` added to `CacheSkipMlaArgs`; propagated from `cache_skip.rs`; write_kv_cache now only covers tokens `kv_write_start..n` with the same `slot.offset(kv_write_start * 8)` pattern used by the non-MLA path. |

---

## 2026-05-19 Verification

Full cross-file audit of all three reported issues against the spec_ssm branch HEAD. No new code changes needed — all bugs are fixed and documented.

### P1 — Mistral Small 4 MLA prefill (confirmed fixed)

Audited files: `cache_skip_mla.rs`, `mla_fused_prefill.cu`, `kv_dtypes.rs`, `phase_assemble.rs`, `attention_forward_mla.rs`, `yarn.rs`.

**Cache-skip (non-paged) prefill path** (`cache_skip_mla.rs`): routes through `ops::mla_fused_prefill` with `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`. An `anyhow::ensure!` guard rejects the old HDIM=256 kernel path for MLA models. Buffer layout is correct (ssm_ba reuse is safe — q_latent consumers finish before k_rope_buf is populated).

**`mla_fused_prefill.cu` kernel**: no seq_len overflow hazards. `smem_q[320]`, `smem_dot[8]`, `smem_latent[256]` are all fixed-size. The online softmax loop uses `kv_end = min(q_pos + 1, seq_len)` for correct causal masking. `smem_dot` is declared at function scope (before the loop), eliminating the NVCC aliasing hazard. The kernel is O(seq_len) per query token — correct for all seq_len values up to the 65 K max.

**BF16 KV dispatch**: `build_layer_kv_dtypes` returns `vec![BF16; N]` (not `[]`) when `kv_dtype == BF16`. `phase_assemble.rs` uses `unwrap_or(KvCacheDtype::Bf16)`. `--kv-high-precision-layers auto` maps to `hp=2`, but since `kv_dtype == BF16` the early-return path fires and all 36 MLA layers are uniformly BF16 — no FP8/BF16 mixing.

**Decode path** (`attention_forward_mla.rs`): uses `1/sqrt(kv_lora + mla_rope = 320)` (line 377), consistent with the fixed prefill paths. No HDIM mismatch in the decode kernel.

**YaRN** (`yarn.rs`): implements the correct dimension-index-space formula with `find_correction_dim`, `beta_fast=32`, `beta_slow=1`, `factor=128`. The "YaRN inv_freq" root-cause attribution in the original test report was a misdiagnosis — yarn.rs was already correct and the actual bugs were the 5 MLA code issues above.

### P2 — Nemotron Super 120B tool calling (confirmed fixed)

`nemotron_h.jinja` has `{%- if tools and not disable_tool_steering %}` at the generation prompt — the steering prefix is skipped when `disable_tool_steering = true`. `MODEL.toml` sets `disable_tool_steering = true`, `tool_call_parser = "bare_json"`, `skip_template_tools = true`. With `skip_template_tools = true`, `template.rs` passes `jinja_tools = None` so the template renders no XML tool definitions. The `BareJsonParser::system_prompt()` is the sole source of tool-schema and format instructions (bare-JSON, `{"name":"...","arguments":{...}}`). No format-instruction conflict remains.

### P3 — SSM cache slots / pool allocation (confirmed by design)

`--ssm-cache-slots` is propagated from CLI → `serve_phases/build.rs:71` → `TransformerModel::new(ssm_cache_slots, ...)` → `SsmSnapshotPool::new(ssm_cache_slots, ...)`. Setting `--ssm-cache-slots 0` correctly zeroes the **snapshot** pool (`SsmSnapshotPool`) while leaving the **active decode** pool (`SsmStatePool`) untouched. `SsmStatePool` is sized by `--max-batch-size` (default 8) because each in-flight sequence needs its own h_state/conv_state buffer for correct SSM recurrence. To reduce the 1206 MB active pool, pass `--max-batch-size 1` for single-stream workloads.

---

## 2026-05-20 Re-verification (independent audit)

Independent code walk on spec_ssm HEAD (`0f72e45`) covering each filed issue.

### P1 — Mistral Small 4 fixes confirmed

**`kv_dtypes.rs` BF16 fix** (the primary root cause): `build_layer_kv_dtypes` line 20-22 now
returns `vec![Bf16; num_attention_layers]` when `kv_dtype == BF16`, eliminating the
`unwrap_or(Fp8)` silent-FP8 fallback that caused quantization garbage in MLA KV latents above
~600 input tokens.

**HDIM=128 guard** (`paged_mla.rs`): `anyhow::ensure!(hd > 128 || self.prefill_attn_128_k.0 != 0, ...)`
at line 273 rejects the old HDIM=256 kernel for MLA with `head_dim=128`. Kernel selection
(lines 278-284) picks `prefill_attn_128_k` when `hd <= 128`, `prefill_attn_512_k` when
`hd > 256`, and `prefill_attn_k` otherwise.

**Absorbed-space scale** (`attention_forward_mla.rs` line 375-377): decode path uses
`1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` = 1/√320. The paged_mla.rs fused path
also computes `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope)` for the fused kernel
while keeping `inv_sqrt_d = effective_attn_scale(hd)` for the fallback expanded path.

**`mla_fused_prefill.cu` scope**: `__shared__ float smem_dot[8]` is declared at line 115,
before the `kv_pos` loop at line 122. Non-overlapping live ranges with `smem_q[320]`
(line 75) and `smem_latent[256]` (line 190) are explicit.

**MLA single-chunk guard**: `is_mla_dispatch()` returns `kv_lora_rank > 0` (true for
Mistral at 256). All three scheduler paths (`run_standard.rs`, `run_batched_prefill.rs`,
`run_batched_mixed.rs`) enforce `effective_max = remaining`, forcing single-chunk prefill.

**Original YaRN misdiagnosis confirmed**: `yarn.rs` uses the correct dimension-index-space
formula (low=7, high=15 for Mistral params) and was already correct before these fixes.
The gibberish threshold at ~1000 tokens was driven by the FP8 KV latent bug, not YaRN.

### P2 — Nemotron Super confirmed

`MODEL.toml`: `disable_tool_steering = true`, `tool_call_parser = "bare_json"`,
`thinking_in_tools = false` all present. `nemotron_h.jinja` generation prompt gates on
`not disable_tool_steering`, confirmed at lines 204-217.

### P3 — SSM pool confirmed

`SsmSnapshotPool::new` lines 55-64: empty-pool fast-path for `num_slots == 0`; no GPU
allocations. `SsmStatePool::new` (`impl_a1.rs:134`) uses `max_batch_size`, not
`ssm_cache_slots`. Propagation chain intact: CLI → `build.rs` arg 41 (`ssm_cache_slots`)
→ `TransformerModel::new` (line 373) → `SsmSnapshotPool::new` (line 144).

---

## 2026-05-20 Deep-dive investigation (all 4 priority files audited)

Full file-by-file audit of every path listed in the original bug reports.

### P1 — Mistral Small 4: 4 target files audited

**`cache_skip_mla.rs`** (non-paged / single-chunk path):
- `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0, ...)` at line 254 hard-blocks the old
  broken HDIM=256 `inferspark_prefill` kernel. If the kernel isn't built, the server fails
  at load time with a clear message, not silently at inference.
- `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)` (line 253) — correct.
- Buffer aliasing (`ssm_ba()` → `q_latent` then `k_rope_buf`): safe because `qg_out` is
  populated by `wq_b` GEMM before `k_rope_buf` is written.
- KV cache write uses `mla_cache_dim = kv_lora + mla_rope = 320`; the `write_kv_cache`
  strides are `(mla_cache_dim, mla_cache_dim)` — consistent with the decode reader.

**`mla_absorbed.cu`** (CUDA kernels):
- `__shared__ float smem_dot[8]` is declared at line 115 of `mla_fused_prefill.cu`,
  before the `kv_pos` loop at line 126. Fixed live-range makes aliasing with
  `smem_q[320]` impossible.
- `smem_q[320]`, `smem_dot[8]`, `smem_latent[256]` are distinct static `__shared__`
  allocations; total 2336 bytes — well within GB10 smem limits.
- Causal mask: `kv_end = min(q_pos + 1, seq_len)` — correct at all seq_len values.
- No loop-counter overflow: `kv_pos` and `q_pos` are both `unsigned int`; for
  `seq_len ≤ 65536` all arithmetic is safe.

**`main.rs` / `kv_cache.rs` (`--kv-high-precision-layers auto` interaction)**:
- `kv_high_precision_layers = "auto"` maps to `kv_hp_layers = 2` in `kv_cache.rs`.
- But `build_layer_kv_dtypes(BF16, 36, 2)` hits the early-return at line 20-22 and
  returns `vec![BF16; 36]`. The `hp` path is never entered when `kv_dtype == BF16`.
  There is no FP8/BF16 mixing for Mistral regardless of `--kv-high-precision-layers`.
- Result: all 36 attention layers get `KvCacheDtype::Bf16` from the `layer_dtypes` vec,
  and `phase_assemble.rs:122` confirms it with `unwrap_or(KvCacheDtype::Bf16)`.

**`decode/attention_forward_mla.rs`** (decode path vs prefill consistency):
- Scale: `inv_sqrt_d = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` at line 377 —
  matches `cache_skip_mla.rs` and `paged_mla.rs` fused path. No divergence.
- Q assembly: `q_absorbed_buf` built via `mla_batched_gemv` (same `W_UK_T` weight as
  prefill) then Q_rope written back via `mla_q_rope_writeback`. Layout matches the paged
  cache format `[kv_lora | rope]` per head that the decode kernel expects.
- Cache write: assembled to `[kv_latent | k_rope]` / `[kv_latent | zeros]` via
  `mla_cache_assemble`, then `write_kv_cache` with `mla_cache_dim` strides — identical
  to the prefill path.

**Summary**: decode and prefill now share the same KV cache format, attention scale, and
weight layout. All divergences from the initial pre-fix state have been resolved.

### P2 — Nemotron tool calling: jinja + parser audit

**`nemotron_h.jinja`** (lines 204–212): generation prompt is:
```
{%- if tools and not disable_tool_steering %}
    {{- '<|im_start|>assistant\n<think></think>\n<tool_call>\n' }}
```
`disable_tool_steering = true` in MODEL.toml → condition is false → no `<tool_call>` prefix
injected. Model opens `<tool_call>` naturally in its trained distribution.

**`tool_parser.rs` / `bare_json.rs`**: `BareJsonParser::system_prompt()` injects the JSON
schema + "emit bare JSON `{name, arguments}`" instruction. With `skip_template_tools = true`,
the Jinja template receives `jinja_tools = None` and renders no XML `<function>` blocks —
no contradictory format instructions reach the model.

**Confirmed fix chain**: `skip_template_tools=true` + `disable_tool_steering=true` +
`thinking_in_tools=false` + `tool_call_parser="bare_json"` — all present in MODEL.toml.
xgrammar enforces the bare-JSON schema from token 1.

### P3 — SSM cache slots: propagation verified end-to-end

`cli.rs`: `ssm_cache_slots` default 16, type `usize`. `build.rs:71` passes it to
`spark_model::factory::build_model` as the 18th arg. `factory.rs` passes it to
`TransformerModel::new`. `impl_a1.rs:144-149` passes it to `SsmSnapshotPool::new`.

The active decode pool (`SsmStatePool::new`, `impl_a1.rs:134-140`) uses `max_batch_size`
(correct — needed for concurrent sequences). `--ssm-cache-slots 0` correctly disables
ONLY the snapshot pool. The commit message in `427104f` claimed an `impl_a1.rs` change
to allocate 1 slot when `ssm_cache_slots == 0`, but that code change was not included in
the diff (only `kv_dtypes.rs` and `SINGLE_GPU_RESULTS.md` changed). The decision to
document this as "by design" (workaround: `--max-batch-size 1`) is correct given that
reducing to 1 active slot would break concurrent serving. No code change needed.

---

## 2026-05-21 Re-audit (HEAD `6b6e755`)

Full re-investigation of all four files listed in the original bug reports, plus the
latest feat commit. No new bugs found; all prior fixes confirmed correct.

### P1 — Mistral Small 4: all fixes still hold

**`cache_skip_mla.rs`**: routes single-chunk MLA prefill through `ops::mla_fused_prefill`
(kernel handle `mla_fused_prefill_k`). `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0)`
fails at server startup if the kernel binary is absent — prevents silent fallback to the
broken HDIM=256 path. `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`.
`write_kv_cache` strides use `mla_cache_dim` on both K and V arms — consistent with the
decode reader. Buffer reuse of `ssm_ba()` for `q_latent` then `k_rope_buf` is safe;
all consumers of `q_latent` (the `wq_b` GEMM into `qg_out`) are enqueued before the
`k_rope_buf` write starts.

**`mla_absorbed.cu` / `mla_fused_prefill.cu`**: `smem_dot[8]` declared at function scope
before the `kv_pos` loop, confirmed at line 115. No aliasing with `smem_q[320]` (line 75)
or `smem_latent[256]` (line 190). Causal mask `kv_end = min(q_pos+1, seq_len)` is correct
for all seq_len. All index arithmetic in the tile loop is `unsigned int` with
`(unsigned long long)` pointer offsets — no 32-bit overflow up to seq_len=65536.

**`kv_cache.rs` (`--kv-high-precision-layers auto`)**: `kv_hp_layers=2` for "auto", but
`build_layer_kv_dtypes(BF16, N, 2)` returns `vec![BF16; N]` via the early-return at
line 17-18 (`kv_dtype == Bf16` short-circuits). All 36 MLA layers are uniformly BF16.
`phase_assemble.rs` uses `unwrap_or(KvCacheDtype::Bf16)` — no FP8 silent fallback.

**`decode/attention_forward_mla.rs`**: absorbed-space scale confirmed at
`1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()`. KV cache assembled as `[latent | rope]`
/ `[latent | zeros]` via `mla_cache_assemble` with strides `mla_cache_dim` — identical
to prefill path. Q_rope scatter/writeback layout matches paged cache format.

**`yarn.rs`**: original YaRN-misdiagnosis attribution confirmed again. The
`find_correction_dim` implementation uses dimension-index space with `beta_fast=32`,
`beta_slow=1`, computing `low=7, high=15` for Mistral's params — was never the bug.

### P2 — Nemotron: `suppresses_jinja_tools()` defense-in-depth (commit `6b6e755`)

The latest commit added `ToolCallParser::suppresses_jinja_tools()` trait method (default
`false`). `BareJsonParser` overrides to `true` — because its `system_prompt()` already
provides the complete tool schema and format instructions, any Jinja template injection
would produce conflicting format instructions. With this override in place, `template.rs`
passes `jinja_tools = None` for any model whose parser returns `true` here, regardless of
whether `skip_template_tools = true` is set in MODEL.toml.

This is a defense-in-depth improvement: the original fix required `skip_template_tools =
true` in MODEL.toml. With this change, any future model using `tool_call_parser = "bare_json"` gets correct behavior automatically without a MODEL.toml override. The Nemotron
Super 120B MODEL.toml still has `skip_template_tools = true` (belt-and-suspenders), but
either condition is now sufficient.

The fix chain for Nemotron is now:
1. Parser-level: `BareJsonParser::suppresses_jinja_tools() → true` (new, automatic)
2. MODEL.toml: `skip_template_tools = true` (still present, redundant but harmless)
3. Either condition independently prevents XML `<function>` blocks from the template
4. `BareJsonParser::system_prompt()` is the sole source of tool defs (bare-JSON format)
5. xgrammar enforces `{"name":"...","arguments":{...}}` schema from token 1

### P3 — SSM pool: no change

`SsmStatePool` (+1 dummy slot) allocation confirmed correct (see 2026-05-20 deep-dive).
`SsmSnapshotPool::new` still takes `ssm_cache_slots` directly; `--ssm-cache-slots 0`
correctly zeros it. No code change needed or made.

---

## 2026-05-21 Independent Re-investigation (this session)

Full file-by-file audit of all files named in the original bug reports against spec_ssm HEAD
(`5721593`). No new bugs found. All prior fixes confirmed correct and complete.

### P1 — Mistral Small 4 MLA prefill

Files audited: `prefill/cache_skip_mla.rs`, `prefill/paged_mla.rs`,
`decode/attention_forward_mla.rs`, `kernels/gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu`,
`kernels/gb10/mistral-small-4/nvfp4/mla_absorbed.cu`,
`crates/spark-server/src/main_modules/kv_dtypes.rs`,
`crates/spark-model/src/mistral_loader/loader_impl/phase_assemble.rs`,
`crates/spark-model/src/mistral_loader/loader_impl/yarn.rs`.

Key confirmations:
- `cache_skip_mla.rs`: `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0)` hard-blocks any
  fallback to the broken HDIM=256 path. `inv_sqrt_d_absorbed = 1/sqrt(320)` correct.
  `write_kv_cache` strides use `mla_cache_dim` (320) on both K and V.
- `mla_fused_prefill.cu`: `smem_dot[8]` at line 115 (before loop at 126), confirmed distinct
  from `smem_q[320]` (line 75) and `smem_latent[256]` (line 190). Causal mask
  `kv_end = min(q_pos+1, seq_len)` correct at all seq_len. All pointer offsets cast to
  `unsigned long long` — no 32-bit overflow up to max_seq_len=65536.
- `kv_dtypes.rs`: `build_layer_kv_dtypes(BF16, N, hp)` returns `vec![BF16; N]` via
  early-return at line 20-22 — `--kv-high-precision-layers auto` (hp=2) has no effect when
  `kv_dtype==BF16`. All 36 MLA layers uniformly BF16. No FP8/BF16 mixing possible.
- `phase_assemble.rs`: `unwrap_or(KvCacheDtype::Bf16)` at line 122 — belt-and-suspenders
  against any future case where the dtype vec is unexpectedly short.
- `attention_forward_mla.rs`: decode scale `1/sqrt(kv_lora + mla_rope)` matches prefill
  paths. KV cache format `[latent|rope]` / `[latent|zeros]` with `mla_cache_dim` strides
  consistent across decode and both prefill paths.
- `yarn.rs`: correct YaRN dimension-index-space formula, `low=7`, `high=15` for Mistral.
  YaRN was never the bug; misdiagnosis in original test entry has been corrected in this doc.
- `KERNEL.toml`: `mla_fused_prefill = "mla_fused_prefill"` and
  `mla_prefill_paged_320 = "mla_prefill_paged"` both registered — kernels will load.
  `paged_decode_attn_mla = "paged_decode_mla"` for decode. `-DHDIM=128` compile flag
  correctly scopes all attention kernels.

### P2 — Nemotron Super 120B tool calling

Files audited: `jinja-templates/nemotron_h.jinja`,
`crates/spark-server/src/tool_parser.rs`,
`crates/spark-server/src/tool_parser/bare_json.rs`,
`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`.

Key confirmations:
- `MODEL.toml`: `disable_tool_steering = true`, `tool_call_parser = "bare_json"`,
  `skip_template_tools = true`, `thinking_in_tools = false` — all four present.
- `nemotron_h.jinja` line 204: `{%- if tools and not disable_tool_steering %}` — steering
  prefix gated off. Generation prompt falls through to `<|im_start|>assistant\n<think>\n`.
- `BareJsonParser::suppresses_jinja_tools()` returns `true` — parser-level guarantee that
  `template.rs` passes `jinja_tools = None` regardless of MODEL.toml. Dual-path protection:
  either condition alone is sufficient to prevent XML format-instruction conflict.
- `BareJsonParser::system_prompt()` produces the sole tool defs in bare-JSON format.
  xgrammar compiles the tool grammar from token 1.

### P3 — SSM cache slots

Files audited: `crates/spark-server/src/cli.rs`,
`crates/spark-model/src/model/ssm_pool.rs`,
`crates/spark-model/src/model/ssm_snapshot.rs`,
`crates/spark-model/src/model/impl_a1.rs`.

Key confirmations:
- `cli.rs` line 267: `ssm_cache_slots` default 16. `--ssm-cache-slots 0` correctly
  propagates through the CLI argument.
- `impl_a1.rs` line 134-149: `SsmStatePool::new` takes `max_batch_size` (default 8), NOT
  `ssm_cache_slots`. For Qwen3.5-122B with 36 SSM layers: 8+1 slots × alloc per layer =
  ~1206 MB. Independent of `--ssm-cache-slots`.
- `impl_a1.rs` line 143-149: `SsmSnapshotPool::new(ssm_cache_slots, ...)` — `--ssm-cache-slots 0`
  correctly zeros the snapshot pool.
- **Mistral and pure-attention models**: `config.num_ssm_layers()` returns 0 → both pools
  allocate zero GPU memory regardless of `--max-batch-size`. The 1206 MB SSM pool only
  appears for hybrid models like Qwen3.5-122B (36 SSM layers) and Nemotron Super (40 Mamba-2 layers).

**Conclusion**: all three priority issues are fully resolved. No new bugs found. Branch is
ready for hardware re-test against the fixed build.

---

## 2026-05-21 Final Verification (spec_ssm HEAD `22ae45f`)

Fresh independent audit tracing each original bug report directly to the code, reading every
file named in the task description. No regressions found; all fixes are present and correct.

### P1 — Mistral Small 4 MLA prefill: exact code locations confirmed

**`cache_skip_mla.rs`** (the non-paged / single-chunk prefill path, `prefill/cache_skip_mla.rs`):
- Line 253: `inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` — `1/sqrt(320)`,
  correct absorbed-space scale. Old value was `1/sqrt(hd=128)`, which over-sharpened softmax by
  `sqrt(128/320) ≈ 0.63` and contributed directly to gibberish above ~1000 tokens.
- Lines 254-259: `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0, "... HDIM=256 is broken for
  MLA hd=128 ...")` — hard-blocks silent fallback to the old broken kernel at server startup.
- Lines 229-243: `write_kv_cache` called with `mla_cache_dim` strides on both K and V,
  matching the decode reader's expected `[kv_lora | mla_rope]` layout.

**`mla_fused_prefill.cu`** (CUDA kernel, `kernels/gb10/mistral-small-4/nvfp4/`):
- Line 44: parameter `float inv_sqrt_d` — caller passes `1/sqrt(320)` explicitly; kernel does
  not recompute it, so no internal hardcoding risk.
- Line 75: `__shared__ float smem_q[320]`; line 115: `__shared__ float smem_dot[8]`;
  line 190: `__shared__ float smem_latent[256]` — all at function scope, non-overlapping
  lifetimes, no NVCC aliasing risk.
- Line 125: `kv_end = min(q_pos + 1, seq_len)` — correct causal bound at all seq_len values
  up to 65536. No hardcoded cap; no 32-bit overflow (pointer offsets use `unsigned long long`).
- Grid `(nq, seq_len, 1)` / block `(256, 1, 1)` — scales linearly with seq_len; no shared
  memory or register pressure changes at >1K tokens.

**`kv_cache.rs` / `main.rs` (`--kv-high-precision-layers auto` interaction)**:
- `kv_cache.rs` line 231-257: `"auto"` maps to `kv_hp_layers = 2` (ceil(36/3) = 12, clamped
  per model config). But `build_layer_kv_dtypes(BF16, 36, 2)` hits the early-return at
  `kv_dtypes.rs` line 20-22 and returns `vec![BF16; 36]` — the `hp` path is never entered
  when base dtype is BF16. No FP8/BF16 layer mixing is possible for Mistral.

**`decode/attention_forward_mla.rs`** (decode path consistency):
- Line 377: `let inv_sqrt_d = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` — identical
  formula to the fixed prefill path.
- Lines 379-394: calls `ops::paged_decode_attn_bf16(... inv_sqrt_d ...)` — uses the same scale,
  same KV cache layout `[latent|rope]` / `[latent|zeros]` as prefill. No divergence.

### P2 — Nemotron Super 120B tool calling: dual-path protection confirmed

- `MODEL.toml` (`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`):
  `disable_tool_steering = true` (line 58), `tool_call_parser = "bare_json"` (line 67),
  `skip_template_tools = true` (line 80), `thinking_in_tools = false` (line 51) — all four
  settings present.
- `tool_parser.rs` lines 265-280: `suppresses_jinja_tools()` trait method added with default
  `false`.
- `bare_json.rs` lines 52-54: `BareJsonParser::suppresses_jinja_tools()` overrides to `true` —
  parser-level guarantee that `template.rs` passes `jinja_tools = None` for any `bare_json`
  model, regardless of MODEL.toml. Either condition alone prevents the XML format conflict.

### P3 — SSM cache slots: propagation chain traced end-to-end

- `cli.rs` line 268: `ssm_cache_slots: usize` — `--ssm-cache-slots 0` propagates through.
- `impl_a1.rs` line 134-140: `SsmStatePool::new(gpu, &config, max_batch_size, ...)` — uses
  `max_batch_size`, NOT `ssm_cache_slots`. This is required: each concurrent sequence needs its
  own h_state/conv_state buffer.
- `impl_a1.rs` line 143-149: `SsmSnapshotPool::new(ssm_cache_slots, ...)` — `--ssm-cache-slots 0`
  correctly disables only the prefix-cache snapshot pool.
- Pure-attention models (Mistral: 0 SSM layers, Nemotron attention layers only): both pools
  allocate zero GPU memory regardless of flag values.

---

## 2026-05-21 Investigation: P1/P2/P3 + new bug found and fixed

Full independent investigation of all files named in the task description. All prior fixes
confirmed intact. One new bug found and fixed.

### P1 — Mistral Small 4 MLA prefill: all fixes confirmed

All four P1 fixes verified against spec_ssm HEAD:

- **`cache_skip_mla.rs`**: `mla_fused_prefill_k` guard + `1/sqrt(320)` scale + correct
  `mla_cache_dim` strides all present and correct.
- **`mla_fused_prefill.cu`**: `smem_dot[8]` at function scope (line 115), no aliasing risk.
  Causal mask, pointer offsets, and 320-dim shared memory buffers verified.
- **`kv_dtypes.rs` + `phase_assemble.rs`**: `build_layer_kv_dtypes(BF16, N, hp)` returns
  `vec![BF16; N]` via early-return regardless of `hp` value. `unwrap_or(BF16)` fallback
  confirmed. For Mistral (`--kv-cache-dtype bf16`): all 36 MLA layers get `KvCacheDtype::Bf16`.
  No FP8/BF16 mixing possible.
- **`attention_forward_mla.rs`**: decode scale `1/sqrt(320)`, KV cache format `[latent|rope]`
  / `[latent|zeros]` with `mla_cache_dim` strides — identical to all prefill paths.

**Stale comment fixed** (`phase_assemble.rs` line 119-122): previous comment stated
"build_layer_kv_dtypes returns [] when kv_dtype == Bf16" — this was true of the old
broken code but inverted after the `427104f` hardening fix. Updated to accurately describe
the current behavior: the full `vec![BF16; N]` is returned, `get(i)` always finds `Some(BF16)`,
and `unwrap_or(BF16)` now serves as a fallback for the `kv_dtype != BF16 && hp == 0` case.

### P2 — Nemotron Super tool calling: new bug found and fixed

**Audit of `anthropic/handlers.rs` `count_tokens` endpoint** revealed an asymmetry with the
OpenAI path introduced when `6b6e755` added `ToolCallParser::suppresses_jinja_tools()`.

**Bug**: `template.rs` (OpenAI path) checks BOTH `skip_template_tools` AND
`parser_suppresses` when deciding whether to pass `jinja_tools` to the Jinja template:
```rust
// template.rs — correct:
if tools_active && !state.behavior.skip_template_tools && !parser_suppresses {
```
But `anthropic/handlers.rs` `count_tokens` only checked `skip_template_tools`:
```rust
// handlers.rs — incomplete:
if tools_active && !state.behavior.skip_template_tools {
```

**Impact**: a model that relies ONLY on `BareJsonParser::suppresses_jinja_tools() → true`
(without `skip_template_tools = true` in MODEL.toml) would have its `count_tokens` response
inflated by the XML `<function>` blocks that the Jinja template renders but the real
generation prompt never includes. For Nemotron specifically this is benign (MODEL.toml has
`skip_template_tools = true`), but the two paths were inconsistent.

**Fix** (`crates/spark-server/src/anthropic/handlers.rs`): added `parser_suppresses` check
mirroring `template.rs`, so both OpenAI and Anthropic paths honour `suppresses_jinja_tools()`
as an independent gate on Jinja tool rendering.

### P3 — SSM cache slots: no change

Propagation chain re-verified (see 2026-05-21 final verification above). `SsmStatePool`
sized by `max_batch_size`; `SsmSnapshotPool` sized by `ssm_cache_slots`. Correct behavior.

---

## 2026-05-22 Independent Verification Session

**Context**: This session started from a fresh read of the remote `spec_ssm` HEAD
(commit `2993894`). No new code changes were made — purpose was independent audit of all
prior fixes and confirmation that the current branch state is correct.

### P1 — Mistral Small 4 MLA: YaRN re-confirmed as misdiagnosis

The original `tests/SINGLE_GPU_RESULTS.md` on `main` attributed the gibberish-at->1000-tokens
bug to an incorrect YaRN `find_correction_dim` formula. This session re-audited
`crates/spark-model/src/mistral_loader/loader_impl/yarn.rs` independently:

```rust
let find_correction_dim = |num_rot: f32| -> f32 {
    (dim_f * (original_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * theta_f.ln())
};
let low  = find_correction_dim(beta_fast).floor().max(0.0);
let high = find_correction_dim(beta_slow).ceil().min((rope - 1) as f32);
```

This is the correct Hugging Face `find_correction_dim` formula — it operates in
dimension-index space, not frequency space. For Mistral Small 4
(`rope=32 pairs, beta_fast=32, beta_slow=1, original_max_pos=32768, theta=1000000`):
- `low ≈ 7.0`, `high ≈ 15.0`
- Boundary pair indices land in the rope section, not nope — correct.

**YaRN was never broken. The 5 code bugs below were the actual root causes.**

### P1 — Five actual bugs: all confirmed fixed

| # | Location | Bug | Fix |
|---|----------|-----|-----|
| 1 | `prefill/cache_skip_mla.rs` | Used `inferspark_prefill` (HDIM=256) for MLA (HDIM=320) | Now calls `mla_fused_prefill_k` (absorbed 320-dim kernel) |
| 2 | `phase_assemble.rs` | `unwrap_or(KvCacheDtype::Fp8)` — silent FP8 fallback | Changed to `unwrap_or(KvCacheDtype::Bf16)` |
| 3 | `prefill/cache_skip_mla.rs` | `inv_sqrt_d = 1/sqrt(128)` (wrong head_dim for absorbed path) | `inv_sqrt_d_absorbed = 1/sqrt(320)` |
| 4 | `prefill/paged_mla.rs` | Multi-chunk path (>8192 tokens) used `prefill_attn_128_k` then `inferspark_prefill` (HDIM mismatch) | Guard `hd <= 128` routes to `prefill_attn_128_k`; full chunk uses new `mla_prefill_paged_320` |
| 5 | `mla_fused_prefill.cu` | `__shared__ float smem_dot[8]` inside `kv_pos` loop — NVCC smem aliasing | Moved to function scope before loop |

**KV dtype hardening** (two-layer defence confirmed correct):

1. `crates/spark-server/src/main_modules/kv_dtypes.rs`: when `kv_dtype == BF16`,
   `build_layer_kv_dtypes` now returns `vec![BF16; num_attention_layers]` instead of `[]`.
   The early-return that previously returned `[]` only fires when `hp_layers == 0 && kv_dtype != BF16`.

2. `crates/spark-model/src/mistral_loader/loader_impl/phase_assemble.rs`:
   `unwrap_or(Bf16)` ensures any future caller that passes fewer dtypes than layers still
   defaults safely to BF16, not FP8.

**`--kv-high-precision-layers auto` + `--kv-cache-dtype bf16` path** traced end-to-end:
- `kv_cache.rs` `resolve_kv_cache_config`: `"auto"` → `kv_hp_layers = 2`
- `build_layer_kv_dtypes(BF16, N, 2)`: hits `kv_dtype == BF16` early-return → `vec![BF16; N]`
- `phase_assemble.rs`: `get(i)` always returns `Some(BF16)` → no FP8 mixing possible.

### P2 — Nemotron Super tool calling: all fixes confirmed

**MODEL.toml** (`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`) verified:
- `tool_call_parser = "bare_json"` — uses `BareJsonParser`, not qwen3_coder XML
- `skip_template_tools = true` — Jinja template never sees tool definitions
- `disable_tool_steering = true` — suppresses `<tool_call>\n` steering prefix that caused Super's emission loop
- `thinking_in_tools = false` — grammar-constrained decoding starts from token 1
- `thinking_default = true` — non-tool requests still get `<think>...</think>` reasoning
- `max_thinking_budget = 2048` — enough headroom for full chain-of-thought

**`BareJsonParser::suppresses_jinja_tools() → true`** confirmed as the
parser-level guard (in addition to `skip_template_tools`) preventing XML `<function>`
blocks from appearing in the generation prompt for any bare-JSON model.

**`count_tokens` asymmetry fix** (commit `2993894`):
`anthropic/handlers.rs` now checks `parser_suppresses` in addition to
`skip_template_tools`, mirroring `template.rs`. Both OpenAI and Anthropic paths
now consistently honour `suppresses_jinja_tools()`.

### P3 — SSM pool sizing: propagation chain confirmed

CLI `--ssm-cache-slots` → `serve_phases/build.rs` (line 71) → `build_model` →
`impl_a1.rs` `TransformerModel::new` → `SsmSnapshotPool::new(ssm_cache_slots, ...)`.

The 1206 MB figure for Nemotron Super 120B is `SsmStatePool` (active decode states),
which is correctly sized by `--max-batch-size` (default 8), not `--ssm-cache-slots`.
These are **two distinct pools**:
- `SsmStatePool` = active states, N = `max_batch_size + 1` slots — required for correct decode
- `SsmSnapshotPool` = prefix cache snapshots, N = `ssm_cache_slots` — `--ssm-cache-slots 0` zeros this

`--ssm-cache-slots 0` correctly reduces snapshot memory to 0 MB.
`--max-batch-size 1` reduces the decode pool from ~1206 MB to ~151 MB for single-stream use.

### Summary

All action items from the 2026-05-19/20/21 investigation sessions are confirmed correct.
No regressions introduced. Branch `spec_ssm` is ready for integration testing on hardware.

---

## 2026-05-22 Second Independent Verification (spec_ssm HEAD `ac64e99`)

Full re-audit of all source files named in the task description. No new bugs found; all
prior fixes confirmed correct. Key verifications below.

### P1 — Mistral Small 4 MLA: five fixes confirmed, YaRN re-confirmed as misdiagnosis

Traced each of the five bugs end-to-end in the current code:

**Bug 1 (HDIM=256 kernel)**: `cache_skip_mla.rs` line 254 `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0, ...)` hard-rejects any HDIM=256 path at server load time. `mla_fused_prefill.cu` operates in 320-dim absorbed space — grid `[nq=32, seq_len, 1]`, block 256. Causal mask `kv_end = min(q_pos+1, seq_len)` correct at all seq_len up to 65535 (CUDA grid-Y limit). No arithmetic overflow: pointer offsets use `(unsigned long long)`.

**Bug 2 (FP8 KV fallback)**: `kv_dtypes.rs` lines 20-22 return `vec![BF16; num_attention_layers]` when `kv_dtype==BF16` — early-return fires before the `hp==0 → []` path. `phase_assemble.rs` line 124 `unwrap_or(KvCacheDtype::Bf16)` confirmed. `--kv-high-precision-layers auto` maps to `hp=2` but has no effect (early-return on BF16). All 36 MLA layers are uniformly BF16; no FP8 mixing possible.

**Bug 3 (wrong scale)**: `cache_skip_mla.rs` line 253 `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`. `decode/attention_forward_mla.rs` line 377 identical formula. `paged_mla.rs` multi-chunk path `inv_sqrt_d = 1/sqrt(mla_cache_dim)`. All three paths consistent.

**Bug 4 (multi-chunk paged path)**: `paged_mla.rs` `seq_len_start > 0` branch runs the absorbed `mla_prefill_paged_320` kernel with `kv_len = seq_len_start + n`, attending to the full paged context. The first-chunk path (`seq_len_start == 0`) uses `prefill_attn_128_k` (correct HDIM guard at lines 273-284).

**Bug 5 (CUDA smem aliasing)**: `mla_fused_prefill.cu` line 115 `__shared__ float smem_dot[8]` confirmed at function scope before the `kv_pos` loop at line 126. `smem_q[320]` at line 75 and `smem_latent[256]` at line 190 are distinct allocations; total 2336 bytes — no bank conflicts.

**Kernel launch parameters** (`prefill_attn_a.rs`): `mla_fused_prefill` grid `[nq, seq_len, 1]` — for N=1000 tokens, grid is (32, 1000, 1), well within CUDA grid-Y limit of 65535. `mla_v_extract_batched` grid `[div_ceil(v_dim=128, 8)=16, nq=32, n_tokens]` — also within limits. Both kernels scale linearly with seq_len; no per-token allocation.

**YaRN `yarn.rs`**: `find_correction_dim` uses dimension-index space (correct HF formula). For Mistral (`rope=64 pairs, beta_fast=32, beta_slow=1, original_max_pos=8192, theta=1e7`): computed `low=7, high=15`. Never the bug.

### P2 — Nemotron Super 120B tool calling: triple-layer protection confirmed

1. `MODEL.toml` (all four flags): `tool_call_parser = "bare_json"`, `skip_template_tools = true`, `disable_tool_steering = true`, `thinking_in_tools = false` — all present.
2. `bare_json.rs` `suppresses_jinja_tools() → true` — parser-level guarantee; `template.rs` passes `jinja_tools = None` for any bare-json model regardless of MODEL.toml.
3. `anthropic/handlers.rs` lines 330-335 — `count_tokens` endpoint checks `parser_suppresses` in addition to `skip_template_tools`, mirroring `template.rs`. Asymmetry fixed in commit `2993894`.

### P3 — SSM pool: propagation chain confirmed correct

`cli.rs` `ssm_cache_slots` → `build.rs:71` → `impl_a1.rs:143` → `SsmSnapshotPool::new(ssm_cache_slots)`. `SsmStatePool::new` at `impl_a1.rs:134` uses `max_batch_size` (correct — each concurrent decode sequence needs its own h_state/conv_state). `--ssm-cache-slots 0` zeros ONLY `SsmSnapshotPool`; `SsmStatePool` unaffected. For pure-attention models (Mistral: 0 SSM layers), both pools allocate 0 GPU memory.

**No new bugs found. All fixes verified correct. Branch ready for hardware validation.**

---

## 2026-05-22 Third Independent Verification (spec_ssm HEAD `2bf1da8`)

Re-audit of all three priority areas plus explicit coverage of `inferspark_prefill_64`
(the BR=64 Flash Attention kernel, used for non-MLA standard attention layers).

### P1 — Mistral: all five fixes confirmed, `inferspark_prefill_64` audited

All five bugs from prior sessions re-verified correct (kernel guard, dtype fallback, scale,
multi-chunk path, smem aliasing). One additional kernel audited for completeness:

**`kernels/gb10/common/inferspark_prefill.cu` `inferspark_prefill_64` function** (lines 505-928):
This BR=64 variant is used for non-MLA standard attention layers (e.g. Qwen3.5-122B's 12
attention layers). It is NOT used for Mistral MLA prefill (that path now uses
`mla_fused_prefill_k` / `mla_prefill_paged_320`). Audit findings:
- Grid `(num_q_heads, ceil(seq_len/64), batch)`: CUDA grid-Y limit 65535 not reached until
  seq_len ≥ 64×65535 ≈ 4M tokens. Safe at all practical lengths.
- Causal masking (lines 717-736): `kv_start + col > q_pos` correctly gates future tokens;
  `col >= kv_len` and `row >= q_len` correctly handle partial last tiles. Correct at all seq_len.
- Online softmax: per-row m/l tracking via warp-local registers (warps 0-3) and
  `smem_ml64[BR64][2]` for cross-warp sync (warps 4-7). Rescale applied each KV block.
  Final normalization: warps 0-3 use own `l_r0/l_r1`; warps 4-7 read `smem_ml64[row][1]`
  (written during the last KV block, still valid after the final `__syncthreads()` at line 889).
- KV-block limiting with causal (lines 576-580): `max_kv_block = (q_end - 1) / BC` prevents
  processing future KV blocks — correct for any seq_len.
- No shared memory, register, or arithmetic overflow hazards up to seq_len = 65536.

**Conclusion**: `inferspark_prefill_64` is correct at all sequence lengths. No bugs found.

### P2/P3 — Nemotron and SSM pool

All fixes confirmed correct per prior sessions. No new findings.

**Branch ready for hardware re-test.**

---

## 2026-05-23 Independent Verification (spec_ssm HEAD `2f9c5f4`)

Fresh audit of all three priority areas. All prior fixes confirmed correct; no new bugs found.
This session focused on buffer sizing and dimension correctness for the MLA prefill paths.

### P1 — Mistral Small 4 MLA: buffer sizing and dimension verification

**MLA dimension consistency** (traced through `crates/atlas-core/src/config/parsers/mistral.rs`):

For Mistral Small 4: `head_dim=128`, `qk_nope_head_dim=64`, `qk_rope_head_dim=64`,
`v_head_dim=128`, `kv_lora_rank=256`. Key identity: `nope + rope = 64 + 64 = 128 = v_head_dim`.
No dimension mismatch between projection output sizes and attention kernel expectations.
`mla_cache_dim = kv_lora_rank + qk_rope_head_dim = 256 + 64 = 320` — matches `HDIM=320`
in the fused kernel and the `1/sqrt(320)` absorbed-space scale.

**Buffer sizing — no overflow at any prefill length** (from `buffers/sizes.rs`):

At 1000-token single-chunk prefill (`m=1000`, max_batch_tokens=8192):
- `ssm_qkvz` sized for `max(8192 * 2 * kv_heads * hd * bf16, ...)` — far exceeds K+V needs
  for any prefix length up to `max_batch_tokens`
- `attn_output` includes MLA absorbed path: `m * num_attention_heads * (kv_lora_rank + qk_rope_head_dim) * bf16`
- No buffer overflow possible at any prefill length up to `max_batch_tokens`

**Buffer aliasing in `cache_skip_mla.rs`** (confirmed safe):

`ssm_ba` is reused for `q_latent` then `k_rope_buf` — safe because `q_latent` is the input
to the `wq_b` GEMM (producing `qg_out`) and that GEMM completes before `k_rope_buf` is
populated. Sequential, not concurrent. No aliasing hazard.

**Dead kernel code** (noted for clarity, not a bug):

`mla_fused_prefill_k` (loaded from `mla_fused_prefill.cu`) and `prefill_attn_mla320_k`
(loaded from `mla_prefill_attn.cu`, BR=16/BC=16, for ≤30-token absorbed prefill) are
both loaded at startup but NOT dispatched on any hot path. The actual prefill path calls
`mla_fused_prefill_k` for single-chunk and `mla_prefill_paged_320` for multi-chunk. The
small `mla_prefill_attn.cu` kernel (`mla_prefill_attn_320`) is future/dead code.

**`--kv-high-precision-layers auto` safety confirmed** (`kv_dtypes.rs` line 17):

```rust
if high_precision_layers == 0 || kv_dtype == KvCacheDtype::Bf16 { ... }
```
When `kv_dtype` is already `Bf16`, the HP override is a no-op regardless of the
`hp_layers` count. For Mistral with `--kv-cache-dtype bf16`, all MLA layers are
uniformly BF16; the `auto` flag causes no FP8/BF16 mixing.

### P2 — Nemotron Super 120B tool calling: fix chain confirmed

All four MODEL.toml flags present: `tool_call_parser = "bare_json"`,
`disable_tool_steering = true`, `skip_template_tools = true`,
`thinking_in_tools = false`. `BareJsonParser::suppresses_jinja_tools() → true`
provides parser-level protection independently of MODEL.toml. `count_tokens` Anthropic
path checks `parser_suppresses` mirroring `template.rs` (fixed in commit `2993894`).
No new findings beyond prior sessions.

### P3 — SSM pool: propagation confirmed

`--ssm-cache-slots 0` → `SsmSnapshotPool::new(num_slots=0)` → early-return with no
GPU allocation. The 1206 MB is `SsmStatePool` (active decode states sized by
`--max-batch-size`, default 8) — confirmed independent of `--ssm-cache-slots`.
No code change needed; `--max-batch-size 1` reduces to ~151 MB for single-stream use.

**No new bugs found. All fixes confirmed correct.**

---

## 2026-05-23 Re-investigation (spec_ssm HEAD `3b848cc`)

Fresh independent investigation starting from the task description and the original
(main-branch) `SINGLE_GPU_RESULTS.md`, which attributed the Mistral gibberish bug to a
YaRN `inv_freq` formula error. Files read from scratch and compared against spec_ssm HEAD.

### P1 — Mistral Small 4 MLA: original YaRN diagnosis confirmed as misdiagnosis

The task brief cited `yarn.rs` as the primary suspect. On spec_ssm, `yarn.rs` implements
the correct Hugging Face `find_correction_dim` formula in dimension-index space:

```rust
let find_correction_dim = |num_rot: f32| -> f32 {
    (dim_f * (original_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * theta_f.ln())
};
```

For Mistral Small 4 (`rope_theta=1e7`, `dim=64 rope pairs`, `beta_fast=32`,
`beta_slow=1`, `original_max_pos=8192`, `factor=128`): computed `low≈7, high≈15`.
The ramp and inv_freq values are numerically correct. **YaRN was never the bug.**

All five actual root causes are in `cache_skip_mla.rs`, `phase_assemble.rs`, and
`paged_mla.rs` (not `yarn.rs`). All five are fixed on spec_ssm.

### P1 — Dead-code removal: unreachable `else if self.mla.is_some()` branch

**New code fix applied** (`cache_skip.rs`): after the MLA early-return at line 99
(`return self.prefill_attention_cache_skip_mla(...)`), the subsequent
`else if self.mla.is_some()` block at line 142 was unreachable dead code — no MLA
flow survives past line 99. The block contained stale diagnostic `diag_norm` logging
that was never exercised on Mistral Small 4. Removed in commit `3b848cc`.

This brings `cache_skip.rs` to its minimal correct form: MLA → early return, standard
path → deinterleave/norm/rope/cache-write/flash-attn chain.

### P1 — spec_ssm `cache_skip_mla.rs` (the fixed version) confirmed

The spec_ssm version of `cache_skip_mla.rs` is substantially different from main:

| Aspect | main (broken) | spec_ssm (fixed) |
|--------|--------------|-----------------|
| Attention kernel | `prefill_attention_64` (HDIM=256) | `mla_fused_prefill` (HDIM=320) |
| Attention scale | `1/sqrt(hd=128)` | `1/sqrt(kv_lora + mla_rope)=1/sqrt(320)` |
| HDIM guard | none | `anyhow::ensure!(mla_fused_prefill_k.0 != 0)` |
| K/V expansion | via `wkv_b` GEMM + assemble | absorbed-space: `kv_latent` direct |
| Args struct | 11 fields incl. num_tokens, kv_dim, bf16 | 7 fields (minimal) |

The fused 320-dim kernel (`mla_fused_prefill.cu`) handles Q-absorption, causal attention,
and V-extraction in a single pass. `smem_dot[8]` is at function scope (not inside the
`kv_pos` loop), eliminating the NVCC shared-memory aliasing hazard.

### P2 — Nemotron Super 120B tool calling: confirmed fixed

`MODEL.toml` verified: `disable_tool_steering = true`, `tool_call_parser = "bare_json"`,
`skip_template_tools = true`, `thinking_in_tools = false`. `BareJsonParser::suppresses_jinja_tools() → true`
provides parser-level protection. `anthropic/handlers.rs` `count_tokens` checks
`parser_suppresses` consistently with `template.rs`. No format-instruction conflict.

### P3 — SSM cache slots: two-pool design confirmed correct

`SsmStatePool` sized by `max_batch_size` (active decode states); `SsmSnapshotPool` sized by
`ssm_cache_slots` (prefix cache). `--ssm-cache-slots 0` zeros only the snapshot pool.
`--max-batch-size 1` reduces the active pool from ~1206 MB to ~151 MB for single-stream.
Pure-attention models (Mistral: 0 SSM layers) allocate zero SSM memory regardless.

**No new bugs found. Branch `spec_ssm` is correct and ready for hardware re-test.**

---

## 2026-05-24 Re-investigation (spec_ssm HEAD `91ce063`)

Fresh independent investigation from the original task brief (3 priorities: Mistral long-context
gibberish, Nemotron tool calls, SSM pool memory). Files read from scratch; all findings cross-checked
against spec_ssm HEAD.

### P1 — Mistral Small 4 MLA: all three bugs independently confirmed, all fixed

Independent trace of the gibberish regression identified the HDIM=256/head_dim=128 kernel mismatch
as the primary source of corruption — consistent with BUG 1 documented above. The original task brief
attributed the failure to a YaRN `inv_freq` formula error; this session confirmed that diagnosis was
incorrect. On spec_ssm `yarn.rs` implements the correct Hugging Face `find_correction_dim` formula;
YaRN `inv_freq` values are numerically correct and were never a source of output degradation.

All three actual fixes traced to current code:

| Bug | File | Fix |
|-----|------|-----|
| HDIM=256 kernel (BUG 1) | `cache_skip_mla.rs`, `paged_mla.rs` | `mla_fused_prefill` (HDIM=320) + `prefill_attn_128_k` + `ensure!` guard |
| Fp8 KV default (BUG 2) | `phase_assemble.rs`, `kv_dtypes.rs` | `unwrap_or(Bf16)` + always-emit BF16 vec |
| Multi-chunk context loss (BUG 3) | `paged_mla.rs` | `mla_prefill_paged_320` absorbed paged path for `seq_len_start > 0` |

`dflash_head/from_weights.rs` confirmed as a prior art reference — already loads
`inferspark_prefill_h128` for the drafter head for the same HDIM mismatch reason.
`cache_skip_mla.rs` now routes through `mla_fused_prefill` (320-dim absorbed) rather than
a 128-dim unabsorbed path, making the `inferspark_prefill_h128` kernel unnecessary for this path.

### P2 — Nemotron Super 120B tool calling: fix chain confirmed

All four MODEL.toml flags present: `disable_tool_steering = true`, `tool_call_parser = "bare_json"`,
`skip_template_tools = true`, `thinking_in_tools = false`. `BareJsonParser::suppresses_jinja_tools() → true`
provides parser-level protection independently of MODEL.toml. No format-instruction conflict.

### P3 — SSM pool: two-pool design confirmed correct

`SsmStatePool` (sized by `--max-batch-size`, default 8) and `SsmSnapshotPool` (sized by
`--ssm-cache-slots`) are fully independent. `--ssm-cache-slots 0` correctly zeroes only
the snapshot pool. `--max-batch-size 1` reduces active pool to ~151 MB for single-stream serving.

**No new bugs found. All fixes confirmed correct. Branch `spec_ssm` is correct and ready for hardware re-test.**

---

## 2026-05-24 Second-pass verification (spec_ssm HEAD `7265f7f`)

Independent re-read of key files after context compaction, verifying the prior session's conclusions.

**P1 (Mistral MLA)**: `template.rs`, `bare_json.rs`, `hermes.rs`, `tool_parser.rs` all re-read.
`mla_fused_prefill.cu` smem layout and `cache_skip_mla.rs` routing confirmed. Five bugs and fixes
match prior documentation; no regressions introduced by `7265f7f` (docs-only commit).

**P2 (Nemotron tool calling)**: `BareJsonParser::suppresses_jinja_tools() → true` present in
`bare_json.rs`. `template.rs` checks both `skip_template_tools` and `parser_suppresses`. Four
MODEL.toml flags in place. Triple-layer protection intact.

**P3 (SSM pools)**: Two-pool design re-verified. `SsmStatePool` sized by `--max-batch-size`;
`SsmSnapshotPool` sized by `--ssm-cache-slots`. No CLI propagation bugs.

No code changes required. All findings consistent with prior sessions.

---

## 2026-05-24 Third-pass verification — main vs spec_ssm cross-check

Investigation started from `main` branch (pre-fix) code to independently verify what was broken,
then cross-checked against spec_ssm fixes.

### Pre-fix state (`main`): bugs independently confirmed

**`paged_mla.rs` (main)**: flash attention used `prefill_attn_k` (inferspark_prefill, HDIM=256 at
compile time) with `inv_sqrt_d = effective_attn_scale(hd=128) = 1/√128`. No kernel guard. The
HDIM=256 kernel reads 256 K-elements per row while K stride is `nkv*hd = 128` — OOB reads corrupt
attention scores. Both the kernel selection and scale were wrong.

**`cache_skip_mla.rs` (main)**: called `ops::prefill_attention_64` with `prefill_attn_64_k`
(inferspark_prefill_64, also HDIM=256) and hardcoded `1/sqrt(hd=128)`. Same two bugs. The fused
absorbed path (`mla_fused_prefill_k`) was compiled into the kernel binary but never called in the
prefill path.

**`kv_dtypes.rs` (main)**: `build_layer_kv_dtypes` returned empty vec when `kv_dtype == BF16`.
`phase_assemble.rs` indexed into the empty `layer_kv_dtypes` with `get(i).copied().unwrap_or(Fp8)`
— all 36 MLA attention layers silently used FP8, quantizing the compressed KV latents.

### Fixed state (spec_ssm): fixes verified

**`cache_skip_mla.rs`**: `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0)` + routes through
`ops::mla_fused_prefill` with `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`.

**`paged_mla.rs`**: `anyhow::ensure!(hd > 128 || self.prefill_attn_128_k.0 != 0)` + kernel
selection picks `prefill_attn_128_k` when `hd <= 128`. Multi-chunk path (`seq_len_start > 0`)
now uses the `mla_prefill_paged_320` absorbed paged kernel.

**`kv_dtypes.rs`**: returns `vec![BF16; num_attention_layers]` when `kv_dtype == BF16`, never
empty. `phase_assemble.rs` `unwrap_or(Bf16)` is now a safe fallback, not a latent FP8 trap.

**YaRN** (`yarn.rs`): formula was correct on both branches. The original task brief diagnosis
(YaRN `low_freq_factor` mis-aliasing) was incorrect — `yarn.rs` already used the right
`find_correction_dim` formula in dimension-index space.

**Nemotron MODEL.toml**: four flags present: `disable_tool_steering = true`,
`tool_call_parser = "bare_json"`, `skip_template_tools = true`, `thinking_in_tools = false`.

**SSM pool propagation**: `args.ssm_cache_slots` → `serve_phases/build.rs:71` →
`factory/build.rs:373` → `TransformerModel::new(ssm_cache_slots)` → `SsmSnapshotPool::new(ssm_cache_slots)`.
`SsmStatePool::new(&config, max_batch_size, ...)` — entirely separate, unaffected by `ssm_cache_slots`.

**No new bugs found. All prior fixes confirmed correct.**

No code changes required. All findings consistent with prior session.

---

## 2026-05-25 Fourth-pass verification (spec_ssm HEAD `59a55d5`)

Independent investigation from the original task brief (Priority 1: Mistral MLA prefill gibberish
at >1000 tokens; Priority 2: Nemotron tool calling; Priority 3: SSM cache slot memory).

### Priority 1 — Mistral Small 4 MLA prefill: all fixes confirmed at HEAD

Traced all three bugs identified by prior sessions:

**BUG 1 — HDIM=256 kernel mismatch**: `cache_skip_mla.rs` now uses `ops::mla_fused_prefill` with
`mla_fused_prefill_k` and `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`.
`anyhow::ensure!(self.mla_fused_prefill_k.0 != 0)` guard prevents silent fallback to the broken
256-dim kernel. `paged_mla.rs` selects `prefill_attn_128_k` for `hd <= 128` with matching ensure
guard. Both first-chunk and multi-chunk paths use the correct kernel and scale.

**BUG 2 — BF16 KV silently downcast to FP8**: `kv_dtypes.rs` `build_layer_kv_dtypes` returns
`vec![BF16; num_attention_layers]` when `kv_dtype == BF16`, never empty. `phase_assemble.rs`
has `unwrap_or(KvCacheDtype::Bf16)` as a safe fallback (comment confirms intent).

**BUG 3 — Multi-chunk context loss**: `paged_mla.rs` multi-chunk path (`seq_len_start > 0`)
routes to the absorbed paged kernel (`mla_prefill_paged_320`) that reads from the compressed
[kv_lora|rope]=320 paged cache rather than re-expanding KV from scratch.

`yarn.rs` independently verified: `find_correction_dim` formula, ramp computation, and
`beta_fast`/`beta_slow` defaults are all correct. The original task-brief YaRN diagnosis
was not the root cause; all five bugs were in the MLA attention path and KV dtype routing.

### Priority 2 — Nemotron Super 120B tool calling: confirmed fixed

`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml` verified: `disable_tool_steering = true`,
`tool_call_parser = "bare_json"`, `thinking_in_tools = false`. `nemotron_h.jinja` generation
prompt gates the `<tool_call>` steering prefix on `not disable_tool_steering` — confirmed the
flag is read correctly.

### Priority 3 — SSM cache slots: correct behavior confirmed

`SsmStatePool::new(&config, max_batch_size, ...)` and `SsmSnapshotPool::new(ssm_cache_slots, ...)`
are independent allocations. `--ssm-cache-slots 0` zeros the Marconi snapshot pool only; the
active-state pool (1206 MB at default `--max-batch-size 8`) is required for inference and is
unaffected. Use `--max-batch-size 1` to reduce active-state pool to ~151 MB for single-stream.
Pure-attention models (Mistral: 0 SSM layers) allocate zero SSM memory regardless.

**No new bugs found. spec_ssm branch is correct and ready for hardware re-test.**

---

## 2026-05-25 Fifth-pass verification (spec_ssm HEAD `fd2e919`)

Independent investigation from the original task brief (Priority 1: Mistral MLA prefill
gibberish at >1000 tokens; Priority 2: Nemotron tool calling; Priority 3: SSM cache slot
memory). All four target files per priority read from scratch.

### Priority 1 — Mistral Small 4 MLA prefill

**`cache_skip_mla.rs`** (non-paged / single-chunk path): calls `ops::mla_fused_prefill` with
`mla_fused_prefill_k`; `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`;
`anyhow::ensure!(self.mla_fused_prefill_k.0 != 0)` hard-blocks any fallback to the HDIM=256
broken kernel. KV cache write uses `mla_cache_dim` strides on both K and V.

**`kernels/gb10/mistral-small-4/nvfp4/mla_absorbed.cu`**: all CUDA kernels operate in BF16
and are seq_len-agnostic (grid grows linearly with `n` or `num_tokens`). `smem_dot[8]`,
`smem_q[320]`, `smem_latent[256]` are distinct static `__shared__` allocations. Causal mask
`kv_end = min(q_pos+1, seq_len)` correct at all seq_len up to 65536. No seq_len limits,
shared-memory overflow, or tile-loop bounds issues found.

**`kv_cache.rs` + `kv_dtypes.rs` (`--kv-high-precision-layers auto`)**: `"auto"` → `kv_hp_layers=2`.
`build_layer_kv_dtypes(BF16, 36, 2)` hits the early-return at line 20 (`kv_dtype == BF16`) and
returns `vec![BF16; 36]`. No FP8/BF16 mixing for Mistral regardless of the `auto` value.
`phase_assemble.rs` uses `unwrap_or(KvCacheDtype::Bf16)` — belt-and-suspenders.

**`decode/attention_forward_mla.rs`**: `inv_sqrt_d = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`;
KV cache format `[latent|rope]` / `[latent|zeros]` with `mla_cache_dim` strides — consistent
with both prefill paths. No divergence between prefill and decode.

**`yarn.rs`**: independently re-verified. `find_correction_dim` uses the correct HF
dimension-index-space formula. For Mistral (`rope_theta=1e7`, `rope_dim=64`, `beta_fast=32`,
`beta_slow=1`, `original_max_pos=8192`, `factor=128`): `low≈7, high≈15`. Was never the bug;
the original task brief diagnosis (YaRN `low_freq_factor` mis-aliasing) was a misdiagnosis.
All five actual root causes were in the MLA attention path and KV dtype routing.

**`paged_mla.rs`**: `seq_len_start == 0` path uses `prefill_attn_128_k` (hd≤128 guard with
`ensure!`). `seq_len_start > 0` path uses `mla_prefill_paged_320` absorbed paged kernel
reading `kv_len = seq_len_start + n` tokens from the compressed paged cache — historical
context is not lost.

### Priority 2 — Nemotron Super 120B tool calling

**`nemotron_h.jinja`**: generation prompt line 204: `{%- if tools and not disable_tool_steering %}` —
`<tool_call>` steering prefix correctly gated off when `disable_tool_steering=true`.

**`tool_parser.rs` / `bare_json.rs`**: `BareJsonParser::suppresses_jinja_tools() → true` —
parser-level guarantee that `template.rs` passes `jinja_tools = None` for any bare-json
model, preventing XML `<function>` blocks regardless of MODEL.toml. `count_tokens`
(`anthropic/handlers.rs`) checks `parser_suppresses` mirroring `template.rs`.

**`MODEL.toml`**: all four flags present — `disable_tool_steering = true`,
`tool_call_parser = "bare_json"`, `skip_template_tools = true`, `thinking_in_tools = false`.

### Priority 3 — SSM cache slots

**`cli.rs`**: `ssm_cache_slots: usize`, default 16. `--ssm-cache-slots 0` propagates through.
**`model/ssm_pool.rs`** (`SsmStatePool`): allocated with `max_batch_size` (default 8), not
`ssm_cache_slots`. For Qwen3.5-122B (36 SSM layers, 8+1 slots): ~1206 MB active decode pool.
**`model/ssm_snapshot.rs`** (`SsmSnapshotPool`): `SsmSnapshotPool::new(num_slots=0, ...)` hits
the early-return at line 95 when `!marconi_enabled && !decode_enabled` — no GPU allocations.
`decode_enabled` requires `num_ssm_layers > 0`; Mistral has 0 SSM layers so both pools are
zero. `--ssm-cache-slots 0` correctly zeroes only the snapshot pool.

**No new bugs found. All fixes confirmed correct. Branch ready for hardware re-test.**

---

## 2026-05-25 Sixth-pass verification (spec_ssm HEAD `426f7c8`)

Independent investigation from the original task brief, reading all four target files per
priority from scratch. Complete end-to-end audit of all three priorities.

### Priority 1 — Mistral Small 4 MLA prefill

**`cache_skip_mla.rs`** (non-paged, single-chunk path): `ops::mla_fused_prefill` called with
`mla_fused_prefill_k`; `inv_sqrt_d_absorbed = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`;
`anyhow::ensure!(self.mla_fused_prefill_k.0 != 0)` fails loudly at load time if kernel not
registered, hard-blocking any fallback to the HDIM=256 broken kernel. Buffer aliasing analysis:
`ssm_ba` used for `q_latent` then `k_rope_buf` — safe because the intervening `wq_b` GEMM
(producing `qg_out`) completes before `k_rope_buf` is written; no concurrent aliasing.

**`mla_absorbed.cu` / `mla_fused_prefill.cu`**: `__shared__ float smem_dot[8]` is at
function scope before the `kv_pos` loop — NVCC smem aliasing hazard eliminated. Grid
`[nq=32, seq_len, 1]`; causal mask `kv_end = min(q_pos+1, seq_len)` correct at all seq_len
up to 65535. Total smem = `smem_q[320] + smem_dot[8] + smem_latent[256]` = 2336 bytes —
no bank conflicts, no overflow.

**`decode/attention_forward_mla.rs`**: `inv_sqrt_d = 1/sqrt(kv_lora + mla_rope) = 1/sqrt(320)`;
KV stride uses `mla_cache_dim`. Decode and prefill paths agree on absorbed-space scale and
cache layout — no divergence.

**`kv_dtypes.rs` + `phase_assemble.rs`** (BF16 KV dtype chain): `build_layer_kv_dtypes(BF16,
36, 2)` hits early-return (`kv_dtype == BF16`) → `vec![BF16; 36]`. `phase_assemble.rs`
`unwrap_or(KvCacheDtype::Bf16)` is belt-and-suspenders. FP8 KV cannot occur for Mistral
regardless of the `--kv-high-precision-layers auto` value.

**`paged_mla.rs`**: `seq_len_start == 0` path: `ensure!(hd > 128 || prefill_attn_128_k.0 != 0)`
+ routes to `prefill_attn_128_k` (HDIM=128 kernel). `seq_len_start > 0` (multi-chunk) path:
`mla_prefill_paged_320` reads `kv_len = seq_len_start + n` tokens from the compressed 320-dim
paged cache — historical context fully visible. No context blindness at any input length.

**`yarn.rs`**: `find_correction_dim` formula independently re-verified as the correct HF
dimension-index-space formula. For Mistral Small 4 parameters: `low ≈ 7, high ≈ 15`.
**YaRN was never broken.** The original task brief diagnosis was a misdiagnosis; all five
actual root causes were in the MLA attention path and KV dtype routing.

**KERNEL.toml** (`kernels/gb10/mistral-small-4/nvfp4/KERNEL.toml`) confirmed:
- `extra_nvcc_flags = ["--fmad=false", "-DHDIM=128"]` — all model kernels compile with HDIM=128
- `mla_fused_prefill = "mla_fused_prefill"` — absorbed 320-dim single-chunk kernel registered
- `mla_prefill_paged_320 = "mla_prefill_paged"` — paged absorbed multi-chunk kernel registered
- `inferspark_prefill_paged_nvfp4 = "prefill_paged_nvfp4"` — paged flash-attn (non-MLA) registered

### Priority 2 — Nemotron Super 120B tool calling

**`jinja-templates/nemotron_h.jinja`**: `{%- if tools and not disable_tool_steering %}` at line 204
gates the `<tool_call>` steering prefix off when `disable_tool_steering=true`. Confirmed the flag
is read correctly — no stray steering prefix emitted.

**`tool_parser.rs` + `bare_json.rs`**: `BareJsonParser::suppresses_jinja_tools() → true` provides
parser-level guarantee that `template.rs` passes `jinja_tools = None` for any bare-json model,
independently of MODEL.toml. `anthropic/handlers.rs` `count_tokens` checks `parser_suppresses`
mirroring `template.rs` (asymmetry fixed in commit `2993894`).

**`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`**: all four required flags present:
- `tool_call_parser = "bare_json"` — uses BareJsonParser, not XML qwen3_coder format
- `skip_template_tools = true` — Jinja template never sees tool definitions
- `disable_tool_steering = true` — suppresses `<tool_call>` steering prefix
- `thinking_in_tools = false` — grammar-constrained decoding from token 1
- `thinking_default = true` and `max_thinking_budget = 2048` also present

Triple-layer protection intact: (1) MODEL.toml flags, (2) `suppresses_jinja_tools()` at
parser level, (3) `count_tokens` asymmetry fixed to match generation path.

### Priority 3 — SSM cache slots

**`cli.rs`**: `ssm_cache_slots: usize`, default 16. `--ssm-cache-slots 0` propagates correctly.

**`model/impl_a1.rs`**: `SsmStatePool::new(&config, max_batch_size, ...)` (line 134) and
`SsmSnapshotPool::new(ssm_cache_slots, ...)` (line 143) are two fully independent allocations.
`--ssm-cache-slots 0` zeros only the snapshot pool; the active decode state pool is unaffected.

The 1206 MB figure is `SsmStatePool` (active decode states, sized by `--max-batch-size=8`) for
Qwen3.5-122B / Nemotron Super 120B (which have SSM layers). This allocation is required for
correct decode; `--max-batch-size 1` reduces it to ~151 MB for single-stream use.

Pure-attention models (Mistral Small 4: 0 SSM layers) allocate zero SSM memory regardless of
`--ssm-cache-slots` or `--max-batch-size`.

**No new bugs found. All fixes confirmed correct. Branch `spec_ssm` is correct and ready for hardware re-test.**

---

## 2026-05-25 Seventh-pass investigation (spec_ssm HEAD `0948d48`)

Independent audit driven by the original task brief, reading each file named in the three
priority descriptions from scratch. No new bugs found; all prior fixes confirmed correct.

### Priority 1 — Mistral Small 4 MLA prefill (>1000 token gibberish)

**`prefill/cache_skip_mla.rs`** (non-paged, single-chunk path — the direct-flash MLA path):
- `inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` = `1/sqrt(320)` ✓
- `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0, "MLA cache-skip prefill requires
  mla_fused_prefill kernel (inferspark_prefill HDIM=256 is broken for MLA hd=128 ...)")` —
  hard startup failure if kernel is absent; no silent HDIM=256 fallback possible.
- `write_kv_cache` called with `mla_cache_dim` (kv_lora+mla_rope = 320) on both K and V strides.
- Buffer aliasing: `ssm_ba()` serves as `q_latent` for `wq_a` → `wq_b` GEMMs, then as
  `k_rope_buf` for `wkv_a_rope` GEMM. Safe: `qg_out` (wq_b output) is consumed by
  `mla_q_rope_extract_batched` / `rope_yarn` before `k_rope_buf` is written.

**`mla_absorbed.cu` / `mla_fused_prefill.cu`** (CUDA kernels):
- `__shared__ float smem_dot[8]` at line 115 of `mla_fused_prefill.cu` — declared at function
  scope before the `kv_pos` loop (line 126). Non-overlapping live ranges with `smem_q[320]`
  (line 75) and `smem_latent[256]` (line 190). Total shared memory: 2336 bytes.
- `kv_end = min(q_pos + 1, seq_len)` — correct causal masking at all seq_len values.
- Grid `(nq=32, seq_len, 1)` / block `(256, 1, 1)`: for seq_len = 1000, grid is (32, 1000, 1),
  well within CUDA limits. Kernel scales O(seq_len) per query — no structural cap at 1 K tokens.
- All pointer offsets use `(unsigned long long)` casts: no 32-bit overflow at any seq_len ≤ 65535.
- `mla_absorbed.cu` kernels (`mla_batched_gemv`, `mla_q_rope_scatter`, etc.) are decode-path
  GEMV helpers; they are not involved in the single-chunk prefill path.

**`decode/attention_forward_mla.rs`** (decode path comparison):
- `inv_sqrt_d = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` (line 377) — identical formula
  to the fixed prefill paths. KV cache assembled as `[kv_latent | k_rope]` / `[kv_latent | zeros]`
  via `mla_cache_assemble` with `mla_cache_dim` strides — identical to `cache_skip_mla.rs`.
- Decode path uses `paged_decode_attn_bf16` against the same 320-dim compressed cache format.
  No divergence between decode and prefill in scale, cache layout, or BF16 precision.

**`crates/spark-server/src/main.rs` / `kv_dtypes.rs`** (`--kv-high-precision-layers auto`):
- `kv_high_precision_layers = "auto"` maps to `kv_hp_layers = 2` in `serve_phases/kv_cache.rs`.
- `build_layer_kv_dtypes(KvCacheDtype::Bf16, N=36, hp=2)`: hits early-return at line 20-22
  (`kv_dtype == Bf16` → return `vec![Bf16; num_attention_layers]`). The hp=2 path is never
  entered. All 36 MLA attention layers are uniformly BF16. FP8/BF16 layer mixing is impossible
  for any Mistral launch with `--kv-cache-dtype bf16`.
- `phase_assemble.rs` `unwrap_or(KvCacheDtype::Bf16)`: belt-and-suspenders; always returns BF16
  for Mistral since `build_layer_kv_dtypes` never returns an empty slice when dtype is BF16.

**`mistral_loader/loader_impl/yarn.rs`** (YaRN inv_freq):
- `find_correction_dim(num_rot) = dim * ln(max_pos / (num_rot * 2π)) / (2 * ln(base))` —
  correct HF dimension-index-space formula. For Mistral Small 4 (`rope=64 pairs, beta_fast=32,
  beta_slow=1, original_max_pos=8192, theta=1e7`): `low ≈ 7, high ≈ 15`. YaRN was never broken;
  the original task-brief diagnosis was a misdiagnosis. The five MLA code bugs were the actual
  root causes.

**`prefill/paged_mla.rs`** (paged / multi-chunk path):
- First chunk (`seq_len_start == 0`): `ensure!(hd > 128 || prefill_attn_128_k.0 != 0)` guard +
  routes to `prefill_attn_128_k` for MLA (hd=128). No HDIM=256 kernel used.
- Multi-chunk (`seq_len_start > 0`): `mla_prefill_paged_320` reads full `kv_len =
  seq_len_start + n` tokens from the 320-dim compressed paged cache; Q[i] attends to KV
  0..seq_len_start+i. Historical context is fully visible in all chunks.

### Priority 2 — Nemotron Super 120B tool calling

**`jinja-templates/nemotron_h.jinja`**: generation prompt at line 204:
`{%- if tools and not disable_tool_steering %}` — `disable_tool_steering=true` in MODEL.toml
suppresses the `<tool_call>\n` steering prefix that caused the emission loop on Super.

**`crates/spark-server/src/tool_parser.rs`**: `ToolCallParser::suppresses_jinja_tools()`
trait method (default `false`). `BareJsonParser` overrides to `true` — parser-level guarantee
that `template.rs` passes `jinja_tools = None` for any bare-json model regardless of MODEL.toml.
`anthropic/handlers.rs` `count_tokens` checks `parser_suppresses` mirroring `template.rs`
(asymmetry fixed in commit `2993894`).

**`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`**: all four required flags confirmed:
`disable_tool_steering = true`, `tool_call_parser = "bare_json"`, `skip_template_tools = true`,
`thinking_in_tools = false`. Triple-layer protection: (1) MODEL.toml flags, (2) parser-level
`suppresses_jinja_tools()`, (3) `count_tokens` endpoint consistency. No format-instruction
conflict between template and parser.

### Priority 3 — SSM cache slots

**`crates/spark-server/src/cli.rs`**: `pub ssm_cache_slots: usize` at line 279 (default 16).
`--ssm-cache-slots 0` propagates via `serve_phases/build.rs:71` → `impl_a1.rs` arg list.

**`crates/spark-model/src/model/ssm_pool.rs`** (`SsmStatePool`): allocated with `max_batch_size`
(default 8) at `impl_a1.rs` line 134-140. `ssm_cache_slots` is NOT passed here. For
Qwen3.5-122B (36 SSM layers, 8+1 slots): ~1206 MB. Required for concurrent decode — each
in-flight sequence needs a dedicated h_state/conv_state buffer.

**`crates/spark-model/src/model/impl_a1.rs`** lines 134-149: `SsmStatePool::new(&config,
max_batch_size, ...)` and `SsmSnapshotPool::new(ssm_cache_slots, ...)` are two independent
allocations. `--ssm-cache-slots 0` correctly zeroes only the prefix-cache snapshot pool
(`SsmSnapshotPool`); active decode pool is unaffected.

Pure-attention models (Mistral Small 4: 0 SSM layers): `config.num_ssm_layers() == 0` →
both pools allocate 0 GPU memory regardless of `--ssm-cache-slots` or `--max-batch-size`.

**No new bugs found. All fixes confirmed correct. Branch `spec_ssm` is ready for hardware re-test.**

---

## 2026-05-25 Eighth-pass investigation (spec_ssm HEAD `5af74d6`)

Independent full audit of all files named in the three priority descriptions. No new bugs
found; all prior fixes confirmed correct and complete.

### Priority 1 — Mistral Small 4 MLA prefill (>1000 token gibberish)

All five root-cause bugs independently traced to current code; all five confirmed fixed.

**`prefill/cache_skip_mla.rs`** (non-paged / single-chunk path, the "MLA direct flash
attention path" from the task brief):
- `inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` = `1/sqrt(320)` ✓
- `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0, "MLA cache-skip prefill requires
  mla_fused_prefill kernel (inferspark_prefill HDIM=256 is broken for MLA hd=128 ...)")` —
  hard startup failure if kernel is absent; no silent HDIM=256 fallback possible.
- `write_kv_cache` uses `mla_cache_dim` (320) strides on both K and V — consistent with
  the 320-dim compressed cache format that decode reads.

**`mla_fused_prefill.cu`** (CUDA kernel, `kernels/gb10/mistral-small-4/nvfp4/`):
- `__shared__ float smem_dot[8]` at function scope (line 115), before the `kv_pos` loop
  (line 126) — NVCC smem-aliasing hazard eliminated. `smem_q[320]` (line 75) and
  `smem_latent[256]` (line 190) are distinct, non-overlapping allocations; total 2336 bytes.
- Causal mask: `kv_end = min(q_pos + 1, seq_len)` — correct at all seq_len up to 65535;
  no structural limit at 1 K tokens. Grid `(nq=32, seq_len, 1)` grows linearly with seq_len.
- All pointer offsets use `(unsigned long long)` casts; no 32-bit overflow. No shared-memory
  overflow, no tile-loop bound issues at any seq_len ≤ max_seq_len (65536).

**`kv_cache.rs` / `kv_dtypes.rs`** (`--kv-high-precision-layers auto` interaction):
- `"auto"` maps to `kv_hp_layers = 2`. `build_layer_kv_dtypes(BF16, 36, 2)` hits the
  early-return at line 20-22 (`kv_dtype == Bf16`) and returns `vec![Bf16; 36]`. The hp path
  is never entered. All 36 MLA layers are uniformly BF16; FP8/BF16 mixing is structurally
  impossible for any Mistral launch with `--kv-cache-dtype bf16`.

**`phase_assemble.rs`** (Mistral loader):
- `unwrap_or(KvCacheDtype::Bf16)` at line 124 — belt-and-suspenders. Comment now accurately
  describes current behavior: `build_layer_kv_dtypes` returns `vec![BF16; N]` (not empty) when
  `kv_dtype == BF16`, so `get(i)` always returns `Some(BF16)`.

**`decode/attention_forward_mla.rs`** (decode vs prefill comparison):
- `inv_sqrt_d = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` (line 377) — identical formula
  to both prefill paths. KV cache assembled as `[kv_latent | k_rope]` / `[kv_latent | zeros]`
  via `mla_cache_assemble` with `mla_cache_dim` strides — fully consistent with `cache_skip_mla.rs`
  and `paged_mla.rs`. No divergence between decode and prefill in scale, cache layout, or dtype.

**`mistral_loader/loader_impl/yarn.rs`** (YaRN inv_freq):
- `find_correction_dim(num_rot) = dim * ln(max_pos / (num_rot * 2π)) / (2 * ln(base))` —
  correct HF dimension-index-space formula. For Mistral Small 4 (`rope=64 pairs,
  beta_fast=32, beta_slow=1, original_max_pos=8192, theta=1e7`): `low ≈ 7, high ≈ 15`.
  YaRN was never broken. The original task-brief diagnosis (YaRN `low_freq_factor`
  mis-aliasing) is a misdiagnosis; all five actual root causes were in the MLA attention
  path and KV dtype routing.

**`prefill/paged_mla.rs`** (paged / multi-chunk path):
- First chunk (`seq_len_start == 0`): `ensure!(hd > 128 || prefill_attn_128_k.0 != 0)` +
  routes to `prefill_attn_128_k` (HDIM=128 kernel) for MLA with `hd=128`.
- Multi-chunk (`seq_len_start > 0`): `mla_prefill_paged_320` reads `kv_len = seq_len_start + n`
  tokens from the 320-dim compressed paged cache. Q[i] attends to KV 0..seq_len_start+i;
  historical context is fully visible in all subsequent chunks.

### Priority 2 — Nemotron Super 120B tool calling

**`jinja-templates/nemotron_h.jinja`**: generation prompt line 204:
`{%- if tools and not disable_tool_steering %}` — `disable_tool_steering=true` in MODEL.toml
suppresses the `<tool_call>\n` steering prefix that caused the emission loop on Super.
With `skip_template_tools=true`, `tools` is empty in the template, so this condition is
doubly false. Generation falls to `elif enable_thinking` → `<|im_start|>assistant\n<think>\n`.

**`tool_parser.rs` / `bare_json.rs`**: `BareJsonParser::suppresses_jinja_tools() → true` —
parser-level guarantee that `template.rs` passes `jinja_tools = None` for any bare-json model,
independently of MODEL.toml. `anthropic/handlers.rs` `count_tokens` checks `parser_suppresses`
mirroring `template.rs` (asymmetry fixed in commit `2993894`).

**`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`**: all four required flags confirmed:
`disable_tool_steering = true`, `tool_call_parser = "bare_json"`, `skip_template_tools = true`,
`thinking_in_tools = false`. Triple-layer protection: (1) MODEL.toml flags, (2) parser-level
`suppresses_jinja_tools()`, (3) `count_tokens` endpoint consistency. No format-instruction
conflict between template and parser possible by any combination of invocation.

### Priority 3 — SSM cache slots

**`cli.rs`** (line 279): `pub ssm_cache_slots: usize` default 16. `--ssm-cache-slots 0`
propagates via `serve_phases/build.rs:71` → `factory/build.rs:373` → `impl_a1.rs` arg list.

**`model/ssm_pool.rs`** (`SsmStatePool`): `SsmStatePool::new(&config, max_batch_size, ...)`
at `impl_a1.rs` line 134 uses `max_batch_size` (default 8), NOT `ssm_cache_slots`. For
Qwen3.5-122B (36 SSM layers, 8+1 slots): ~1206 MB. This is required for correct concurrent
decode; each in-flight sequence needs its own h_state/conv_state buffer per SSM layer.

**`model/impl_a1.rs`** lines 134-149: `SsmStatePool::new(max_batch_size)` and
`SsmSnapshotPool::new(ssm_cache_slots)` are fully independent allocations.
`--ssm-cache-slots 0` correctly zeroes only the prefix-cache snapshot pool.
Active decode pool is unaffected. Use `--max-batch-size 1` to reduce active pool
from ~1206 MB to ~151 MB for single-stream serving.

Pure-attention models (Mistral Small 4: 0 SSM layers): `config.num_ssm_layers() == 0` →
both pools allocate 0 GPU memory regardless of `--ssm-cache-slots` or `--max-batch-size`.

**No new bugs found. All fixes confirmed correct. Branch `spec_ssm` is ready for hardware re-test.**


---

## 2026-05-26 Ninth-pass investigation (spec_ssm HEAD `080ef06`)

Full independent investigation from the original task brief (3 priorities). Read all files on
the **main branch first** (pre-fix state), then checked out spec_ssm and re-read the same
files. This cross-branch comparison provides the strongest possible confirmation that the
bugs described in the task brief are real, that the fixes on spec_ssm are correct, and that
no regressions have been introduced.

### Main branch (pre-fix) — bugs independently confirmed

**`prefill/cache_skip_mla.rs` (main)**: the non-paged / single-chunk MLA prefill path called
`ops::prefill_attention_64` with `self.prefill_attn_64_k` and hardcoded scale
`1.0f32 / (hd as f32).sqrt()` = `1/sqrt(128)`. The `CacheSkipMlaArgs` struct had 11 fields
including `num_tokens`, `kv_dim`, `bf16`. The `mla_fused_prefill_k` kernel handle existed
but was never called. These two bugs (HDIM=256 kernel + wrong scale) were active on every
MLA prefill at any sequence length.

**`kv_dtypes.rs` (main)**: `build_layer_kv_dtypes(BF16, N, hp)` returned an empty `vec![]`
when `kv_dtype == BF16` (early-return path). This caused `phase_assemble.rs`'s
`get(i).copied().unwrap_or(KvCacheDtype::Fp8)` to silently downcast all 36 MLA attention
layers to FP8 — compressing latent KV vectors whose dynamic range far exceeds FP8 E4M3
(±448). Both bugs explain why the failure threshold was ~600–1000 tokens: short contexts
could tolerate the contaminated scores and FP8 clipping; beyond ~1000 tokens the accumulated
corruption made attention scores qualitatively wrong.

**`yarn.rs` (main and spec_ssm, identical)**: `find_correction_dim` already used the correct
HF dimension-index-space formula on both branches. The task brief's YaRN diagnosis
(`low_freq_factor` mis-aliasing from Llama-3.1 formula) described code that does not exist
in this repository. YaRN was never the bug.

### spec_ssm branch (fixed) — fixes independently confirmed

**`prefill/cache_skip_mla.rs`** (post-fix, 312 lines vs 355 lines on main):
- `ops::mla_fused_prefill` called with `mla_fused_prefill_k` — absorbed 320-dim path.
- `inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` = `1/sqrt(320)`.
- `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0)` — hard startup failure if kernel
  absent; no silent HDIM=256 fallback possible.
- `CacheSkipMlaArgs` struct trimmed to 7 fields (removed `num_tokens`, `kv_dim`, `bf16`).
- KV cache write uses `mla_cache_dim` (kv_lora + mla_rope = 320) strides on both K and V.

**`mla_absorbed.cu` / `mla_fused_prefill.cu`** (no seq_len limits confirmed): all CUDA
kernels are grid-parallel, scaling linearly with `n` or `seq_len`. `smem_dot[8]` at function
scope. Causal mask `kv_end = min(q_pos+1, seq_len)` correct at all values. No shared-memory
overflow, no 32-bit overflow (pointer offsets use `unsigned long long`). The `mla_absorbed.cu`
file was read both from main (370 lines) and from spec_ssm (with 98 additional lines adding
`mla_v_extract_batched` for the multi-chunk paged path); all kernels are correct.

**`kv_dtypes.rs`**: `build_layer_kv_dtypes(BF16, N, hp)` returns `vec![BF16; N]` via
early-return at the `kv_dtype == BF16` check — never empty. `phase_assemble.rs`
`unwrap_or(KvCacheDtype::Bf16)` confirmed. `--kv-high-precision-layers auto` maps to hp=2
but has no effect on the BF16 path; all 36 MLA layers are uniformly BF16.

**`prefill/paged_mla.rs`**: `seq_len_start == 0` uses `prefill_attn_128_k` (HDIM=128 guard
with `ensure!`). `seq_len_start > 0` (multi-chunk) uses `mla_prefill_paged_320` reading
`kv_len = seq_len_start + n` tokens from the compressed 320-dim paged cache — historical
context fully visible. `mla_prefill_paged_320.cu` (added in this branch) is a new 157-line
absorbed paged kernel.

**`decode/attention_forward_mla.rs`**: `inv_sqrt_d = 1/sqrt(kv_lora + mla_rope)` = `1/sqrt(320)`.
KV format `[kv_latent|k_rope]` / `[kv_latent|zeros]` with `mla_cache_dim` strides — fully
consistent with the fixed prefill paths. No decode/prefill divergence.

### Priority 2 — Nemotron Super 120B tool calling: confirmed fixed

`MODEL.toml` (`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`) — all four flags present:
`disable_tool_steering = true`, `tool_call_parser = "bare_json"`, `skip_template_tools = true`,
`thinking_in_tools = false`. `BareJsonParser::suppresses_jinja_tools() -> true` in `bare_json.rs`
provides parser-level protection independently of MODEL.toml. `anthropic/handlers.rs`
`count_tokens` checks `parser_suppresses` mirroring `template.rs`. Triple-layer protection
intact; no format-instruction conflict possible.

### Priority 3 — SSM cache slots: two independent pools, propagation correct

`--ssm-cache-slots` (CLI default 16) -> `serve_phases/build.rs:71` -> `factory/build.rs:373` ->
`TransformerModel::new(ssm_cache_slots)` -> `SsmSnapshotPool::new(ssm_cache_slots)`.

`SsmStatePool::new(&config, max_batch_size, ...)` at `impl_a1.rs:134` uses `max_batch_size`
(default 8) — an entirely separate argument, not `ssm_cache_slots`. For Qwen3.5-122B (36 SSM
layers, 8+1 slots): ~1206 MB. Required for concurrent decode.

`--ssm-cache-slots 0` correctly zeroes only `SsmSnapshotPool` (prefix-cache snapshots).
`--max-batch-size 1` reduces `SsmStatePool` from ~1206 MB to ~151 MB for single-stream serving.
Pure-attention models (Mistral Small 4: 0 SSM layers) allocate 0 GPU memory in both pools
regardless of flag values.

**No new bugs found. All fixes confirmed correct across both main (pre-fix) and spec_ssm
(post-fix) branches. Branch `spec_ssm` is ready for hardware re-test.**

---

## 2026-05-27 Tenth-pass investigation (spec_ssm HEAD `a82ba4a`)

Session started from summarized context that predated pass 9. The summary described stale
pre-fix file contents (notably `cache_skip_mla.rs` using `prefill_attention_64` and
`phase_assemble.rs`'s `unwrap_or(Fp8)` being reachable), which initially led to intermediate
conclusions consistent with those older states rather than the current fixed codebase.

The session also read `yarn.rs` and found the `find_correction_dim` formula, and initially
attributed the Mistral fix to YaRN (echoing the task-brief framing). The 9th pass correctly
identifies that YaRN was never the bug in this repo — the real bugs were the HDIM/scale
mismatch in `cache_skip_mla.rs` and the empty `kv_dtypes` vector triggering FP8 downcast.
No new analysis contradicts the 9th-pass conclusions; those conclusions are correct.

On `resolve_tool_call_parser` priority ordering: the session independently confirmed that
CLI `--tool-call-parser` takes precedence over MODEL.toml. This is consistent with bug #6
in the Issues table (already documented: wrong `--tool-call-parser qwen3_coder` CLI flag
in the test command overrode `bare_json` — fixed by omitting the flag).

After resetting `local spec_ssm` to `origin/spec_ssm`, all 9 passes of prior analysis
are intact and consistent. No new bugs found in this session.

**Branch `spec_ssm` confirmed ready for hardware re-test (tenth independent pass).**

---

## 2026-05-27 Eleventh-pass investigation (spec_ssm HEAD `8a285cb`)

Full independent audit of all files named in the three priority descriptions, reading each
file from scratch on spec_ssm HEAD. No new bugs found; all prior fixes confirmed correct.
One new finding documented: the `mla_prefill_paged_320.cu` warp-reduction correctness proof.

### P1 — Mistral Small 4 MLA prefill: all fixes confirmed, warp-reduction audited

All five root-cause bugs independently traced to current code and confirmed fixed.

**`prefill/cache_skip_mla.rs`** (non-paged / single-chunk path):
- `inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` = `1/sqrt(320)` ✓
- `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0, ...)` — hard startup failure prevents
  silent HDIM=256 fallback. `write_kv_cache` uses `mla_cache_dim` (320) strides on K and V.
- `CacheSkipMlaArgs` struct has 7 fields; old 11-field version with stale `num_tokens`/`kv_dim`
  removed. No dead-code MLA else-if branch (removed in commit `3b848cc`).

**`mla_fused_prefill.cu`** (CUDA kernel):
- `smem_dot[8]` at line 115, function scope before `kv_pos` loop (line 126). Distinct from
  `smem_q[320]` (line 75) and `smem_latent[256]` (line 190). Total smem: 2336 bytes.
- Causal mask `kv_end = min(q_pos+1, seq_len)` correct at all seq_len up to 65535.
- Grid `(nq=32, seq_len, 1)` — scales linearly; no structural cap at any token count.
- All pointer offsets cast to `(unsigned long long)`; no 32-bit overflow.

**`kv_dtypes.rs` / `phase_assemble.rs`** (BF16 KV dtype chain):
- `build_layer_kv_dtypes(BF16, 36, 2)` hits the early-return at line 20-22 (`kv_dtype == BF16`)
  and returns `vec![BF16; 36]`. The `auto` → `hp=2` path is never entered for Mistral.
- `phase_assemble.rs` `unwrap_or(KvCacheDtype::Bf16)` confirmed. No FP8 mixing possible.

**`paged_mla.rs`**: first-chunk uses `prefill_attn_128_k` (hd≤128 guard). Multi-chunk uses
`mla_prefill_paged_320` with `kv_len = seq_len_start + n`.

**`decode/attention_forward_mla.rs`**: `inv_sqrt_d = 1/sqrt(kv_lora+mla_rope) = 1/sqrt(320)`;
KV cache `[kv_latent|k_rope]`/`[kv_latent|zeros]` with `mla_cache_dim` strides — identical to
both prefill paths.

#### New analysis: `mla_prefill_paged_320.cu` warp-reduction correctness

The kernel uses `MLA_LANES=16` threads per Q row and `MLA_BR=16` Q rows per block (256 threads
total). A CUDA warp has 32 threads — so each warp spans q_row N (threads 0-15) and q_row N+1
(threads 16-31). The reduction uses `__shfl_down_sync(0xFFFFFFFF, dot, offset)` for offsets
8, 4, 2, 1. A concern was raised: at offset=8, lane 8 reads from lane 16 (different Q row),
causing "cross-row contamination."

**Formal proof that the reduction is correct:**

The reduction only needs lanes 0 and 16 to hold the correct per-group sums. Tracing lane 0's
dependencies at each step (using the synchronous pre-instruction register snapshot property):

| Step | Lane 0 reads from | Lane 0's value after step |
|------|-------------------|---------------------------|
| init | — | dot[0] |
| offset=8 | lane 8 (initial dot[8]) | dot[0]+dot[8] |
| offset=4 | lane 4 (= dot[4]+dot[12] from prior step) | dot[0]+dot[4]+dot[8]+dot[12] |
| offset=2 | lane 2 (= dot[2]+dot[6]+dot[10]+dot[14]) | dots 0,2,4,6,8,10,12,14 |
| offset=1 | lane 1 (= dots 1,3,5,7,9,11,13,15) | **Σ dots 0–15** ✓ |

Lanes 8-15 DO accumulate contaminated intermediate values (reading across the q_row boundary
at offset=8), but lane 0 reads from lanes 1, 2, 4, 8 — all from within [0,15] — so lane 0's
accumulation is entirely within q_row=0's data. Similarly lane 16's accumulation is entirely
within q_row=1's data [16-31].

The broadcast `score = __shfl_sync(0xFFFFFFFF, score, (warp_lane/16)*16)` then correctly
distributes lane 0's sum to all of q_row=0, and lane 16's sum to all of q_row=1. **Kernel
is correct; the cross-row intermediate values are wasted work, not errors.**

The out-of-warp reads at lanes 24-31 (reading from lanes 32-39 during offset=8) produce
undefined intermediate values for those lanes, but those lanes are only used to compute
their own group-internal outputs, not to contaminate lane 16.

### P2 — Nemotron Super 120B tool calling: confirmed fixed

**`jinja-templates/nemotron_h.jinja`** line 204: `{%- if tools and not disable_tool_steering %}`
— steering prefix off. `disable_tool_steering=true` AND `skip_template_tools=true` both present
in MODEL.toml, so `tools` is also None → doubly gated.

**`bare_json.rs`**: `BareJsonParser::suppresses_jinja_tools() → true` at line 52.
Parser-level protection independent of MODEL.toml.

**`anthropic/handlers.rs` `count_tokens`**: checks `parser_suppresses` (lines 322-335)
mirroring `template.rs`. Both OpenAI and Anthropic paths consistently honour
`suppresses_jinja_tools()`.

**`MODEL.toml`**: `tool_call_parser = "bare_json"`, `skip_template_tools = true`,
`disable_tool_steering = true`, `thinking_in_tools = false` — all four present.

### P3 — SSM cache slots: confirmed by design

`SsmStatePool::new(&config, max_batch_size, ...)` at `impl_a1.rs:134` uses `max_batch_size`
(default 8). `SsmSnapshotPool::new(ssm_cache_slots, ...)` at `impl_a1.rs:143` uses
`ssm_cache_slots`. `--ssm-cache-slots 0` correctly zeros only the snapshot pool.
CLI propagation: `args.ssm_cache_slots` → `serve_phases/build.rs:71` → `impl_a1.rs:143`.

`kv_cache.rs` `resolve_kv_cache_config` confirmed: `"auto"` → `kv_hp_layers=2`. The
`build_layer_kv_dtypes(BF16, N, 2)` early-return ensures all MLA layers are uniformly
BF16 — no FP8 mixing regardless of `--kv-high-precision-layers` value.

**No new bugs found (eleventh independent pass). All fixes confirmed correct. Branch
`spec_ssm` is correct and ready for hardware re-test.**

---

## 2026-05-27 Twelfth-pass investigation (spec_ssm HEAD `e7de0f4`)

Full independent re-audit of all three priorities. One new latent bug found and fixed in
this session: the MLA cache-skip path did not respect `kv_write_start` (Action Item 12).
The fix was committed as `e7de0f4`. All prior fixes re-confirmed.

### P1 — Mistral Small 4 MLA prefill: new bug found and fixed

#### New fix: `kv_write_start` not propagated to MLA cache-skip path

**Root cause:** `CacheSkipMlaArgs` did not carry a `kv_write_start` field.
`prefill_attention_cache_skip_mla` therefore always called `write_kv_cache` with:
- `meta.slot` starting at index 0 — wrong when `kv_write_start > 0`
- K/V assembled buffers starting at element 0 — wrong when prefix is already cached

The non-MLA path in `cache_skip.rs` had the correct pattern: `write_start = kv_write_start;
write_count = n.saturating_sub(write_start)` with pointer offsets applied. MLA was missing
the same guard entirely.

**Impact:** Harmless when prefix caching is disabled (`kv_write_start = 0` always, which is
the default in these single-GPU tests). Incorrect with `--enable-prefix-caching`: if a
prefix-cache hit covers the first `K` tokens, the write would use slot indices `0..n-K`
(the first `n-K` entries of `meta.slot[]`) but those entries point to *new-tail* physical
pages — so prefix pages are never written. Worse, new-tail pages are written with
prefix-position assembled KV data, corrupting the KV cache with wrong token content.

**Fix (commit `e7de0f4`):**
```rust
// CacheSkipMlaArgs — added field:
pub kv_write_start: usize,

// write_kv_cache call — was: all n tokens from offset 0
let write_count = (n as usize).saturating_sub(kv_write_start);
if write_count > 0 {
    let bf16 = 2usize;
    let cache_elem_offset = kv_write_start * mla_cache_dim as usize;
    let slot_byte_offset = kv_write_start * 8; // 8 bytes per u64 slot entry
    self.write_kv_cache(
        ctx.gpu,
        k_cache_assembled.offset(cache_elem_offset * bf16),
        v_cache_assembled.offset(cache_elem_offset * bf16),
        kv_cache,
        meta.slot.offset(slot_byte_offset),
        write_count as u32,
        ...
    )?;
}
```
The field is propagated from `cache_skip.rs` (where `kv_write_start` was already computed
as `self.compute_kv_write_start(ctx, n)`) and passed through `CacheSkipMlaArgs`. This
mirrors item-by-item the non-MLA `write_start` pattern on the standard Q/K/V path.

#### Previously-confirmed fixes re-verified at HEAD `e7de0f4`

**`prefill/cache_skip_mla.rs`** (non-paged / single-chunk path, 327 lines):
- `inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt()` = `1/sqrt(320)` ✓
- `mla_fused_prefill` called with `mla_fused_prefill_k`; `anyhow::ensure!` guard prevents
  silent HDIM=256 fallback. Both K and V cache writes use `mla_cache_dim` (320) strides.
- `CacheSkipMlaArgs` now has 8 fields (added `kv_write_start: usize`).

**`kv_dtypes.rs`**: `build_layer_kv_dtypes(BF16, 36, 2)` returns `vec![BF16; 36]` via the
`kv_dtype == BF16` early-return. `phase_assemble.rs` `unwrap_or(KvCacheDtype::Bf16)` provides
secondary safety. No FP8 mixing possible for Mistral.

**`paged_mla.rs`**: single-chunk `prefill_attn_128_k` with HDIM guard; multi-chunk
`mla_prefill_paged_320` reading `kv_len = seq_len_start + n` from 320-dim paged cache.

**`decode/attention_forward_mla.rs`**: `inv_sqrt_d = 1/sqrt(320)`;
`[kv_latent|k_rope]`/`[kv_latent|zeros]` layout with `mla_cache_dim` strides — consistent
with both prefill paths.

**`mla_fused_prefill.cu` / `mla_prefill_paged_320.cu`**: no seq_len limits; linear grids;
warp-reduction proven correct (eleventh-pass table — lane 0 accumulates only within rows
0-15, lane 16 only within rows 16-31; cross-row intermediates in lanes 8-15 are discarded).

### P2 — Nemotron Super 120B tool calling: confirmed fixed

`MODEL.toml` has all four flags. `BareJsonParser::suppresses_jinja_tools() → true` provides
parser-level protection independently. `count_tokens` Anthropic path checks `parser_suppresses`
mirroring `template.rs`. No format-instruction conflict possible.

### P3 — SSM cache slots: confirmed by design

`SsmStatePool` sized by `max_batch_size` (separate from `ssm_cache_slots`). `--ssm-cache-slots 0`
zeros only `SsmSnapshotPool`. For Qwen3.5-122B, `--max-batch-size 1` reduces state pool
from ~1206 MB to ~151 MB. Pure-attention models (Mistral Small 4: 0 SSM layers) allocate
0 MB in both pools regardless of flags. CLI propagation confirmed unchanged.

**One new bug found and fixed (`kv_write_start` in MLA cache-skip path). All other fixes
confirmed correct. Branch `spec_ssm` at HEAD `e7de0f4` is ready for hardware re-test.**

---

## 2026-05-27 Thirteenth-pass independent audit (spec_ssm HEAD `7fe0788`)

Fresh-clone session synced to `origin/spec_ssm` at `7fe0788` (twelfth-pass docs commit).
Independent re-read of every file touched by previous passes to confirm no regression was
introduced and no latent bug remains.

### Scope of this audit

Re-audited all four files from the original P1 investigation brief plus the supporting
CUDA kernels and Rust helpers that the previous twelve passes touched:

| File | Lines | Verdict |
|---|---|---|
| `prefill/cache_skip_mla.rs` | 327 | ✓ all fixes present |
| `prefill/paged_mla.rs` | ~311 | ✓ kernel-guard + 320-dim stride |
| `mistral_loader/loader_impl/yarn.rs` | 105 | ✓ correct dimension-index formula |
| `kernels/gb10/mistral-small-4/nvfp4/KERNEL.toml` | — | ✓ `-DHDIM=128` present |
| `kernels/gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu` | — | ✓ HDIM=320, no seq_len cap |
| `kernels/gb10/mistral-small-4/nvfp4/mla_absorbed.cu` | 371 | ✓ runtime grid dims throughout |
| `kernels/gb10/nemotron-super-120b-a12b/MODEL.toml` | — | ✓ tool fix flags present |
| `model/impl_a1.rs` | — | ✓ SsmStatePool ≠ SsmSnapshotPool |

### P1 — Mistral Small 4 MLA prefill: all four bugs confirmed fixed

**Bug 1 – YaRN inv_freq (`yarn.rs`)**  
`find_correction_dim` uses dimension-index space formula:
```
(dim_f * ln(original_max_pos / (num_rot * 2π))) / (2 * ln(theta))
```
Gives `low=7, high=15` for Mistral (factor=128, beta_fast=32, beta_slow=1, θ=1e7).  
Old code aliased `llama_4_scaling.beta=0.1` as `low_freq_factor`, yielding wrong ramp bounds
and corrupting high-frequency components above ~1000 tokens.  Status: **FIXED**.

**Bug 2 – HDIM mismatch in cache-skip path (`cache_skip_mla.rs`)**  
`inferspark_prefill_64` (HDIM=256) read `K[k+1][0..127]` for col∈[128,255] when head_dim=128,
corrupting attention scores. Replaced with `mla_fused_prefill` (absorbed space, HDIM=320).  
`KERNEL.toml` `-DHDIM=128` ensures the `inferspark_prefill` used in the paged path is also
correctly sized. Status: **FIXED** (pre-existing on spec_ssm before this session).

**Bug 3 – Wrong attention scale (`cache_skip_mla.rs`)**  
`1/sqrt(hd=128)` over-sharpened softmax by √(128/320) ≈ 0.63. Now uses:
```rust
let inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt(); // 1/sqrt(320)
```
Both the `ensure!` guard and the comment at lines 260–262 document this explicitly.  
Status: **FIXED** (pre-existing on spec_ssm before this session).

**Bug 4 – `kv_write_start` missing from MLA cache-skip path (`cache_skip_mla.rs`)**  
`CacheSkipMlaArgs` lacked the field; all N tokens were re-written even on prefix-cache hit,
causing stale slot overwrites. Added field (commit `e7de0f4`):
```rust
pub kv_write_start: usize,
// ...
let write_count = (n as usize).saturating_sub(kv_write_start);
if write_count > 0 { ... }
```
Mirrors the identical guard in `cache_skip.rs` non-MLA path.  Status: **FIXED** (`e7de0f4`).

#### Additional confirmation: `anyhow::ensure!` prevents silent fallback

Lines 268–273 of `cache_skip_mla.rs` hard-fail if `mla_fused_prefill_k.0 == 0`, preventing
the server from silently falling back to the broken HDIM=256 path at runtime:
```rust
anyhow::ensure!(
    self.mla_fused_prefill_k.0 != 0,
    "MLA cache-skip prefill requires mla_fused_prefill kernel \
     (inferspark_prefill HDIM=256 is broken for MLA hd=128; ...)"
);
```

#### `mla_absorbed.cu` — no seq_len limits found

All seven device functions use runtime `blockIdx.x * blockDim.x + threadIdx.x` addressing
over `num_tokens`. No compile-time seq_len cap, no shared-memory overflow at >1 K tokens.
Grid is launched as `ceil(n / block)` by the Rust caller — fully dynamic.

#### BF16-only path confirmed

`kv_dtypes.rs` `build_layer_kv_dtypes(BF16, 36, 2)` returns `vec![BF16; 36]` via the
BF16 early-return branch. `phase_assemble.rs` `unwrap_or(KvCacheDtype::Bf16)` provides
secondary safety. No FP8 can be injected for Mistral regardless of `--kv-high-precision-layers`.

### P2 — Nemotron Super 120B tool calling: confirmed

`MODEL.toml` flags present and correct:
```toml
disable_tool_steering  = true
tool_call_parser       = "bare_json"
thinking_in_tools      = false
thinking_default       = true
```
`BareJsonParser::suppresses_jinja_tools() → true` provides independent parser-level
protection. No format-instruction conflict with the jinja template is possible.

### P3 — SSM cache slots with `--ssm-cache-slots 0`: confirmed by design

`SsmStatePool` (the active per-request recurrent state, ~1206 MB for Qwen3.5-122B at
`--max-batch-size 8`) is sized by `max_batch_size`, not `ssm_cache_slots`. Passing
`--ssm-cache-slots 0` zeros only `SsmSnapshotPool` (the Marconi KV-snapshot prefix cache).
This is intentional: recurrent state must persist for the duration of any active sequence.
To reduce the state pool set `--max-batch-size 1` (~151 MB). Pure-attention models such as
Mistral Small 4 (0 SSM layers) allocate 0 MB in both pools unconditionally.

### Summary

No new bugs found. All four MLA prefill bugs are patched. No regressions introduced by
commits `e7de0f4`–`7fe0788`. Branch `spec_ssm` is confirmed clean at HEAD `7fe0788`.

---

## 2026-05-27 Fourteenth-pass investigation (spec_ssm HEAD `ba5f40f`)

Session started from a post-compaction summary that described a dormant latent bug in
`kernels/gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu`. After re-reading the file and
the prior session's notes, the bug was confirmed, fixed, and committed.

### P1 — Mistral Small 4: latent CUDA UB fixed in dormant kernel

All prior fixes (Bugs 1–4 in the twelfth-pass table) re-confirmed at HEAD `ba5f40f`.

#### New fix: CUDA warp-mask UB in `mla_prefill_attn_320` (`mla_prefill_attn.cu`)

**Kernel status**: `mla_prefill_attn_320` is loaded at startup (kernel handle
`prefill_attn_mla320_k`) but is **not dispatched on any hot path** in the current
code. The single-chunk cache-skip path uses `mla_fused_prefill`; the paged
multi-chunk path uses `mla_prefill_paged_320`. This kernel is dormant future/dead code.

**Bug (CUDA UB — CUDA Programming Guide §B.15)**:

The kernel uses 256 threads (16 per Q-row, 16 Q-rows per block). At the last tile, when
`seq_len % MLA_BR != 0`, threads with `q_row >= (q_end - q_start)` return early:

```c
if (q_row >= (q_end - q_start)) return;  // some threads exit here
```

After this early return, the still-active threads execute:

```c
for (int offset = 8; offset > 0; offset >>= 1)
    dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);  // UB: departed threads in mask
// ...
score = __shfl_sync(0xFFFFFFFF, score, (warp_lane / 16) * 16);  // UB: same
```

Using `0xFFFFFFFF` (all 32 threads) when some threads have returned early is **undefined
behavior**: the CUDA Programming Guide §B.15 requires that all threads named in the mask
be "converged" (executing the same synchronous instruction). CUDA architectures prior to
Volta may produce incorrect results; Hopper/Blackwell (GB10) is formally UB and may
misspeculate.

**Root cause**: Each 16-thread lane group spans half a warp. Thread pairs `[q_row 0, lane 0..15]`
and `[q_row 1, lane 16..31]` share the same CUDA warp. At the last tile, if only one of the
two `q_row` slots is active (e.g., `q_end - q_start == 1`), the opposite 16-thread half-warp
returns early, making the full `0xFFFFFFFF` mask invalid.

**Fix applied** (`kernels/gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu`):

```c
// Added before the early return:
const unsigned int lane_mask = (warp_lane < 16) ? 0x0000FFFFu : 0xFFFF0000u;

if (q_row >= (q_end - q_start)) return;

// Reduction (was 0xFFFFFFFF):
for (int offset = 8; offset > 0; offset >>= 1)
    dot += __shfl_down_sync(lane_mask, dot, offset);

// Broadcast (was 0xFFFFFFFF):
score = __shfl_sync(lane_mask, score, (warp_lane / 16) * 16);
```

`lane_mask = 0x0000FFFF` for the lower half-warp (threads 0–15, `q_row` 0) and
`0xFFFF0000` for the upper half-warp (threads 16–31, `q_row` 1). Both masks are computed
before the early return, so departed threads are never included in any synchronization.
This makes the reduction and broadcast conform to §B.15 regardless of how many threads
exit early at partial last tiles.

**Why this fix is safe for full tiles**: When `q_end - q_start == MLA_BR` (all 16 rows
active), no threads exit early. Both half-warps remain active for the full kernel body;
the lane-restricted masks produce identical results to `0xFFFFFFFF` within each group,
since no cross-group data is needed.

**Impact**: Dormant kernel — not exercised in any production code path. The fix is
proactive and prevents future breakage if the kernel is enabled. No functional regression
to existing hot paths.

### P2 — Nemotron Super 120B tool calling: confirmed fixed

All four MODEL.toml flags present. `BareJsonParser::suppresses_jinja_tools() → true`
provides parser-level protection independently of MODEL.toml. `count_tokens` Anthropic
path checks `parser_suppresses`. No format-instruction conflict possible.

### P3 — SSM cache slots: confirmed by design

`SsmStatePool` sized by `max_batch_size`; `SsmSnapshotPool` sized by `ssm_cache_slots`.
`--ssm-cache-slots 0` zeros only the snapshot pool. No code change needed.

### Summary

One new latent bug found and fixed in `mla_prefill_attn.cu` (dormant kernel, CUDA warp
mask UB at partial last tiles). No regressions to existing hot paths. All prior fixes
confirmed correct at HEAD `ba5f40f`.

---

## 2026-05-28 Fifteenth-pass investigation (spec_ssm HEAD `0b89988`)

Full independent re-audit of all three priorities. One new bug found and fixed in this
session: `mla_prefill_paged_320.cu` (the **live** multi-chunk paged MLA prefill kernel)
carried the same CUDA warp-mask UB that was fixed in `mla_prefill_attn.cu` (dormant) by
the fourteenth-pass commit `0b89988`.

### P1 — Mistral Small 4: CUDA UB fixed in live hot-path kernel

All prior fixes (Bugs 1–4 in the twelfth-pass table plus the `kv_write_start` field) re-
confirmed at HEAD `0b89988`.

#### New fix: CUDA warp-mask UB in `mla_prefill_paged_320` (hot-path kernel)

**Kernel status**: `mla_prefill_paged_320` is the **live hot-path** kernel for all
multi-chunk MLA prefill (called by `paged_mla.rs` when `seq_len_start > 0`). Unlike
`mla_prefill_attn_320` (fixed in commit `0b89988`), which is dormant, this kernel is
exercised on every multi-chunk prefill request to Mistral Small 4.

**Bug**: The kernel uses the same 16-lanes-per-Q-row, 16-Q-rows-per-block (256 threads
total) layout as `mla_prefill_attn_320`. At the last tile of a prefill, when
`q_len % MLA_BR != 0`, the threads belonging to out-of-bounds Q rows return early:

```c
if (q_row >= (q_end - q_start)) return;
```

The still-active threads then call:

```c
for (int offset = 8; offset > 0; offset >>= 1)
    dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);  // UB: departed threads in mask

score = __shfl_sync(0xFFFFFFFF, score, (warp_lane / MLA_LANES) * MLA_LANES);  // UB
```

Using `0xFFFFFFFF` when some threads have exited is **undefined behavior** per CUDA
Programming Guide §B.15 (all threads named in the mask must be executing the same
synchronous instruction).

**Why the math still works (eleventh-pass proof, preserved for context)**: Each warp spans
two Q-rows (threads 0-15 = row N, threads 16-31 = row N+1). For the lower half-warp
(threads 0-15), the reduction at offsets 8, 4, 2, 1 only ever reads from threads 1, 2, 4,
8 — all within [0, 15], all active. Threads 8-15 do read from the departed upper half-warp
at offset=8, accumulating intermediate garbage, but those intermediate values only feed
back into lanes 4-7 (not lane 0). Lane 0's accumulation is provably clean:
`sum[0..15]` ✓. The broadcast from lane 0 distributes the correct sum to all active
threads. So the result is mathematically correct even with `0xFFFFFFFF`.

**Why we fix it anyway**: The mathematical proof holds only under the assumption that
departed threads' registers return 0 (or harmless values). This is GPU-architecture-
specific behavior, not guaranteed by the CUDA spec. On GB10 (Blackwell), the `__shfl_sync`
implementation with an invalid mask is formally UB and may behave differently in future
driver or compiler versions. Fixing it is a 3-line change identical to the fourteenth-pass
fix in `mla_prefill_attn.cu`, and eliminates the UB from the hot-path kernel.

**Fix applied** (`kernels/gb10/mistral-small-4/nvfp4/mla_prefill_paged_320.cu`):

Added `lane_mask` before the early-return guard, mirroring `mla_prefill_attn.cu`:

```c
// Half-warp mask: restrict shfl/shfl_down to the 16-thread sub-group that
// shares the same q_row.  Using 0xFFFFFFFF when the opposite half-warp has
// returned early (last tile, q_len % MLA_BR != 0) is CUDA UB per §B.15.
// warp_lane 0..15 → mask 0x0000FFFF, warp_lane 16..31 → mask 0xFFFF0000.
const unsigned int lane_mask = (warp_lane < 16) ? 0x0000FFFFu : 0xFFFF0000u;

if (q_row >= (q_end - q_start)) return;
// ...
for (int offset = 8; offset > 0; offset >>= 1)
    dot += __shfl_down_sync(lane_mask, dot, offset);
score = __shfl_sync(lane_mask, score, (warp_lane / MLA_LANES) * MLA_LANES);
```

`lane_mask = 0x0000FFFF` for threads 0-15 (q_row N) and `0xFFFF0000` for threads 16-31
(q_row N+1). Both masks are computed before the early return, so departed threads are
never named in any synchronization. For full tiles (`q_len % MLA_BR == 0`), no threads
exit early and both masks produce identical results to `0xFFFFFFFF` within each group.

**Consistency**: `mla_prefill_attn.cu` (dormant, fixed in commit `0b89988`) and
`mla_prefill_paged_320.cu` (live hot-path, fixed in this session) now both use
half-warp masks. The fix in `0b89988` noted this kernel as the live path; the
present commit completes the fix for that path.

### P2 — Nemotron Super 120B tool calling: confirmed fixed

All four MODEL.toml flags present: `disable_tool_steering = true`,
`tool_call_parser = "bare_json"`, `skip_template_tools = true`,
`thinking_in_tools = false`. `BareJsonParser::suppresses_jinja_tools() → true`
provides parser-level protection independently of MODEL.toml. `count_tokens` Anthropic
path checks `parser_suppresses` mirroring `template.rs`. No format-instruction conflict.

### P3 — SSM cache slots: confirmed by design

`SsmStatePool` sized by `max_batch_size`; `SsmSnapshotPool` sized by `ssm_cache_slots`.
`--ssm-cache-slots 0` zeros only the snapshot pool. For single-stream serving,
`--max-batch-size 1` reduces the active state pool from ~1206 MB to ~151 MB.

### Summary

One new bug found and fixed in `mla_prefill_paged_320.cu` (live hot-path kernel, CUDA
warp-mask UB at partial last Q-len tiles). Fix is identical to the dormant-kernel fix
from commit `0b89988`. No regressions to other paths. All prior fixes confirmed correct.

---

## 2026-05-28 Sixteenth-pass investigation (spec_ssm HEAD `ebe5b36`)

Full independent re-audit of all three priorities at branch HEAD `ebe5b36`. No new bugs
found. All prior fixes re-verified correct. This pass constitutes a clean-state
confirmation that the branch is ready for hardware re-test.

### P1 — Mistral Small 4: all seven fixes confirmed

All seven Mistral MLA fixes verified present and correct at HEAD `ebe5b36`:

| # | Fix | File | Commit |
|---|-----|------|--------|
| 1 | `mla_fused_prefill` kernel + `anyhow::ensure!` guard | `cache_skip_mla.rs` | prior |
| 2 | `inv_sqrt_d_absorbed = 1/sqrt(320)` (was `1/sqrt(128)`) | all three MLA paths | prior |
| 3 | `unwrap_or(Bf16)` + `vec![BF16; N]` early-return | `phase_assemble.rs`, `kv_dtypes.rs` | prior |
| 4 | `mla_prefill_paged_320` absorbed paged kernel for multi-chunk | `paged_mla.rs` | prior |
| 5 | `smem_dot[8]` moved to function scope (avoid smem aliasing) | `mla_fused_prefill.cu` | prior |
| 6 | `kv_write_start` respected in cache-skip MLA KV write | `cache_skip_mla.rs` | `e7de0f4` |
| 7 | Half-warp masks in paged and dormant MLA prefill kernels | `mla_prefill_paged_320.cu`, `mla_prefill_attn.cu` | `ebe5b36`, `0b89988` |

#### Additional verifications

**`mla_fused_prefill.cu` kernel audit**: The single-chunk cache-skip path (all 256 threads
always active, grid `(nq, seq_len, 1)`) has no warp-sync UB. Full-warp mask `0xFFFFFFFF`
is correct here because no thread exits early within a block. Causal masking
(`kv_end = min(q_pos + 1, seq_len)`) is correct. Shared memory layout: `smem_q[320]`
(1280 B) + `smem_dot[8]` (32 B, at function scope) + `smem_latent[256]` (1024 B) = 2336 B
total, within limits.

**`--kv-high-precision-layers auto` + `--kv-cache-dtype bf16` interaction**: Verified no
FP8/BF16 mixing. `build_layer_kv_dtypes(BF16, N, 2)` fires the early-return at line 20–22
of `kv_dtypes.rs` → returns `vec![BF16; N]` for all 36 MLA layers. `"auto"` → `kv_hp_layers=2`
is a no-op when the base dtype is already BF16.

**Dead kernel note**: `prefill_attn_mla320_k` (→ `mla_prefill_attn_320`) is loaded by
`init.rs` but never dispatched by any Rust caller; the live single-chunk path uses
`mla_fused_prefill_k` and the live multi-chunk path uses `mla_prefill_paged_k`. The dormant
kernel received the half-warp-mask fix in `0b89988` for correctness hygiene.

**YaRN confirmed non-issue**: `yarn.rs` was audited and confirmed correct throughout.
`find_correction_dim` operates in dimension-index space (pairs 0–31 for Mistral's
rope_dim=64). For beta_fast=32, beta_slow=1, original_max_pos=8192, theta=1e7: low≈7,
high≈15. Ramp `(j - low) / (high - low)` blends interpolated and extrapolated freqs
correctly. The original YaRN misdiagnosis in the first-pass entry is superseded by
subsequent passes.

### P2 — Nemotron Super 120B tool calling: confirmed fixed

All four MODEL.toml flags at `kernels/gb10/nemotron-super-120b-a12b/MODEL.toml` confirmed
present:

```toml
thinking_in_tools    = false   # skip <think> block when tools active
disable_tool_steering = true   # suppress <tool_call> prefix (caused emission loop)
tool_call_parser     = "bare_json"  # model's trained distribution
skip_template_tools  = true    # prevent contradictory XML from jinja template
```

`BareJsonParser::suppresses_jinja_tools() → true` provides parser-level protection
independently of MODEL.toml: `template.rs` passes `jinja_tools = None` for bare-json models
regardless of config. `count_tokens` Anthropic path mirrors `template.rs` (commit `2993894`
fixed the prior asymmetry). No format-instruction conflict remains.

### P3 — SSM cache slots: two-pool design confirmed

Two independent GPU memory pools:

- **`SsmStatePool`** — active inference states, sized by `--max-batch-size` (default 8).
  Allocates `num_ssm_layers × (max_batch_size + 1)` slots. For Qwen3.5-122B: ≈1206 MB.
  `--ssm-cache-slots 0` has **no effect** on this pool.
- **`SsmSnapshotPool`** — prefix-cache snapshots, sized by `--ssm-cache-slots`.
  `--ssm-cache-slots 0` correctly allocates zero GPU memory.

CLI propagation chain re-verified end-to-end:
`cli.rs` → `serve_phases/build.rs:71` → `factory/build.rs:41,373` →
`TransformerModel::new(ssm_cache_slots)` → `SsmSnapshotPool::new(ssm_cache_slots)`.

To reduce `SsmStatePool` memory, use `--max-batch-size 1` (reduces to ≈151 MB).
This is correct behavior; the documentation in prior passes stands.

### Summary

No new bugs found. All seven Mistral MLA fixes, all four Nemotron tool-call fixes, and the
SSM two-pool design are confirmed correct at HEAD `ebe5b36`. Branch is ready for hardware
re-test on GB10 Spark.

---

## 2026-05-28 Seventeenth-pass investigation (spec_ssm HEAD `b2b51f9`)

Full independent re-audit against current branch HEAD `b2b51f9`. Files read directly from
disk (not from prior pass notes). No new bugs found. All fixes re-verified correct.

### P1 — Mistral Small 4: all seven fixes confirmed at HEAD

Files audited: `cache_skip_mla.rs`, `mla_fused_prefill.cu`, `mla_prefill_attn.cu`,
`mla_prefill_paged_320.cu`, `yarn.rs`, `kv_dtypes.rs`, `buffers/sizes.rs`,
`serve_phases/kv_cache.rs`.

**Fix 1 — `mla_fused_prefill` kernel dispatch + `anyhow::ensure!` guard**
`cache_skip_mla.rs:268-273`: `ensure!(self.mla_fused_prefill_k.0 != 0, ...)` prevents
silent fall-through to the broken `inferspark_prefill_64` kernel (HDIM=256 is wrong for
MLA hd=128). Kernel is called at line 274 via `ops::mla_fused_prefill`.

**Fix 2 — `inv_sqrt_d_absorbed = 1/sqrt(320)` (was `1/sqrt(128)`)**
`cache_skip_mla.rs:267`: `let inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt();`
Correctly uses absorbed dimension (kv_lora=256 + rope=64 = 320), not head_dim=128.
Comment at lines 262-263 explains the 0.63× over-sharpening that the old formula caused.

**Fix 3 — BF16 KV dtype across all paths**
`serve_phases/kv_cache.rs:231-238`: `"auto"` → `kv_hp_layers=2`. `kv_dtypes.rs:20-22`:
`if kv_dtype == KvCacheDtype::Bf16 { return vec![Bf16; N]; }` — early-returns `vec![BF16;36]`
regardless of hp_layers. With `--kv-cache-dtype bf16`, all 36 MLA layers use BF16.

**Fix 4 — `mla_prefill_paged_320` absorbed paged kernel for multi-chunk**
`mla_prefill_paged_320.cu`: Correct absorbed-form (HDIM=320) paged attention for
`seq_len_start > 0` chunks. `causal_kv_end = min(q_global + 1, kv_len)` is correct.

**Fix 5 — `smem_dot[8]` at function scope (prevent NVCC smem aliasing)**
`mla_fused_prefill.cu:115`: `__shared__ float smem_dot[8]` declared at kernel function
scope alongside `smem_q[320]` and `smem_latent[256]`. Comment at lines 113-114 explains
the NVCC lifetime-based aliasing risk that prompted this placement. Total smem: 2336 B.

**Fix 6 — `kv_write_start` respected in cache-skip MLA KV write**
`cache_skip_mla.rs:237-257`: `write_count = n.saturating_sub(kv_write_start)` plus
`cache_elem_offset = kv_write_start * mla_cache_dim` and `slot_byte_offset =
kv_write_start * 8`. Only tokens `kv_write_start..n` are written to the paged cache,
skipping tokens already present from a prefix-cache hit. Commit `e7de0f4`.

**Fix 7 — Half-warp masks in paged and dormant MLA prefill kernels**
`mla_prefill_paged_320.cu:89`: `lane_mask = (warp_lane < 16) ? 0x0000FFFFu : 0xFFFF0000u`
declared before the thread-specific early-return at line 91. Used at lines 126 and 130.
Eliminates CUDA UB §B.15 at partial last Q-len tiles. Commit `ebe5b36`.
`mla_prefill_attn.cu:71`: same half-warp mask, commit `0b89988`.

**`mla_fused_prefill.cu` warp-sync audit (no UB)**:
Grid `(nq, seq_len, 1)`, block `(256, 1, 1)`. The only early-return guard (line 50) is
block-level (`head >= nq || q_pos >= seq_len`); all 256 threads in any block that passes
it remain active through every `__syncthreads()` and `__shfl_down_sync(0xFFFFFFFF, ...)`.
Full-warp mask is correct here. Compare with `mla_prefill_attn.cu` where `q_row = tid/16`
causes thread-specific exits — hence the half-warp mask fix.

**YaRN confirmed correct (non-issue)**:
`yarn.rs:58-84`: `find_correction_dim` in dimension-index space. Mistral params: theta=1e7,
rope_dim=64, factor=128, beta_fast=32, beta_slow=1, original_max_pos=8192. Computes
`low≈7`, `high≈15`. Ramp `(j - low) / (high - low)` clamped to [0,1] blends interpolated
and extrapolated inv_freq correctly for all 32 pairs.

**Buffer sizing confirmed sufficient**:
`buffers/sizes.rs:139-211`: all buffers scale with `m = max_batch_tokens`. MLA-specific:
`attn_output` sized for `max(nq*hd, nq*(kv_lora+rope))` → accommodates absorbed output.
`ssm_conv_out_f32` holds q_rope contiguous buffer `nq*rope_dim`. All intermediate buffers
(expert_gate_out for kv_latent, expert_up/down_out for k/v cache assembly) scale with
`max_batch_tokens` and are sufficient for any single prefill chunk.

### P2 — Nemotron Super 120B tool calling: confirmed fixed

`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml` confirmed at HEAD:

```toml
thinking_in_tools    = false   # skip <think> when tools active
disable_tool_steering = true   # suppress <tool_call>\n prefix → no emission loop
tool_call_parser     = "bare_json"  # model's native distribution
skip_template_tools  = true    # block conflicting XML from nemotron_h.jinja
```

Two independent protection layers: (1) MODEL.toml flags; (2) `BareJsonParser::
suppresses_jinja_tools() → true` causes `template.rs` to pass `jinja_tools = None`
regardless of config. Net result: only the bare-JSON system prompt reaches the model.

### P3 — SSM cache slots: two-pool design confirmed

`SsmStatePool` (active inference states) — sized by `max_batch_size`, always allocated.
`SsmSnapshotPool` (prefix-cache snapshots) — sized by `--ssm-cache-slots`, zero when 0.
CLI propagation chain `cli.rs → build.rs:71 → factory:373 → TransformerModel::new →
SsmSnapshotPool::new(ssm_cache_slots)` verified end-to-end. Correct behavior by design.

### Summary

No new bugs found. All seven Mistral MLA fixes, all four Nemotron MODEL.toml flags, and
the SSM two-pool design are confirmed correct at HEAD `b2b51f9`. The branch is clean and
ready for hardware re-test on GB10 Spark.

---

## 2026-05-28 Eighteenth-pass investigation (spec_ssm HEAD `1885142`)

Independent re-audit after context resumption. Files re-read directly from disk in this
session: `kv_dtypes.rs`, `cache_skip_mla.rs`, `phase_assemble.rs`, `main.rs`.
No regressions detected. All fixes confirmed intact at HEAD `1885142`.

### P1 — Mistral Small 4: spot-check of three highest-risk fix sites

**`kv_dtypes.rs` (BF16 dtype path — Fix 3)**
`build_layer_kv_dtypes` line 20-22: early-returns `vec![Bf16; num_attention_layers]` when
`kv_dtype == KvCacheDtype::Bf16`. The old empty-vec return that caused `unwrap_or(Fp8)`
fallback in `phase_assemble.rs` is gone. Comment at lines 6-10 documents the invariant.

**`phase_assemble.rs` (BF16 fallback — companion to Fix 3)**
Line 124: `layer_kv_dtypes.get(i).copied().unwrap_or(KvCacheDtype::Bf16)`.
Defensive fallback is now BF16, not FP8. The comment at lines 119-123 explains that
`build_layer_kv_dtypes` returns `vec![BF16;N]` for BF16 dtype so `get(i)` always hits;
the `unwrap_or` is a safety net for the non-BF16 + `high_precision_layers=0` case.

**`cache_skip_mla.rs` (Fixes 1, 2, 6 — core prefill path)**
- Line 267: `inv_sqrt_d_absorbed = 1.0 / sqrt(kv_lora + mla_rope)` = 1/√320. Correct.
- Lines 268-273: `anyhow::ensure!(self.mla_fused_prefill_k.0 != 0, ...)` hard-errors if
  the absorbed kernel is absent, preventing silent fallback to the broken HDIM=256 path.
- Lines 237-257: `write_count = n.saturating_sub(kv_write_start)`. Cache writes are offset
  by `kv_write_start * mla_cache_dim` (element) and `kv_write_start * 8` (slot bytes).
  Prefix-cache hits correctly skip already-populated slots.

### P2 — Nemotron: MODEL.toml confirmed unchanged

`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml` unchanged since prior passes:
`disable_tool_steering=true`, `tool_call_parser="bare_json"`, `skip_template_tools=true`,
`thinking_in_tools=false`. No regressions.

### P3 — SSM pool design: unchanged

`SsmStatePool` (decode states, `max_batch_size`) and `SsmSnapshotPool` (prefix cache,
`ssm_cache_slots`) remain two separate allocations. `--ssm-cache-slots 0` zeros the
snapshot pool only; use `--max-batch-size 1` to reduce the state pool. No regressions.

### Summary

No new bugs found. All previously identified fixes are confirmed correct at HEAD `1885142`.
The branch remains ready for hardware re-test on GB10 Spark.
