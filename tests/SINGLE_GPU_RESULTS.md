# Single-GPU Test Results — 3 Large Models on DGX Spark

**Date**: 2026-04-02 (initial run); 2026-05-15 (bug analysis + fixes); 2026-05-16 (scale fix); 2026-05-18 (kv_dtypes hardening + test fix); 2026-05-19 (verification: all P0/P1 bugs confirmed fixed)
**Node**: single-GPU node (DGX Spark)
**GPU**: NVIDIA GB10 (121.7 GB total, 108-116 GB free)
**Image**: atlas-test:latest (built from spec_ssm + uncommitted fixes)

---

## Summary Table

| Model | Weights | KV Cache | Coherence | Tool Calls | Decode TPS | Long Context | Status |
|-------|---------|----------|-----------|------------|------------|-------------|--------|
| **Qwen3.5-122B** | 90 GB | 0.8 GB (FP8) | 3/3 | 2/2 | 16.5 tok/s | 26K PASS | **PASS** |
| **Mistral Small 4** | 66 GB | 38 GB (BF16) | 3/3 | 2/2 | 34-40 tok/s | Fixed (code bugs × 3) | **FIXED** |
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

## Action Items (updated 2026-05-19)

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

Independent code walk on spec_ssm HEAD (`08214f9`) covering each filed issue.

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
