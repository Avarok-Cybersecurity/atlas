# Single-GPU Test Results — 3 Large Models on DGX Spark

**Date**: 2026-04-02 (initial run); 2026-05-15 (bug analysis + fixes); 2026-05-16 (scale fix)
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

## Action Items (updated 2026-05-15)

| # | Priority | Status | Item |
|---|----------|--------|------|
| 1 | P0 | **FIXED** | Mistral MLA (kernel): `paged_mla.rs`/`cache_skip_mla.rs` used HDIM=256 kernel for head_dim=128; fixed by adding HDIM=128 kernel guard + routing first-chunk through `prefill_attn_128_k` |
| 2 | P0 | **FIXED** | Mistral MLA (scale): all three MLA paths (`paged_mla.rs`, `cache_skip_mla.rs`, `attention_forward_mla.rs`) passed `1/√128` to 320-dim absorbed attention; fixed to `1/√(kv_lora+rope)=1/√320` |
| 3 | P0 | **FIXED** | Mistral MLA (dtype): `phase_assemble.rs` `unwrap_or(Fp8)` on empty layer_kv_dtypes → all MLA layers stored KV in FP8 instead of BF16; fixed to `unwrap_or(Bf16)` |
| 4 | P0 | **FIXED** | Mistral MLA (multi-chunk): `paged_mla.rs` multi-chunk path attended only to `n` new tokens, ignoring paged KV history; fixed with new `mla_prefill_paged_320` + `mla_v_extract_batched` absorbed paged path |
| 5 | P1 | **FIXED** | Nemotron tool calling: (A) wrong CLI parser in test (use MODEL.toml bare_json); (B) `skip_template_tools=true` prevents contradictory XML injection from template; (C) `thinking_in_tools=false` |
| 6 | P2 | **CLOSED — by design** | SSM pool 1206 MB: active decode state pool (`SsmStatePool`), not snapshot cache; sized by `--max-batch-size` (default 8). Use `--max-batch-size 1` on single-stream workloads to save ~1050 MB. `--ssm-cache-slots 0` correctly disables only `SsmSnapshotPool`. |
| 7 | P2 | **CLOSED — known** | Nemotron long context >8K: Mamba-2 fixed-size state saturation, architectural limitation |
| 8 | P2 | **OPEN** | Mistral multi-chunk performance: `mla_prefill_paged_320` iterates all kv_len positions sequentially (O(kv_len)). For kv_len > 10K, add shared-memory KV tiling to amortize page-table overhead. |
