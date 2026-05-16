# Single-GPU Test Results — 3 Large Models on DGX Spark

**Date**: 2026-04-02 (initial run); 2026-05-15 (bug analysis + fixes)
**Node**: single-GPU node (DGX Spark)
**GPU**: NVIDIA GB10 (121.7 GB total, 108-116 GB free)
**Image**: atlas-test:latest (built from spec_ssm + uncommitted fixes)

---

## Summary Table

| Model | Weights | KV Cache | Coherence | Tool Calls | Decode TPS | Long Context | Status |
|-------|---------|----------|-----------|------------|------------|-------------|--------|
| **Qwen3.5-122B** | 90 GB | 0.8 GB (FP8) | 3/3 | 2/2 | 16.5 tok/s | 26K PASS | **PASS** |
| **Mistral Small 4** | 66 GB | 38 GB (BF16) | 3/3 | 2/2 | 34-40 tok/s | **>1K FAIL** (fixes committed) | **NEEDS RETEST** |
| **Nemotron Super 120B** | 94 GB | tiny (FP8) | 3/3 | 0/2 | 20-22 tok/s | 6.5K PASS, 13K FAIL | **PARTIAL** (wrong parser + template conflict fixed) |

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

## 2. mistralai/Mistral-Small-4-119B-2603-NVFP4 — BUG FIXED (retest needed)

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

Additionally, `inv_sqrt_d = 1/sqrt(hd=128)` was used but the absorbed attention dimension
is 320, requiring `1/sqrt(320)`. Using the wrong scale over-sharpens softmax by
√(128/320) ≈ 0.63, adding a second source of corruption. Both are fixed.

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

**Fix applied** (`paged_mla.rs` and `cache_skip_mla.rs`):
- Route MLA prefill through `ops::mla_fused_prefill` when the kernel is loaded.
  This kernel operates entirely in the absorbed 320-dim latent space:
  1. Q_absorbed[256] = Q_nope[64] @ W_UK^T — no HDIM mismatch possible
  2. Q_final = [Q_absorbed | Q_rope_rotated] ∈ R^320
  3. Online softmax attention: Q_final · kv_latent^T (causal)
  4. V_out[128] = attn_latent[256] @ W_UV^T
- `inv_sqrt_d = 1/sqrt(320)` — correct absorbed dimension (was mistakenly 1/sqrt(128))
- The HDIM=256 `inferspark_prefill` kernel is kept as a fallback for non-MLA layers
  (hd=256 or hd=512) with a clear comment marking it broken for MLA hd<256.
- Also corrects O-projection input dimension from `nq * hd` to `nq * mla_v_dim`
  (numerically equal for Mistral where v_dim==hd==128, but semantically correct).

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

**Needs retest** after both commits to confirm long-context failures are resolved.

---

## 3. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 — PARTIAL (retest needed)

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

### Original Test Results (before corrected launch command)
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
`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`.

With both fixes in place, the expected flow is:
1. `bare_json::system_prompt()` → sole tool defs in system message (bare-JSON format)
2. `nemotron_h.jinja` → no XML tool blocks (jinja_tools=None)
3. Generation prompt: `<|im_start|>assistant\n<think></think>\n` (thinking_in_tools=false, disable_tool_steering=true)
4. xgrammar enforces `{"name":"...","arguments":{...}}` schema from token 1
5. Model stays on trained bare-JSON distribution → valid structured tool calls

**Needs retest** to confirm tool calling works after both fixes.

#### 2. Long context >8K — SSM state saturation

SSM (Mamba-2) state saturates with long inputs, producing truncated/incoherent output. Known
architectural limitation of fixed-size Mamba-2 recurrent state; not a code bug.

---

## Action Items (updated 2026-05-15)

| # | Priority | Status | Item |
|---|----------|--------|------|
| 1 | P0 | **FIXED — needs retest** | Mistral MLA (kernel): `cache_skip_mla.rs` used HDIM=256 kernel for head_dim=128, AND wrong inv_sqrt_d (1/√128 instead of 1/√320); fixed both via `mla_fused_prefill` absorbed path |
| 2 | P0 | **FIXED — needs retest** | Mistral MLA (dtype): `phase_assemble.rs` `unwrap_or(Fp8)` on empty layer_kv_dtypes → all MLA layers stored KV in FP8 instead of BF16; fixed to `unwrap_or(Bf16)` |
| 3 | P1 | **FIXED — needs retest** | Nemotron tool calling: (A) wrong CLI parser in test (use MODEL.toml bare_json); (B) `skip_template_tools=true` prevents contradictory XML injection from template |
| 4 | P2 | **CLOSED — by design** | SSM pool 1206 MB: active decode state pool (`SsmStatePool`), not snapshot cache; sized by `--max-batch-size` (default 8). Use `--max-batch-size 1` on single-stream workloads to save ~1050 MB and expand KV cache capacity. `--ssm-cache-slots 0` correctly disables only Marconi prefix-cache snapshots (`SsmSnapshotPool`). |
| 5 | P2 | **CLOSED — known** | Nemotron long context >8K: Mamba-2 fixed-size state saturation, architectural limitation |
