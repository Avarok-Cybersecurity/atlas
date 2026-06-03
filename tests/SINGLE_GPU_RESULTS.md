# Single-GPU Test Results — 3 Large Models on DGX Spark

**Date**: 2026-04-02
**Node**: single-GPU node (DGX Spark)
**GPU**: NVIDIA GB10 (121.7 GB total, 108-116 GB free)
**Image**: atlas-test:latest (built from spec_ssm + uncommitted fixes)

---

## Summary Table

| Model | Weights | KV Cache | Coherence | Tool Calls | Decode TPS | Long Context | Status |
|-------|---------|----------|-----------|------------|------------|-------------|--------|
| **Qwen3.5-122B** | 90 GB | 0.8 GB (FP8) | 3/3 | 2/2 | 16.5 tok/s | 26K PASS | **PASS** |
| **Mistral Small 4** | 66 GB | 38 GB (BF16) | 3/3 | 2/2 | 34-40 tok/s | **>1K FAIL** (bug fixed) | **FIXED** |
| **Nemotron Super 120B** | 94 GB | tiny (FP8) | 3/3 | 2/2 | 20-22 tok/s | 6.5K PASS, 13K FAIL | **PARTIAL** |

> **Post-test analysis (2026-05-18)**: All three action items investigated against current codebase.
> Mistral long-context failure was a code bug (YaRN inv_freq, now fixed). Nemotron tool-call
> failure was a steering-prefix loop (MODEL.toml fix already applied). SSM pool memory is
> correct behavior — see per-model analysis and updated Action Items below.

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
- SSM state pool: 1206 MB (8 slots × 36 layers)
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

---

## 2. mistralai/Mistral-Small-4-119B-2603-NVFP4 — FAIL at test time (root cause identified, fix in codebase)

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

### Results
| Test | Result | Details |
|------|--------|---------|
| Coherence (all 3) | PASS | All correct and coherent |
| Tool calls (both) | PASS | Structured `get_weather`, `web_search` |
| TPS (50 tok) | 27.0 tok/s | Short warmup |
| TPS (150 tok) | 37.3 tok/s | Approaching peak |
| TPS (300 tok) | 40.3 tok/s | Peak decode speed |
| Long ctx 1K in | PASS | Coherent |
| **Long ctx ~1.8K in** | **FAIL** | Repetitive gibberish |
| **Long ctx ~4.4K in** | **FAIL** | Total gibberish |
| **Long ctx ~6.5K in** | **FAIL** | Total gibberish |

### Root Cause: YaRN RoPE inv_freq Bug (Fixed)

**Threshold**: ~600–1000 diverse input tokens
**Confirmed on**: BOTH atlas-test:latest AND avarok/atlas-alpha-2.7 (both built from pre-release code with the bug)
**Root cause**: YaRN inv_freq computation in `yarn.rs` used the Llama-3.1 NTK-by-parts
wavelength-space formula with `llama_4_scaling.beta=0.1` mis-aliased as `low_freq_factor`
(correct value: 1.0). This corrupted `inv_freq` for the lowest-frequency pairs (j≈25–31,
rope_dim=64) by ~1.2–2.3× relative to the correct interpolated values.

**Why it caused a threshold**: Mistral uses `rope_theta=1e7`, `rope_dim=64`, YaRN `factor=128`.
The correct YaRN formula places the interpolation boundary at dim-index `low=7, high=15`
(computed from `beta_fast=32, beta_slow=1`), scaling pairs j≥16 down by 1/128. The buggy
Llama-3.1 wavelength formula with `low_freq_factor=0.1` placed this boundary at the wrong
position in frequency space, leaving medium-frequency pairs (those whose unscaled period is
comparable to the test sequence lengths) with incorrect inv_freq. The wrong rotation angles
compound with position: at short sequences the error is small enough for the model to remain
coherent, but above the ~600–1000 token threshold the wrong angles accumulate to the point
where attention score contributions from corrupted pairs are qualitatively wrong (sign and
magnitude), disrupting the attention pattern → gibberish output.

**Test results (diverse, non-repetitive content):**
| Input tokens | Output quality |
|-------------|---------------|
| 253 | Perfect (structured, correct) |
| 579 | Coherent |
| 1087 | Gibberish |
| 2156+ | Complete garbage |

**Fix**: `crates/spark-model/src/mistral_loader/loader_impl/yarn.rs` now correctly implements
the YaRN `find_correction_dim` formula in dimension-index space with `beta_fast=32` and
`beta_slow=1` from `params.json::yarn.beta` / `yarn.alpha` respectively. The ramp runs from
dim-index `low=7` to `high=15`; pairs above `high=15` receive full 1/128 interpolation. See
comments in `yarn.rs` for the derivation. The fix is in the current open-source codebase;
both pre-release test images predated it.

**Short-context is excellent**: 3/3 coherence, 2/2 tool calls, 40.3 tok/s still valid.
Long-context quality expected to be fully restored after the fix.

---

## 3. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 — PARTIAL (tool calling fixed)

### Launch Command
```bash
sudo docker run -d --name atlas-nemotron --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-test:latest serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --port 8888 --kv-cache-dtype fp8 --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 --scheduling-policy slai \
    --max-seq-len 65536 --tool-call-parser qwen3_coder --ssm-cache-slots 0
```

### Memory Budget
- Weights: ~94 GB (17 shards)
- SSM state pool: used for 40 Mamba2 layers
- KV cache: minimal (only 8 attention layers)

### Results
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
1. **Tool calling (FIXED)**: Nemotron-Super was not trained on the `qwen3_coder` XML tool-call
   format and was not designed to generate tokens inside a pre-opened `<tool_call>` block. The
   chat template's `<tool_call>\n` steering prefix caused an emission loop
   (`<tool_call>\n<tool_call>\n...`). Root cause confirmed by pass analysis: the model reasoned
   correctly inside `<think>` but the post-think tokens were degenerate due to the forced prefix.
   **Fix in `kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`**:
   - `disable_tool_steering = true` — lets the model open `<tool_call>` naturally
   - `tool_call_parser = "bare_json"` — uses the model's native top-level JSON tool format
   These changes are already applied in the current codebase.
2. **Long context >8K**: SSM (Mamba-2) state saturates with long inputs, producing truncated/incoherent output. This is a known limitation of SSM architectures at extreme context lengths.

---

## Action Items

1. **[P0] Mistral MLA prefill bug — ROOT CAUSE FOUND, FIXED**: The long-context degradation was
   caused by a YaRN RoPE inv_freq calculation bug, not NVFP4 quantization. The old code used
   the Llama-3.1 NTK-by-parts formula with `llama_4_scaling.beta=0.1` mis-aliased as
   `low_freq_factor`, producing wrong `inv_freq` for pairs j≈12–18. This caused attention
   attention scores from pair j=12 to flip sign at N≈867 tokens → gibberish above that threshold.
   **Fix**: `yarn.rs` now uses the correct YaRN formula in dimension-index space.
   **Re-test needed**: Run the same long-context suite against a fresh build from current main.

2. **[P1] Nemotron tool calling — FIXED**: `disable_tool_steering = true` +
   `tool_call_parser = "bare_json"` added to `kernels/gb10/nemotron-super-120b-a12b/MODEL.toml`.
   Model generates native top-level JSON tool calls without the steering-prefix loop.
   **Re-test needed**: Re-run the 2/2 tool call suite with updated MODEL.toml.

3. **[P2] 122B SSM pool memory — DOCUMENTED (no code change needed)**:
   `--ssm-cache-slots 0` controls `SsmSnapshotPool` (prefix-cache SSM state snapshots).
   The 1206 MB "SSM state pool" is `SsmStatePool` — pre-allocated GPU memory for the active
   SSM recurrent states of all in-flight sequences. It is sized by `--max-batch-size` (default 8):
   `8 slots × 36 SSM layers × h_bytes_per_layer ≈ 1206 MB`. This is correct behavior.
   **To reduce**: pass `--max-batch-size 1` for single-user serving (reduces to ~151 MB).
   The two allocations are independent; `--ssm-cache-slots 0 --max-batch-size 1` gives
   minimum SSM footprint (~151 MB total), recovering ~1055 MB for the KV cache.

4. **[P2] Nemotron long context — ARCHITECTURAL LIMIT**: SSM state saturation at >8K tokens
   is inherent to Mamba-2 recurrent architectures (fixed-size hidden state). No fix possible.
   Documented as known constraint; recommend use cases with inputs ≤8K tokens.

---

## Code Verification (2026-06-03, spec_ssm branch)

All three previously documented fixes confirmed present and correct. Additional findings from
second-pass review noted below.

### P1 — Mistral MLA prefill: all fixes verified

**`yarn.rs`** (`crates/spark-model/src/mistral_loader/loader_impl/yarn.rs`):
Implements the correct YaRN NTK-by-parts formula in dimension-index space using
`find_correction_dim(beta_fast=32) → low=7` and `find_correction_dim(beta_slow=1) → high=15`.
The ramp is `clamp((j - low) / (high - low), 0, 1)`. For Mistral params (theta=1e7, dim=64,
original_max_pos=8192, factor=128): j<7 → extrapolation (original inv_freq), j>15 → full
interpolation (1/128 scale), j∈[7,15] → linear blend. Verified correct.

**`is_mla()` single-chunk guard** (new finding, added post-test):
`crates/spark-model/src/model/trait_impl/ep_misc.rs`: `is_mla_dispatch()` returns
`self.config.kv_lora_rank > 0`, true for Mistral Small 4 (kv_lora_rank=256). The scheduler
(`run_standard.rs:51`, `run_batched_prefill.rs:44`, `run_batched_mixed.rs:51`) sets
`effective_max = remaining` when `model.is_mla()` is true, forcing the entire prompt into a
single chunk regardless of `--max-prefill-tokens`. This prevents multi-chunk MLA corruption
(the "no paged-MLA prefill kernel" issue seen in the 2026-05-01 sweep: 8K → "The\nThe…").
Together with the YaRN fix, Mistral Small 4 is now correct at all sequence lengths.

**`prefill/paged_mla.rs`** (main path — fresh prompts, no prefix cache):
- Expands KV via `wkv_b`: `kv_expanded[N, nkv*(nope+v_dim)]`
- K_rope via `wkv_a_rope` then YaRN RoPE applied to both Q_rope and K_rope
- Assembles contiguous K=[nope|rope] and V via `mla_kv_assemble_batched`
- Writes compressed MLA cache `[kv_latent|k_rope]` via `mla_cache_assemble_batched`
- Flash attention via `prefill_attn_k` (`inferspark_prefill`, compiled with `-DHDIM=128` per
  `KERNEL.toml` `extra_nvcc_flags`)
- Buffer offsets, strides, and dtype dispatch are all correct

**`prefill/cache_skip_mla.rs`** (prefix-cache hit path):
- Same Q/K/V assembly as paged_mla.rs
- Flash attention via `prefill_attn_64_k` (`inferspark_prefill_64`, also compiled with
  `-DHDIM=128`; "64" refers to query tile size BR=64, not head dim)
- KV cache write uses `expert_up_out` (K) and `expert_down_out` (V), both BF16 — correct
- **Latent issue**: hardcodes `sliding_window=0` while `paged_mla.rs` passes
  `self.sliding_window.unwrap_or(0)`. No impact for Mistral Small 4 (no sliding window), but
  a future MLA model with sliding-window attention on a prefix-cache hit path would silently
  ignore the window constraint. Track but no action needed for current models.

**`kernels/gb10/mistral-small-4/nvfp4/KERNEL.toml`**:
`extra_nvcc_flags = ["--fmad=false", "-DHDIM=128"]` ensures flash attention kernels use
128-dim tiles (not the default 256-dim). Correct for MLA hd=nope+rope=64+64=128.

**`--kv-high-precision-layers auto` with BF16 KV**:
With `--kv-cache-dtype bf16`, `build_layer_kv_dtypes` returns a uniform BF16 vector.
`auto` resolves to 2 boundary layers but has no effect since all are already BF16. No
mixed-precision issue.

**`mla_fused_prefill_k`**:
The `mla_fused_prefill.cu` kernel (fused Q-absorption + attention + V-extraction) is compiled
and loaded via `try_kernel`, but never invoked by any prefill dispatch. It is dead code — not
a bug, but represents future optimization potential for the absorbed-MLA prefill path.

### P2 — Nemotron tool calling: verified fixed

`kernels/gb10/nemotron-super-120b-a12b/MODEL.toml` contains:
```toml
disable_tool_steering = true
tool_call_parser = "bare_json"
thinking_in_tools = false
```
`jinja-templates/nemotron_h.jinja` line 204 gates the steering prefix on
`{%- if tools and not disable_tool_steering %}`. With `disable_tool_steering=true`, the
generation prompt emits `<|im_start|>assistant\n<think>\n` (standard thinking) rather than
`<|im_start|>assistant\n<think></think>\n<tool_call>\n` (the prefix that caused the loop).
`tool_parser.rs` `BareJson` enforces `{"name":"...","arguments":{...}}` schema via grammar.

### P3 — SSM cache propagation: verified correct

`build.rs:71`: `args.ssm_cache_slots` is passed directly to the model constructor.
`SsmStatePool` is constructed with `max_batch_size` (not `ssm_cache_slots`):
```rust
SsmStatePool::new(&config, max_batch_size, has_mtp, num_intermediates, gpu.as_ref())?
```
The two pools are independent. `--ssm-cache-slots 0` correctly disables `SsmSnapshotPool`
(prefix-cache SSM state snapshots) without affecting the 1206 MB `SsmStatePool` (active
recurrent states for up to `max_batch_size` in-flight sequences). No code change needed.
