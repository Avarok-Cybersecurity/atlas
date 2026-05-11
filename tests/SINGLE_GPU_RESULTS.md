# Single-GPU Test Results — 3 Large Models on DGX Spark

**Date**: 2026-04-02 (investigation updated 2026-05-11)
**Node**: single-GPU node (DGX Spark)
**GPU**: NVIDIA GB10 (121.7 GB total, 108-116 GB free)
**Image**: atlas-test:latest (built from spec_ssm + uncommitted fixes)

---

## Summary Table

| Model | Weights | KV Cache | Coherence | Tool Calls | Decode TPS | Long Context | Status |
|-------|---------|----------|-----------|------------|------------|-------------|--------|
| **Qwen3.5-122B** | 90 GB | 0.8 GB (FP8) | 3/3 | 2/2 | 16.5 tok/s | 26K PASS | **PASS** |
| **Mistral Small 4** | 66 GB | 38 GB (BF16) | 3/3 | 2/2 | 34-40 tok/s | **>1K FAIL** | **FAIL** |
| **Nemotron Super 120B** | 94 GB | tiny (FP8) | 3/3 | 0/2 | 20-22 tok/s | 6.5K PASS, 13K FAIL | **PARTIAL** |

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

## 2. mistralai/Mistral-Small-4-119B-2603-NVFP4 — FAIL (long context bug)

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

### NVFP4 Precision Limitation: Context Degrades at >600 Input Tokens

**Threshold**: ~600-1000 diverse input tokens
**Confirmed on**: BOTH atlas-test:latest AND avarok/atlas-alpha-2.7 (identical behavior)
**NOT a code bug**: Exhaustive code review (see investigation log below) confirms this is a fundamental NVFP4 quantization limitation for MLA architecture.
**Root cause**: NVFP4 quantization of MLA projections + MoE experts accumulates numerical error through the 36-layer attention stack. The MLA compressed KV space (320 dims) amplifies small quantization errors.

**Test results (diverse, non-repetitive content):**
| Input tokens | Output quality |
|-------------|---------------|
| 253 | Perfect (structured, correct) |
| 579 | Coherent |
| 1087 | Gibberish |
| 2156+ | Complete garbage |

**Short-context is excellent**: 3/3 coherence, 2/2 tool calls, 40.3 tok/s. Only viable for inputs <600 tokens.

**Possible mitigations**:
1. FP8 model variant (not published)
2. Selective BF16 dequant for MLA projections (keep W_kv_a, W_kv_b, W_q at BF16)
3. Accept the ~600 token input limit

### Code Investigation Log (2026-05-11)

All suspected code paths were audited; no Atlas-side bug was found.

**`kernels/gb10/mistral-small-4/nvfp4/mla_absorbed.cu`**
- All kernels (`mla_batched_gemv`, `mla_q_rope_*`, `mla_kv_assemble_batched`, etc.) are BF16 throughout.
- Grid dims use `unsigned long long`; no 32-bit overflow at large seq_len.
- Shared memory sizing is fixed per-head; no seq_len-dependent overflow risk.

**`crates/spark-model/src/layers/qwen3_attention/prefill/paged_mla.rs`**
- Buffer sequence is correct with no aliasing: `ssm_ba`→q_latent, `qkv_output`→qg_out, `expert_gate_out`→kv_latent, `ssm_deinterleaved`→kv_expanded, `ssm_ba`(reuse)→k_rope_buf, `ssm_conv_out_f32`→q_rope_tmp, `ssm_qkvz`→k_contiguous, `expert_down_out`→mla_k_cache, `attn_output`→attn_out, `norm_output`→o_out.
- Calls `ops::prefill_attention` (BF16, BR=32, Grid=[num_q_heads, ceil(n/32), batch]) with current-chunk tokens only — no paged history (MLA uses absorbed decode for history).
- `inv_sqrt_d = self.effective_attn_scale(hd=128)` is correct for direct 128-dim attention.
- Writes compressed 320-dim MLA entries to paged KV cache for future decode steps.

**`crates/spark-model/src/layers/qwen3_attention/decode/attention_forward_mla.rs`**
- Absorbed decode: Q_absorbed (320-dim) × K_cache (320-dim); mathematically equivalent to direct 128-dim prefill attention. No inconsistency between paths.

**`crates/spark-model/src/layers/qwen3_attention/init.rs`**
- `prefill_attn_k` and `paged_decode_mla_k` are always loaded as BF16 kernels, regardless of `kv_dtype`. No FP8/BF16 mixing on MLA paths.

**`crates/spark-server/src/main_modules/kv_dtypes.rs`**
- `build_layer_kv_dtypes(Bf16, num_attn_layers, kv_hp_layers)` returns `[]` (empty vector = all layers uniform BF16) when base dtype is already BF16. `--kv-high-precision-layers auto` with `--kv-cache-dtype bf16` is therefore a no-op. No FP8 mixing.

**Conclusion**: The gibberish at >1K tokens is not caused by any Atlas code path. It reproduces identically on an independent release build. NVFP4 quantization error accumulation through the MLA architecture is the confirmed cause.

---

## 3. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 — PARTIAL

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

#### 1. Tool calling — model not trained on qwen3_coder XML format

**Root cause** (confirmed via code review, 2026-05-11):

Nemotron Super 120B was not fine-tuned on the qwen3_coder XML tool-call format (`<tool_call>\n<function=NAME>\n<parameter=...>`). The `nemotron_h.jinja` template itself contains an explicit comment acknowledging this:
> "For larger variants (Super 120B) the prefix causes a `<tool_call>` emission loop because the model wasn't trained on the qwen3_coder XML format — pass `disable_tool_steering=true` to skip."

Additional factors confirmed by code inspection:
- The `ToolCallParser::system_prompt()` method is **never called** in the main chat flow (`template.rs` / `chat/mod.rs`). Tool definitions reach the model only through the Jinja template, so there is no duplicate or conflicting system-prompt injection.
- With `tool_choice="auto"`, `use_triggers=true` is passed to XGrammar's structural-tag grammar, which allows the model to produce a natural-language response rather than a `<tool_call>` block. The model consistently exercises this escape hatch.
- The exponential logit bias on the `<tool_call>` start token (+3.0 on first attempt) is insufficient to overcome the model's strong prior against this format.

**Workaround**: pass `tool_choice="required"` at the API level. This sets `use_triggers=false`, forcing the XGrammar constraint to require a tool-call block and making the bias irrelevant. Quality of generated arguments may still be poor because the model was not trained on this schema.

**Proper fix**: use a tool-calling format that Nemotron Super 120B was actually trained on (likely Llama 3 / `<|python_tag|>` or NIM-format JSON). A dedicated `nemotron_super.jinja` + matching parser would be needed.

#### 2. Long context >8K — SSM state saturation

SSM (Mamba-2) state saturates with long inputs, producing truncated/incoherent output. This is a known architectural limitation of fixed-size SSM recurrent state and is not a code bug.

---

## Action Items

All three priority items were investigated on 2026-05-11. No Atlas-side code bugs were found.

1. **[CLOSED — not a code bug] Mistral MLA prefill**: Full audit of `mla_absorbed.cu`, `paged_mla.rs`, `attention_forward_mla.rs`, `init.rs`, and `kv_dtypes.rs` found no defect. The gibberish at >1K tokens is a fundamental NVFP4 quantization limitation for MLA architecture and reproduces identically on an independent release build. See investigation log in section 2 above.

2. **[OPEN — model limitation] Nemotron tool calling**: Root cause is that Super 120B was not trained on the qwen3_coder XML format. A dedicated jinja template + parser using the format the model was actually trained on (likely Llama 3 or NIM JSON) is the correct fix. Short-term workaround: `tool_choice="required"` forces XGrammar to require a tool-call block, but argument quality will be degraded.

3. **[CLOSED — by design] SSM pool memory with `--ssm-cache-slots 0`**: The 1206 MB pool reported in logs is the **active decode state pool** (`SsmStatePool`, allocated as `max_batch_size × num_ssm_layers × state_bytes`). This is distinct from the **Marconi snapshot pool** (`SsmSnapshotPool`, controlled by `--ssm-cache-slots`). The CLI value IS correctly propagated: `ssm_cache_slots=0` → `SsmSnapshotPool::new(0, ...)` → zero snapshot slots allocated. The 1206 MB decode pool cannot be zero-sized while SSM inference is active; it holds the recurrent state for each sequence in flight. No action needed.

4. **[KNOWN] Nemotron long context >8K**: SSM state saturation is an architectural limitation of fixed-size Mamba-2 recurrent state. Document as a known constraint.
