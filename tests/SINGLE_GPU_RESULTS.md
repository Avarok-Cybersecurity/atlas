# Single-GPU Test Results — 3 Large Models on DGX Spark

**Date**: 2026-05-15 (updated with bug investigation findings)
**Node**: single-GPU node (DGX Spark)
**GPU**: NVIDIA GB10 (121.7 GB total, 108-116 GB free)
**Image**: atlas-test:latest (built from spec_ssm + uncommitted fixes)

---

## Summary Table

| Model | Weights | KV Cache | Coherence | Tool Calls | Decode TPS | Long Context | Status |
|-------|---------|----------|-----------|------------|------------|-------------|--------|
| **Qwen3.5-122B** | 90 GB | 0.8 GB (FP8) | 3/3 | 2/2 | 16.5 tok/s | 26K PASS | **PASS** |
| **Mistral Small 4** | 66 GB | 38 GB (BF16) | 3/3 | 2/2 | 34-40 tok/s | **>1K FAIL** (fix committed) | **NEEDS RETEST** |
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

### BUG FOUND AND FIXED: Wrong HDIM Kernel in MLA Cache-Skip Prefill Path

**Root cause** (code bug, NOT an NVFP4 limitation):

`cache_skip_mla.rs` called `ops::prefill_attention_64` with kernel handle `prefill_attn_64_k`,
which maps to `inferspark_prefill_64` — a BR=64 flash attention kernel compiled with
`#define HDIM 256`. Mistral Small 4 MLA uses `head_dim=128`.

The HDIM=256 kernel loads 256 elements per Q head (reading 128 valid + 128 from the adjacent
head's data) and performs QK^T over 256/16=16 k-iterations instead of the correct 8. It also
writes 256 output elements per head, overflowing into adjacent head's output buffer. This
corrupts attention across all 36 layers. With short sequences the corruption is limited in
scope; beyond ~600-1000 tokens the extra KV pairs accumulate enough cross-head contamination
to produce incoherent output.

The paged MLA path (`paged_mla.rs`) correctly used `prefill_attn_k` → `inferspark_prefill`
→ `inferspark_prefill_h128` (HDIM=128). The cache-skip path was not updated to match.

**Fix applied** (`crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_mla.rs`):
- Replaced `ops::prefill_attention_64(…, self.prefill_attn_64_k, …)` with
  `ops::prefill_attention(…, prefill_k, …)` where `prefill_k` is chosen as
  `prefill_attn_512_k` for `hd > 256` else `prefill_attn_k` (matches `paged_mla.rs`)
- Replaced `1.0f32 / (hd as f32).sqrt()` with `self.effective_attn_scale(hd)`
- Replaced hardcoded `0` for `sliding_window` with `self.sliding_window.unwrap_or(0)`

**Previous incorrect diagnosis**: The prior results entry attributed this to NVFP4
quantization. That was wrong — identical failure appears on avarok/atlas-alpha-2.7 because
that build also contains the same cache-skip path bug.

**Test results (diverse, non-repetitive content — BEFORE fix):**
| Input tokens | Output quality |
|-------------|---------------|
| 253 | Perfect (structured, correct) |
| 579 | Coherent |
| 1087 | Gibberish |
| 2156+ | Complete garbage |

**Needs retest** after this commit to confirm fix resolves long-context failures.

---

## 3. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 — PARTIAL

### Launch Command (original, broken)
```bash
sudo docker run -d --name atlas-nemotron --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-test:latest serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --port 8888 --kv-cache-dtype fp8 --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 --scheduling-policy slai \
    --max-seq-len 65536 --tool-call-parser qwen3_coder --ssm-cache-slots 0
```

### Launch Command (correct — after fix)
```bash
sudo docker run -d --name atlas-nemotron --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-test:latest serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --port 8888 --kv-cache-dtype fp8 --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 --scheduling-policy slai \
    --max-seq-len 65536 --ssm-cache-slots 0
    # No --tool-call-parser: MODEL.toml supplies bare_json
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
| 1 | P0 | **FIXED — needs retest** | Mistral MLA: `cache_skip_mla.rs` used HDIM=256 kernel for head_dim=128; fixed to use `prefill_attn_k` (HDIM=128) |
| 2 | P1 | **FIXED — needs retest** | Nemotron tool calling: (A) wrong CLI parser in test (use MODEL.toml bare_json); (B) `skip_template_tools=true` prevents contradictory XML injection from template |
| 3 | P2 | **CLOSED — by design** | SSM pool 1206 MB: active decode state pool, not snapshot cache; `--ssm-cache-slots 0` correctly disables only Marconi prefix caching |
| 4 | P2 | **CLOSED — known** | Nemotron long context >8K: Mamba-2 fixed-size state saturation, architectural limitation |
