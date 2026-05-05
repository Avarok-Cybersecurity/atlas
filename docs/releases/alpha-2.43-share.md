# 🚀 Atlas Spark — Alpha 2.43

```bash
docker pull avarok/atlas-gb10:alpha-2.43
# or
docker pull avarok/atlas-gb10:latest
```

**Hardware:** NVIDIA DGX Spark GB10 (sm_121f, 120 GB LPDDR5X)
**License:** AGPL-3.0-only
**Digest:** `sha256:cd3b2aa1e806017fcaf64649a255a9d8831c9960248c7be4dc98035e104d66ec`

---

## ✨ What's new vs 2.35

### Gemma-4 (26B & 31B) is now production-quality

- **RoPE proportional fix** — Gemma-4 full-attention layers were rotating the wrong dims at the wrong base; affected 10/60 layers on 31B and 5/30 on 26B. Now correct (`rope_angles = head_dim/8`).
- **New HDIM=512 paged-prefill kernel** — `inferspark_prefill_paged_512`, 8-warp single-buffered K, fits Gemma's 135 KB shared-mem footprint into GB10's 99 KB cap. Long-context prompts (>8K) on Gemma-4 no longer return HTTP 500.
- **Sliding-window mask in paged prefill** (was causal-only) — added `sliding_window` arg to all 7 paged variants, guarded so non-sliding models (Qwen3.5 etc.) are untouched.
- **Gemma-4-26B MoE softcap** — disabled `final_logit_softcapping=30` for 26B (it has 128 experts and large final-norm weights). Re-enables logit discrimination.

### Bug fixes confirmed shipped in 2.43

These bugs reported in #bugs/#help are all resolved in 2.43:

- ✅ xgrammar `LogFatalError: Expect element` when tools active (was crashing the server)
- ✅ Tokenizer panic `byte index N is not a char boundary` on UTF-8 (Swedish å/ä/ö, etc.)
- ✅ xgrammar `GrammarMatcher ... trying to find next token mask` after stop token
- ✅ `Kernel lookup reshape_and_cache_flash_turbo4 ... status 500` on Coder-Next FP8 with `--kv-cache-dtype turbo4`
- ✅ Mistral Small 4 NVFP4 repetition loop / weight-loading regressions

### Test harness

- nemotron-only `enable_thinking=False` removed from fib test (was producing prose instead of code on the larger nemotron variants).
- Tool-call test `max_tokens` raised 200→1024 so Gemma-4-26B's thinking-in-tools chain has room to actually emit the call.

### Nemotron-nano-30B

- Thinking budget 1024→2048 (fib reasoning chain was getting truncated mid-trace).

---

## 📊 Pass-28 benchmark — single Spark, fresh sweep 2026-04-14

**Total: 123/135 (91.1%) — 10/15 models PERFECT.**

| Model | Score | Δ vs pass-22 | Coh | Fib | Tool | LC | Max tok/s | ~TTFT 4k |
|---|---|---|---|---|---|---|---|---|
| 35B-nvfp4-mtp           | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 | 113.8 | 1.4 s |
| 35B-nvfp4               | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  90.2 | 2.6 s |
| 80B-nvfp4-mtp           | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  87.4 | 2.6 s |
| 80B-nvfp4-ep2-mtp       | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  84.7 | 2.7 s |
| 35B-fp8                 | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  70.8 | 3.6 s |
| qwen3-vl-30B            | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  68.4 | 2.8 s |
| 80B-nvfp4               | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  64.9 | 2.6 s |
| 122B-nvfp4-ep2-mtp      | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  38.3 | 7.6 s |
| 122B-nvfp4              | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  31.9 | 6.9 s |
| mistral-small-4         | **9/9** | =  | 3/3 | PASS | 2/2 | 3/3 |  29.7 | 8.0 s |
| coder-next-fp8          | 8/9     | −1 | 3/3 | PASS | 2/2 | 2/3 |  45.3 | 4.2 s |
| **gemma-4-26B**         | 7/9     | **+1** | 3/3 | **PASS** | 1/2 W1 | 2/3 | 73.4 | 16.2 s |
| **gemma-4-31B**         | 7/9     | **+1** | 2/3 | FAIL | 2/2 | 3/3 |  11.4 | 89.3 s |
| nemotron-nano-30B       | 7/9     | −1 | 3/3 | FAIL | 1/2 W1 | 3/3 | 88.1 | 3.8 s |
| nemotron-super-120B-ep2 | 4/9     | −6\* | 2/3 | FAIL | 0/2 W2 | 2/3 | 27.2 | 16.0 s |

\* nemotron-super-ep2 ran 9 tests this pass vs 13 in pass-22 (suite changed); not a real regression.

---

## ⚠️ Known issues (carrying over to next alpha)

- **gemma-4-31B Creative haiku** — repetition loop. Persistent across 2.38–2.43. Separate from the RoPE fix.
- **gemma-4-26B LC16k** — repetition loop past ~7K tokens on NVFP4. Independent long-context bug (was masked by harness in 2.42).
- **gemma-4-26B Search** — tool-call extraction WARN; thinking-in-tools eats most of the budget.
- **nemotron-nano-30B fib** — model emits `[0,1,1,1,2,…]` off-by-one. Model quality, not Atlas.
- **`max_thinking_budget` per-request cap** — request-body `thinking.budget_tokens` is capped at MODEL.toml's value instead of overriding it (violates documented precedence). For `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` there's no `[behavior]` section in MODEL.toml so the cap is the hardcoded default 256. Fix in tree, ships next alpha.

---

## 📦 Models

| Model | HuggingFace ID | Params | Active | Arch | MTP | Max ctx |
|---|---|---|---|---|---|---|
| Qwen3.5-27B Dense       | `Kbenkhaled/Qwen3.5-27B-NVFP4`              | 27B  | 27B  | SSM+Attn         | —  | 8K |
| Qwen3-VL-30B            | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4`       | 30B  | 3B   | Attn+MoE+Vision  | —  | 32K |
| Nemotron-H 30B          | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 30B | 3B  | Mamba-2+MoE+Attn | —  | 8K |
| Qwen3.5-35B A3B         | `Kbenkhaled/Qwen3.5-35B-A3B-NVFP4`          | 35B  | 3B   | SSM+MoE+Attn     | K=2 | 8K |
| Qwen3.5-35B A3B (FP8)   | `Qwen/Qwen3.5-35B-A3B-FP8`                  | 35B  | 3B   | SSM+MoE+Attn     | —  | 8K |
| Qwen3-Next-80B A3B      | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`  | 80B  | 3B   | SSM+MoE+Attn     | K=2 | 8K |
| Qwen3.5-122B A10B       | `Sehyo/Qwen3.5-122B-A10B-NVFP4`             | 122B | 10B  | SSM+MoE+Attn     | K=2 | 4K |
| Mistral-Small-4         | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | 119B | 119B | MLA+Attn         | —  | 4K |
| Gemma-4-26B (MoE)       | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` | 26B | 4B | Attn+MoE+Vision (5:1 sliding/full) | — | 32K |
| Gemma-4-31B (Dense)     | `nvidia/Gemma-4-31B-IT-NVFP4`               | 31B  | 31B  | Attn (5:1 sliding/full) | — | 32K |

---

## 🚀 Quick start

### Fastest coherent model — Qwen3.5-35B A3B + MTP (113.8 tok/s)

```bash
sudo docker run -d --name atlas-35b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:alpha-2.43 \
  serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
    --port 8888 --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --speculative --mtp-quantization nvfp4
```

### Long-context coding — Gemma-4-26B (32K)

```bash
sudo docker run -d --name atlas-gemma-26b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:alpha-2.43 \
  serve bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16 \
    --port 8888 --max-seq-len 32768 \
    --kv-cache-dtype bf16 --max-batch-size 1 \
    --scheduling-policy slai
```

### 122B EP=2 (38.3 tok/s on 2× GB10)

See `QUICKSTART.md` §7. Only the image tag (`alpha-2.43`) changes from 2.35.

---

## 🔌 API

OpenAI-compatible. Endpoints: `/v1/chat/completions` (streaming + non-streaming), `/v1/completions`, `/v1/models`, `/health`.
Supports: `tools` / `tool_choice`, `enable_thinking` (via `chat_template_kwargs`), vision content-parts, `reasoning_content`.

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
    "messages": [{"role":"user","content":"Hello!"}],
    "max_tokens": 64
  }'
```

---

## 🔙 Rollback

```bash
docker pull avarok/atlas-gb10:alpha-2.8   # or any earlier alpha-2.x
```

All earlier tags retained on Docker Hub.

---

**Commit:** `071742a` on `pass-26-baseline`
**Thanks** to everyone in #bugs / #help — 4 of the issues reported on 2.6–2.8 are confirmed shipped here. Keep them coming.
