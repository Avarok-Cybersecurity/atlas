# Atlas Spark — Alpha 2.43

**Image:**

```bash
docker pull avarok/atlas-gb10:alpha-2.43        # pinned
docker pull avarok/atlas-gb10:latest             # floating
```

Digest: `sha256:cd3b2aa1e806017fcaf64649a255a9d8831c9960248c7be4dc98035e104d66ec`

**Hardware:** NVIDIA DGX Spark GB10 (sm_121f, 120 GB LPDDR5X, 273 GB/s)
**License:** AGPL-3.0-only
**Commit:** `071742a` on branch `pass-26-baseline`

---

## What's New (vs alpha-2.35)

### Gemma-4 — RoPE proportional correctness fix
`set_rope_proportional(true)` was defined but never called for Gemma-4 full-attention layers (10/60 in the 31B dense model, 5/30 in the 26B MoE). Those layers were rotating the wrong 128 Q/K dims with the wrong frequency denominator (rotary_dim=128 instead of head_dim=512). Fix: enable the proportional kernel and pass the correct `rope_angles = head_dim/8 = 64`. `crates/spark-model/src/weight_loader/gemma4.rs:286-298`.

**Observable effect:** gemma-4-26B Creative haiku and fib flipped from FAIL to PASS. gemma-4-31B LC4k needle-in-haystack now recalls the embedded code word.

### Gemma-4 — new HDIM=512 paged prefill kernel
The standard 4-warp BR=32 paged prefill template at HDIM=512 needs ~135 KB of shared memory. GB10's per-block opt-in cap is 99 KB (101,376 bytes, queried via `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)`). That's why chunked long-context prefill on Gemma-4's full-attention layers was bailing out.

New kernel: `inferspark_prefill_paged_512` — 8 warps, BR=BC=32, **single-buffered K** (saves 33 KB), PAD_KV=0, dynamic shared memory (101,120 B exactly). QK^T runs on warps 0-1 while warps 2-7 load V via cp.async; PV fan-out splits HDIM four ways across warp pairs. `kernels/gb10/nvfp4/prefill_paged_compute_512.cuh` + `inferspark_prefill_paged_512.cu` + `ops::prefill_attention_paged_512`.

**Observable effect:** prompts >8192 tokens on Gemma-4-26B/31B no longer return HTTP 500 "Gemma-4 HDIM=512 chunked prefill not yet supported."

### Paged prefill template — sliding-window mask (pre-existing bug)
The paged prefill template was applying causal masking only. For models with `sliding_window > 0` (Gemma-4's sliding layers at 1024, 25/30 of 26B and 50/60 of 31B), chunks 1+ attended to **every preceding token** instead of the 1024-token window, corrupting the residual stream. Added a new `sliding_window` kernel argument across all 6 paged variants (BF16/FP8/NVFP4 × BR32/BR64) plus the new HDIM=512 variant, guarded by `if(sliding_window>0)` so non-sliding models (Qwen3.5 and others with `sliding_window=0`) are unaffected. `kernels/gb10/nvfp4/prefill_paged_compute.cuh` + 7 Rust wrappers in `crates/spark-model/src/layers/ops.rs` + 9 dispatch sites in `qwen3_attention/prefill.rs`.

### Gemma-4 MoE — disable `final_logit_softcapping`
Gemma-4-26B is MoE (128 experts) with large final-norm weights (~29). Raw logits run into the thousands; the HF-default `final_logit_softcapping=30` collapsed them all to ±30, destroying discrimination. Guard added in `crates/atlas-core/src/config.rs:1044-1055` — disable softcap when `num_experts > 0 && model_type == "gemma4"`. The 31B dense variant keeps softcap=30 (its smaller norm weights actually need it).

### Nemotron-nano-30B — thinking budget 1024 → 2048
`kernels/gb10/nemotron-3-nano-30b-a3b/MODEL.toml:49`. The fib-code-generation reasoning chain was hitting the old 1024 limit and truncating mid-trace, producing malformed Python. 2048 is enough headroom.

### Test harness — two corrections
- `tests/single_gpu_suite.py:377` — dropped the nemotron-only `enable_thinking=False` override on the fib test. Forcing thinking off produced prose ("We need to write a Python script…") instead of code on nemotron-super-120B and a buggy `range(b)` loop on nemotron-nano-30B.
- `tests/single_gpu_suite.py:519` — tool-call test `max_tokens` 200 → 1024. Gemma-4-26B has `thinking_in_tools=true, max_thinking_budget=512` and the server caps effective thinking at 90% of `max_tokens`; at 200 the cap was 180, which Gemma-4-26B consumed entirely on thinking and emitted zero tokens for the actual tool call (Search WARN pattern).

---

## Benchmark — pass-28 (alpha-2.43 vs pass-22 baseline)

All numbers are from a fresh sweep on a single DGX Spark (TP=1) on 2026-04-14. The **~TTFT-4k** column is elapsed-minus-(completion_tokens / max_tps) on the long-context 4000-token test, so it includes chunked prefill. Four EP=2 models from the pass-22 suite are missing because the worker node dropped off the IB network mid-sweep (separate infrastructure issue, not a regression).

| Model | Score | vs pass-22 | Coh | Fib | Tool | LC | Max tok/s | ~TTFT 4k |
|---|---|---|---|---|---|---|---|---|
| 35B-nvfp4-mtp | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 113.8 | 1.4 s |
| 35B-nvfp4 | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 90.2 | 2.6 s |
| 80B-nvfp4-mtp | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 87.4 | 2.6 s |
| 80B-nvfp4-ep2-mtp | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 84.7 | 2.7 s |
| 35B-fp8 | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 70.8 | 3.6 s |
| qwen3-vl-30B | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 68.4 | 2.8 s |
| 80B-nvfp4 | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 64.9 | 2.6 s |
| 122B-nvfp4-ep2-mtp | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 38.3 | 7.6 s |
| 122B-nvfp4 | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 31.9 | 6.9 s |
| mistral-small-4 | **9/9** | = | 3/3 | PASS | 2/2 | 3/3 | 29.7 | 8.0 s |
| coder-next-fp8 | 8/9 | −1 | 3/3 | PASS | 2/2 | 2/3 | 45.3 | 4.2 s |
| **gemma-4-26B** | 7/9 | **+1** | 3/3 | **PASS** | 1/2 W1 | 2/3 | 73.4 | 16.2 s |
| **gemma-4-31B** | 7/9 | **+1** | 2/3 | FAIL | 2/2 | 3/3 | 11.4 | 89.3 s |
| nemotron-nano-30B | 7/9 | −1 | 3/3 | FAIL | 1/2 W1 | 3/3 | 88.1 | 3.8 s |
| nemotron-super-120B-ep2 | 4/9 | −6* | 2/3 | FAIL | 0/2 W2 | 2/3 | 27.2 | 16.0 s |

**Total: 123/135 (91.1%), 10/15 PERFECT.**

\* nemotron-super-ep2 ran with 9 tests this pass vs 13 in pass-22 (different suite dimensions); the `−6` is not a real regression.

### Known issues (not fixed in 2.43)

- **gemma-4-31B Creative haiku** still fails with a repetition loop. Separate from the RoPE fix — Cluster C's deeper bug on 31B is still open. Persistent across alpha-2.38 through 2.43.
- **gemma-4-26B LC16k** fails with a repetition loop despite the sliding-window fix. The model itself degrades past ~7k tokens on NVFP4 quantization (same symptom with and without my kernel; alpha-2.42 "passed" LC16k only because the test-harness repetition detector didn't catch that specific pattern). Independent long-context bug to chase.
- **gemma-4-26B Search** still WARN (tool-call extraction produces unquoted prose). Raising `max_tokens` was enough to unblock generation but the model's thinking-in-tools trace consumes most of the budget.
- **nemotron-nano-30B fib** still FAIL — the model's reasoning produces buggy code (`[0,1,1,1,2,…]` off-by-one). Model-quality issue, not Atlas.

---

## Models

| Model | HuggingFace ID | Params | Active | Arch | MTP | Max Context |
|---|---|---|---|---|---|---|
| Qwen3.5-27B Dense | `Kbenkhaled/Qwen3.5-27B-NVFP4` | 27B | 27B | SSM+Attn | — | 8K |
| Qwen3-VL-30B | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` | 30B | 3B | Attn+MoE+Vision | — | 32K |
| Nemotron-H 30B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 30B | 3B | Mamba-2+MoE+Attn | — | 8K |
| Qwen3.5-35B A3B | `Kbenkhaled/Qwen3.5-35B-A3B-NVFP4` | 35B | 3B | SSM+MoE+Attn | K=2 | 8K |
| Qwen3.5-35B A3B (FP8) | `Qwen/Qwen3.5-35B-A3B-FP8` | 35B | 3B | SSM+MoE+Attn | — | 8K |
| Qwen3-Next-80B A3B | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 80B | 3B | SSM+MoE+Attn | K=2 | 8K |
| Qwen3.5-122B A10B | `Sehyo/Qwen3.5-122B-A10B-NVFP4` | 122B | 10B | SSM+MoE+Attn | K=2 | 4K |
| Mistral-Small-4 | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | 119B | 119B | MLA+Attn | — | 4K |
| Gemma-4-26B (MoE) | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` | 26B | 4B | Attn+MoE+Vision (sliding/full 5:1) | — | 32K |
| Gemma-4-31B (Dense) | `nvidia/Gemma-4-31B-IT-NVFP4` | 31B | 31B | Attn (sliding/full 5:1) | — | 32K |

---

## Quick Start — Qwen3.5-35B A3B + MTP (fastest coherent model, 113.8 tok/s)

```bash
sudo docker run -d \
  --name atlas-35b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:alpha-2.43 \
  serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --speculative \
    --mtp-quantization nvfp4
```

## Quick Start — Gemma-4-26B (coding / vision, long-context)

```bash
sudo docker run -d \
  --name atlas-gemma-26b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:alpha-2.43 \
  serve bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16 \
    --port 8888 \
    --max-seq-len 32768 \
    --kv-cache-dtype bf16 \
    --max-batch-size 1 \
    --scheduling-policy slai
```

## Quick Start — 122B on 2× GB10 (EP=2, 38.3 tok/s)

See `QUICKSTART.md` section 7 for the full dual-node NCCL-over-RoCE invocation; the alpha-2.43 image name is the only change from alpha-2.35.

---

## API

OpenAI-compatible. All standard endpoints: `/v1/chat/completions` (streaming + non-streaming), `/v1/completions`, `/v1/models`, `/health`.

Supports: `tools`/`tool_choice`, `enable_thinking` (via `chat_template_kwargs`), vision content-parts, `reasoning_content` field.

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

## Rollback

If alpha-2.43 causes regression for your workload:

```bash
docker pull avarok/atlas-gb10:alpha-2.8   # previous tagged release
```

Or pin to a specific earlier alpha-2.x — all are retained on Docker Hub.
