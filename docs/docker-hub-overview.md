# Atlas Spark

State-of-the-art LLM serving on a single NVIDIA DGX Spark (GB10). OpenAI-compatible API. NVFP4 weights + KV cache. MTP speculative decoding. Tool calling. Vision.

```bash
docker pull avarok/atlas-gb10:alpha-2.43   # pinned
docker pull avarok/atlas-gb10:latest        # floating
```

**Current release:** `alpha-2.43` — digest `sha256:cd3b2aa1e806017fcaf64649a255a9d8831c9960248c7be4dc98035e104d66ec`

## Supported models (tok/s on GB10 single Spark, TP=1)

| Model | HF ID | Params | Active | Decode |
|---|---|---|---|---|
| Qwen3.5-35B A3B + MTP | `Kbenkhaled/Qwen3.5-35B-A3B-NVFP4` | 35B | 3B | **114 tok/s** |
| Qwen3.5-35B A3B | same | 35B | 3B | 90 tok/s |
| Qwen3-Next-80B A3B + MTP | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 80B | 3B | 87 tok/s |
| Qwen3.5-35B A3B FP8 | `Qwen/Qwen3.5-35B-A3B-FP8` | 35B | 3B | 71 tok/s |
| Qwen3-VL-30B (Vision) | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` | 30B | 3B | 68 tok/s |
| Qwen3.5-122B A10B + MTP (EP=2) | `Sehyo/Qwen3.5-122B-A10B-NVFP4` | 122B | 10B | 38 tok/s |
| Qwen3.5-122B (single Spark) | same | 122B | 10B | 32 tok/s |
| Mistral-Small-4 | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | 119B | 119B | 30 tok/s |
| Gemma-4-26B (MoE+Vision) | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` | 26B | 4B | 73 tok/s |
| Gemma-4-31B (Dense) | `nvidia/Gemma-4-31B-IT-NVFP4` | 31B | 31B | 11 tok/s |
| Nemotron-H 30B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 30B | 3B | 88 tok/s |

Measured ISL=128, concurrency=1, OSL=128, p50. Full scoreboard + methodology: `alpha-2-43-release.md` in the repo.

## Quick start (35B MoE, fastest model)

```bash
sudo docker run -d \
  --name atlas \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --speculative --mtp-quantization nvfp4
```

Then:

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
    "messages": [{"role":"user","content":"Hello!"}],
    "max_tokens": 64
  }'
```

## What's new in alpha-2.43

- **Gemma-4 RoPE proportional** fixed — full-attention layers (5/30 in 26B, 10/60 in 31B) now rotate the correct dims with the right frequency. 26B Creative+fib flipped to PASS; 31B LC4k now recalls the needle.
- **New HDIM=512 paged prefill kernel** — 8 warps, single-buffered K, 101,120 B dynamic shared memory. Unblocks chunked long-context prefill on Gemma-4 full-attention (was returning HTTP 500).
- **Paged prefill sliding-window mask** — closes a pre-existing correctness gap: chunks 1+ were attending to all preceding tokens instead of the 1024-window for sliding-attention layers.
- Gemma-4 MoE final-logit-softcap guard, nemotron-nano thinking budget 1024→2048, and two test-harness bug fixes.

**Pass-28 score: 123/135 (91.1%), 10/15 PERFECT single-GPU.** Full table + deltas vs pass-22 baseline in `alpha-2-43-release.md`.

## Compatibility

- Hardware: NVIDIA DGX Spark GB10 only (sm_121f compute capability).
- API: OpenAI-compatible — any OpenAI SDK (Python, TS, etc.), Open WebUI, or opencode.
- License: AGPL-3.0-only.

## Links

- **QUICKSTART + docs:** `QUICKSTART.md` (per-model commands, tool calling, streaming, EP=2 setup)
- **Release notes:** `alpha-2-43-release.md`
- **Rollback:** `docker pull avarok/atlas-gb10:alpha-2.8` (or any earlier tag)
