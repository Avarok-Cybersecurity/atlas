# Supported Models

Twelve `(GB10, model, quant)` targets ship in the default image today. One multi-model binary, one Docker image, one `serve <hf-id>` command per model. The binary reads the model's `config.json`, computes the canonical `model_type`, and dispatches to the matching kernel set at startup.

## The matrix

| Family | Model | HF ID | Params / active | Architecture | Best tok/s | MTP |
|---|---|---|---:|---|---:|:---:|
| Qwen3.5 | Qwen3.5-27B | `Kbenkhaled/Qwen3.5-27B-NVFP4` | 27B dense | Hybrid SSM + attention, dense FFN, MRoPE | 14 | ✗ |
| Qwen3.5 | Qwen3.5-35B-A3B | `Sehyo/Qwen3.5-35B-A3B-NVFP4` | 35B / 3B | GDN + attention + MoE | **131** | K=2 |
| Qwen3.5 | Qwen3.5-122B-A10B | `Sehyo/Qwen3.5-122B-A10B-NVFP4` | 122B / 10B | GDN + attention + MoE | 46 (EP=2) | K=2 |
| Qwen3.6 | Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B-FP8` | 35B / 3B | GDN + attention + MoE, MRoPE, vision tower | 90 | ✗ |
| Qwen3-Next | Qwen3-Next-80B-A3B | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 80B / 3B | SSM + attention + MoE | 104 | K=2 |
| Qwen3-VL | Qwen3-VL-30B-A3B | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` | 30B / 3B | Vision + attention + MoE | 97 | ✗ |
| Gemma-4 | Gemma-4-26B-A4B | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` | 26B / 4B | Attention + MoE, GeGLU | 67 | ✗ |
| Gemma-4 | Gemma-4-31B | `nvidia/Gemma-4-31B-IT-NVFP4` | 31B dense | Attention (sliding + full), GeGLU | 9 | ✗ |
| Mistral | Mistral-Small-4-119B | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | 119B / 6.5B | Attention + MoE | 33 | ✗ |
| MiniMax | MiniMax-M2.7 | `lukealonso/MiniMax-M2.7-NVFP4` | 229B / ~10B | Attention + 256-expert MoE | — | ✗ |
| Nemotron-H | Nemotron-3-Nano-30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 30B / 3B | Mamba-2 + attention + MoE | 98 | ✗ |
| Nemotron-H | Nemotron-3-Super-120B-A12B | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 120B / 12B | Mamba-2 + attention + MoE | 24 | ✗ |

Throughput figures are p50 single-request decode on a short prompt (`max_tokens ≤ 128`, `temperature ≤ 0.1`). The "Best tok/s" column reflects the flag set that wins for that model (e.g. MTP enabled where supported, NVFP4 KV cache for Qwen models). Full numbers for alternative flag combinations are in [Benchmarking](../operations/benchmarks.md).

## How to pick

- **Fastest** — Qwen3.5-35B-A3B with MTP. The flagship. 131 tok/s.
- **Largest on one node** — Qwen3-Next-80B-A3B or Nemotron-3-Super-120B. Both fit in 119.7 GB with FP8 KV.
- **Vision** — Qwen3-VL-30B (pure attention) or Qwen3.6-35B (hybrid SSM + vision).
- **Largest overall** — MiniMax-M2.7 at 229B / 256-expert MoE, or Qwen3.5-122B-A10B. Both require **EP=2** (two GB10 nodes over RoCEv2).
- **Long reasoning traces** — any Qwen3.5 model; thinking budget is configurable via `--max-thinking-budget`.
- **Function calling** — all Qwen-family and Nemotron models support OpenAI-style tools. See [Tool Calling](../operations/tools.md).

## Per-model serve commands

Every command below uses `avarok/atlas-gb10:latest`, `--network host --gpus all --ipc=host`, and the `-v ~/.cache/huggingface:/root/.cache/huggingface` volume mount — omitted here for readability. Full copy-pasteable commands are in [`QUICKSTART.md`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/QUICKSTART.md).

### Qwen3.5-35B-A3B (flagship)
```
serve Sehyo/Qwen3.5-35B-A3B-NVFP4 \
  --max-seq-len 8192 --kv-cache-dtype nvfp4 \
  --scheduling-policy slai --speculative --mtp-quantization nvfp4
```

### Qwen3-Next-80B-A3B (largest single-node MTP)
```
serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --max-seq-len 8192 --kv-cache-dtype nvfp4 \
  --speculative --mtp-quantization nvfp4
```

### Qwen3-VL-30B (vision)
```
serve ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4 \
  --max-seq-len 32768 --kv-cache-dtype nvfp4
```
Send images in OpenAI content-parts format (see [Tool Calling & Streaming](../operations/tools.md)).

### Nemotron-3-Nano-30B (Mamba-2 hybrid)
```
serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --max-seq-len 8192 --kv-cache-dtype nvfp4
```

### Qwen3.5-122B-A10B — EP=2
Two nodes connected via RoCEv2 (head on `<head-ip>`, worker on `<worker-ip>`). The canonical launcher is `scripts/start-ep2.sh`. See [Multi-GPU & EP=2](../operations/multi-gpu.md) for the full flow and the **critical MTP-flag symmetry rule** between head and worker.

### Single-node 122B (tight budget)
```
serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  --kv-cache-dtype fp8 --kv-high-precision-layers 2 \
  --max-batch-size 1 --max-prefill-tokens 2048 --oom-guard-mb 512
```
~32 tok/s. The `--kv-high-precision-layers 2` keeps the first and last two attention layers at BF16 — costs a few hundred MB, buys coherence at very long context.

## Adding a new model

The entire model-specific surface is **one new `ModelWeightLoader` impl** and **one match arm in `spark-model/src/factory.rs`**. The KV cache, buffer arena, scheduler, and HTTP server are all model-agnostic. The full walkthrough with a live example (Mistral-Small-4) is in the repo's [Adding a new model](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#adding-a-new-model) guide. The chapter on [spark-model](../crates/spark-model.md) covers the trait shape.
