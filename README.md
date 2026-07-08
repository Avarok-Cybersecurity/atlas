<p align="center">
  <img src="assets/logo.svg" alt="Atlas Inference Engine" width="640" />
</p>
<p align="center">
  <h1 align="center">Atlas Inference Engine</h1>
  <p align="center">
    <strong>Pure Rust LLM Inference</strong><br>
    <em>Universal Inference At Unimaginable Speeds</em>
  </p>
  <p align="center">
    <img alt="NVIDIA" src="https://img.shields.io/badge/NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white">
    <img alt="AMD" src="https://img.shields.io/badge/AMD-ED1C24?style=flat-square&logo=amd&logoColor=white">
    <img alt="Intel" src="https://img.shields.io/badge/Intel-0071C5?style=flat-square&logo=intel&logoColor=white">
  </p>
  <p align="center">
    <a href="LICENSE"><img alt="License: AGPLv3" src="https://img.shields.io/badge/license-AGPLv3-yellow?style=flat-square"></a>
    <a href="#quick-start"><img alt="Pure Rust" src="https://img.shields.io/badge/runtime-pure%20Rust-orange?style=flat-square"></a>
    <a href="https://hub.docker.com/r/avarok/atlas-gb10"><img alt="Docker Hub" src="https://img.shields.io/badge/Docker%20Hub-avarok%2Fatlas--gb10-2496ED?style=flat-square&logo=docker&logoColor=white"></a>
    <a href="https://discord.gg/6vDbKaKrKD"><img alt="Discord" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2F6vDbKaKrKD%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=discord&suffix=%20members&style=flat-square&logo=discord&logoColor=white&color=5865F2"></a>
  </p>
</p>

<p align="center">
  <a href="assets/atlas-demo.mp4"><img alt="Atlas demo — click for full-quality MP4" src="assets/atlas-demo.gif" width="820" /></a>
</p>

<p align="center">
  <a href="#quick-start"><img alt="Quick Start — under 2 minutes" src="https://img.shields.io/badge/%E2%9A%A1%20Quick%20Start%20%E2%80%94%20%3C%202%20min-2EA44F?style=for-the-badge&logo=docker&logoColor=white"></a>
  <a href="https://atlasinference.io"><img alt="atlasinference.io" src="https://img.shields.io/badge/%F0%9F%8C%90%20atlasinference.io-F48C06?style=for-the-badge"></a>
  <a href="https://x.com/AIshaqui81766/status/2052121270506930276"><img alt="Launch announcement on X" src="https://img.shields.io/badge/%F0%9D%95%8F%20Launch%20Announcement-000000?style=for-the-badge&logo=x&logoColor=white"></a>
</p>

---

## 📑 Table of Contents

- [🧭 Philosophy](#philosophy)
- [🏛️ Architecture](#architecture)
- [📦 What We Ship Today](#models)
- [⚡ Performance](#performance)
- [🗜️ KV Cache Quantization](#kv-cache)
- [🚀 Quick Start](#quick-start)
- [🔬 Kernel Debugging](#debugging)
- [🔌 Adding a New Hardware Target](#new-hardware)
- [🧬 Adding a New Model](#new-model)
- [📚 Citations](#citations)
- [⚖️ License and Enterprise Edition](#license)

---

<a id="philosophy"></a>

## 🧭 Philosophy

The foundation of any given field of science is philosophy. It is that which inspires direction, structure, and mission.

Atlas began as a solution to widely known problem in using other (python) inference engines built by data scientists: the code was steeped in a poly codebase with an ever shifting ecosystem of dependencies, patches, and cross-dependencies. One day your workaround for running a model works, the next day you have to update to a nightly branch of several dependencies and inject a new workaround. This is not how you build a software ecosystem; that's how you build a proof of concept. We thank the great and hard work data scientists made in proving LLMs can revolutionize our world, its economy, and how it challenges us to higher epochs. Now, the software engineers take the torch to turn a proof of concept into something that is designed to withstand the test of time.

### Main Objective

Similar to how llama.cpp was built with the intent to prove you don't need $10000-$100000 GPUs to run LLMs, Atlas is built with the intent to consistently force the narrative that as hardware continues to advance, we should not have to pay premium Cloud API prices for inference. Atlas, by virtue of its philosophy, maximizes speed for each hardware/model combination, thus paving the way for meaningfully powerful and intelligent LLMs to be run locally in such a way the model is truly useful.

### Design Choices

#### Free and Open Source, Always

We promised this since the beginning. We believe great software comes from opening the source, not from just keeping it closed. The more eyes, the better. And therein brings us to the next point.

#### Community-First

For those who've followed us this far since the inception of our Discord, you know the extent to which our commitment to the community is, according to one user humourously put, "cracked". We want to build something incredible, and that means we not only build for you, but you, now having access to the source code, can now build for others in ways that triumph over existing solutions. This is the only way we all win. We are the Pirates of the inference space.

#### Monorepo

We chose a monorepo design to ensure that, as we head further into the agentic age of coding, the average data scientist or engineer can contribute meaningful PRs to any part of the system. Eventually, since this is a monorepo, there will be a day where the repo is autonomously self-improving and self-patching. This is most efficient and most effective when all the code is in one place, not many.

#### Hardware+Model Specific Kernels

We make no compromises or generalizations. Each hardware and model combination has its own unique properties that require fine-tuning custom kernels that leverage the model for that specific hardware configuration. The end result? 2-3x faster kernels all around.

#### AI-Friendly Codebase

It took a significant amount of time to build this codebase. We also know people will want to submit AI-generated PRs. We can't stop you, and in fact, given SOTA, you might just have to! The good news is that this codebase was built with enough railguards, structure, and abstraction to guide your AI to absorb the entire monorepo and contribute meaningfully. There's enough context to keep this going off the rails like a crazy train.This means ultimately that instead of waiting for days to weeks before getting model support, you can just fork this repo, and ask your AI to integrate it, then within hours you'll more likely than not have a working model running. We will not be condescending, [unlike some other inference engines out there when good-faith PRs that simply work are posted](https://github.com/ggml-org/llama.cpp/pull/18680#issuecomment-3723954542). We are not stymied by bureacracy, and want to enable the community to rapidly expand this monorepo ecosystem safely and effectively.

#### Theory-Friendly Codebase

Arxiv is getting countless papers published every day on AI. Nobody can keep up. Yet, some papers may be relevant to this project, others may not. Research endeavors to improve quality, alignment, and speed ought to be considered by our community as something we can integrate cleanly. Feel free to open a PoC PR here and just explain what you did and why, and how it works.

#### Plug and Play Design

Our system is modular, with tight abstraction boundaries and trait requirements that force the architecture to take on a certain form. This form is designed to prevent pigeon-holing the project into the wrong direction. The business logic is the same across all hardware/model combinations, just the concrete implementations differ.

<a id="architecture"></a>

## 🏛️ Architecture

The diagram below shows how a single HTTP request flows from the API surface down to hardware-specific CUDA kernel execution. **Dashed borders** mark the **plug-and-play** abstraction boundaries — the traits and registries where a new hardware target, model family, communication backend, or storage backend plugs in without touching the layers above or below it.

```mermaid
flowchart TB
    %% ── Colours & styles ──────────────────────────────────────────────
    classDef server fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef scheduler fill:#1e6091,stroke:#184e77,color:#d9ed92
    classDef model fill:#b5179e,stroke:#7209b7,color:#ffe5fc
    classDef layer fill:#7209b7,stroke:#560bad,color:#ffd6ff
    classDef kernel fill:#f48c06,stroke:#dc2f02,color:#fff
    classDef storage fill:#264653,stroke:#1d3557,color:#a8dadc
    classDef comm fill:#3a86ff,stroke:#1d3557,color:#fff
    classDef trait stroke-dasharray: 6 4,stroke-width:2px

    %% ── Top layer: HTTP API ───────────────────────────────────────────
    HTTP["HTTP Server (spark-server)<br/>OpenAI · Anthropic · Responses"]:::server
    SCHED["Scheduler<br/>batches, MTP verify, KV alloc"]:::scheduler

    HTTP --> SCHED

    %% ── Model abstraction (plug-in #1) ────────────────────────────────
    subgraph MODEL ["🔌 trait Model"]
      direction TB
      TRANSFORMER["TransformerModel<br/>generic prefill/decode loop"]:::model
    end
    class MODEL trait

    SCHED --> MODEL

    %% ── Weight loader abstraction (plug-in #2) ────────────────────────
    subgraph LOADER ["🔌 trait ModelWeightLoader"]
      direction LR
      QW35["Qwen3.5<br/>27B/35B/122B"]:::layer
      QW36["Qwen3.6<br/>35B-A3B"]:::layer
      QWNEXT["Qwen3-Next<br/>80B-A3B"]:::layer
      QWVL["Qwen3-VL<br/>30B-A3B"]:::layer
      GEMMA["Gemma-4<br/>26B/31B"]:::layer
      MISTRAL["Mistral-Small-4<br/>119B"]:::layer
      MINIMAX["MiniMax M2.7<br/>229B-A10B"]:::layer
      NEMO["Nemotron-3<br/>Nano/Super"]:::layer
    end
    class LOADER trait

    TRANSFORMER --> LOADER

    %% ── Layer trait (plug-in #3) ──────────────────────────────────────
    subgraph LAYERS ["🔌 trait TransformerLayer"]
      direction LR
      ATTN["Attention<br/>(GQA, MLA, sliding)"]:::layer
      SSM["SSM<br/>(Mamba-2, GDN)"]:::layer
      MOE["MoE<br/>(routed + shared)"]:::layer
      FFN["Dense FFN<br/>(GeGLU, SwiGLU)"]:::layer
      MTP["MTP Head<br/>(draft proposer)"]:::layer
    end
    class LAYERS trait

    LOADER --> LAYERS

    %% ── GPU backend (plug-in #4) ──────────────────────────────────────
    subgraph GPU ["🔌 trait GpuBackend"]
      direction LR
      CUDA["CUDA backend<br/>(GB10 / Blackwell)"]:::kernel
      AMD["AMD ROCm<br/>(future)"]:::kernel
      APPLE["Apple Metal<br/>(future)"]:::kernel
    end
    class GPU trait

    LAYERS --> GPU

    %% ── Kernel registry (plug-in #5) ──────────────────────────────────
    subgraph KERNELS ["🔌 kernels/<hw>/<model>/<quant>/ — auto-discovered"]
      direction LR
      K_GB10["gb10/qwen3.5-35b-a3b/nvfp4<br/>+ 11 other targets"]:::kernel
    end
    class KERNELS trait

    CUDA --> KERNELS

    %% ── EP / multi-GPU (plug-in #6) ───────────────────────────────────
    subgraph EP ["🔌 trait CommBackend"]
      direction LR
      NCCL["NCCL<br/>(EP=2, all-reduce)"]:::comm
    end
    class EP trait

    LAYERS -.-> EP

    %% ── Storage backend (plug-in #7) ──────────────────────────────────
    subgraph STORE ["🔌 trait StorageBackend"]
      direction LR
      IORING["io_uring<br/>(NVMe KV offload)"]:::storage
    end
    class STORE trait

    SCHED -.-> STORE

    %% ── Cross-references ──────────────────────────────────────────────
    KERNELS -. "kernels selected by<br/>(hardware × model × quant)<br/>at build time" .-> CUDA
```

### Reading the Diagram

**Solid boxes** are concrete implementations. **Dashed borders with 🔌** are the trait-based abstraction boundaries — each is a Rust trait (or a filesystem convention for kernels) where a new integration plugs in:

| Plug Point | What It Abstracts | To Add New Support |
|---|---|---|
| `trait Model` | Full model forward pass | Rarely needed — the existing `TransformerModel` handles all architectures via composable layers |
| `trait ModelWeightLoader` | HuggingFace → layer translation | **Implement one struct** with weight-name patterns for your model family ([`factory.rs`](crates/spark-model/src/factory.rs) adds one match arm) |
| `trait TransformerLayer` | Per-layer compute (attn, SSM, MoE, FFN) | Compose existing layer types or implement a new one for novel architectures |
| `trait GpuBackend` | All GPU memory and kernel ops | Swap the CUDA driver for another accelerator backend |
| `kernels/<hw>/<model>/<quant>/` | Hardware-tuned CUDA kernels | Drop a new directory with `MODEL.toml` + `.cu` files; `build.rs` auto-discovers it |
| `trait CommBackend` | Multi-GPU collective communication | Implement for MPI, GDR, or custom interconnects |
| `trait StorageBackend` | NVMe KV-cache offload I/O | Implement for CXL, RDMA, or other storage tiers |

### Data Flow Summary

1. **HTTP** → `spark-server` receives OpenAI/Anthropic requests, tokenizes, and enqueues
2. **Scheduler** → batches sequences, orchestrates prefill/decode/speculative-verify steps
3. **Model** → generic loop: `embed → [layer₀ … layerₙ] → norm → lm_head`
4. **Layers** → each layer dispatches through `GpuBackend` to launch kernels from `AtlasRegistry`
5. **Kernels** → pre-compiled PTX selected by `(hardware × model × quant)` target at build time
6. **EP** → `CommBackend` handles cross-GPU all-reduce after MoE expert computation
7. **Storage** → `StorageBackend` spills/restores KV blocks to NVMe for long-context sequences

<a id="models"></a>

## 📦 What We Ship Today

We have to walk before we can run. Today's Atlas is targeted at a single hardware platform — NVIDIA's GB10 (DGX Spark, SM121) — and fifteen hand-tuned (Hardware × Model × Quantization) targets. Every supported model below runs off one multi-model binary; the right kernel set is selected at startup from the model's `config.json`. No swapping images, no rebuilding, no per-model magic — just point Atlas at a HuggingFace ID.

| Family | Model | HuggingFace ID | Params / active | Architecture |
|---|---|---|---:|---|
| Qwen3.5 | Qwen3.5-27B | `Kbenkhaled/Qwen3.5-27B-NVFP4` | 27B dense | Hybrid SSM + attention, dense FFN, MRoPE |
| Qwen3.5 | Qwen3.5-35B-A3B | `Sehyo/Qwen3.5-35B-A3B-NVFP4` | 35B / 3B | GDN + attention + MoE, MTP |
| Qwen3.5 | Qwen3.5-122B-A10B | `Sehyo/Qwen3.5-122B-A10B-NVFP4` | 122B / 10B | GDN + attention + MoE, MTP |
| Qwen3.6 | Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B-FP8` | 35B / 3B | GDN + attention + MoE, MRoPE, vision tower |
| Holo-3.1 | Holo-3.1-35B-A3B | `Hcompany/Holo-3.1-35B-A3B-NVFP4` | 35B / 3B | GDN + attention + 256-expert MoE, Qwen3-VL vision |
| Holo-3.1 | Holo-3.1-0.8B | `Hcompany/Holo-3.1-0.8B` | 0.8B dense | GDN + attention + dense FFN, Qwen3-VL vision |
| Ornith | Ornith-1.0-9B | `deepreinforce-ai/Ornith-1.0-9B` | 9B dense | GDN + attention + dense FFN, Qwen3-VL vision, MRoPE |
| Qwen3-Next | Qwen3-Next-80B-A3B | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 80B / 3B | SSM + attention + MoE |
| Qwen3-VL | Qwen3-VL-30B-A3B | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` | 30B / 3B | Vision + attention + MoE |
| Gemma-4 | Gemma-4-26B-A4B | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` | 26B / 4B | Attention + MoE, GeGLU |
| Gemma-4 | Gemma-4-31B | `nvidia/Gemma-4-31B-IT-NVFP4` | 31B dense | Attention (sliding + full), GeGLU |
| Mistral | Mistral-Small-4-119B | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | 119B / 6.5B | Attention + MoE |
| MiniMax | MiniMax-M2.7 | `lukealonso/MiniMax-M2.7-NVFP4` | 229B / ~10B | Attention + 256-expert MoE + MTP |
| Nemotron-H | Nemotron-3-Nano-30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 30B / 3B | Mamba-2 + attention + MoE |
| Nemotron-H | Nemotron-3-Super-120B-A12B | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 120B / 12B | Mamba-2 + attention + MoE |

This is a starting point, not a destination. The plug-and-play design above exists precisely so that AMD, Apple Silicon, Intel, and the next round of Blackwell parts can land here as community contributions, and so that the Llama 4s and DeepSeek V4s of next quarter slot in the same way the Qwens did this quarter. We did the hard part — bolting in the abstractions while bringing up the first fifteen targets — so that adding the sixteenth is a weekend, not a quarter.

> **New to Atlas on a Spark?** The [**GB10 Deployment & Compatibility Guide**](docs/GB10_DEPLOYMENT_GUIDE.md) is the one page to read first: which model and quant fit your box and your goal, what to do when it OOMs, the known gotchas, and what "verified" means — then it hands you the exact recipe. If you're deciding *what to run*, start there.

<a id="performance"></a>

## ⚡ Performance

We're not going to spend much real estate on benchmark theatre. The numbers below are what the binary in this repository does on a single NVIDIA GB10, on a short prompt (`"What is the capital of France?"`, `max_tokens ≤ 30`, `temperature = 0.1`), measured end-to-end through the HTTP API. They are reproducible: `scripts/sweep_all_models.sh` is the harness, and the source for every kernel that produced them is in this repository.

| Model | Mode | tok/s |
|---|---|---:|
| Qwen3.5-35B-A3B | MTP speculative (K=2) | **131** |
| Qwen3.5-35B-A3B | turbo4 KV | 77 |
| Qwen3.5-35B-A3B | No speculative | 70 |
| Qwen3-Next-80B-A3B | FP8 KV | 74 |
| Qwen3.5-122B-A10B | EP=2, MTP K=2 (600-tok sustained) | 46 |
| Qwen3.5-122B-A10B | FP8 KV, single-GPU tuned | 32 |
| Qwen3-VL-30B-A3B | NVFP4 KV | 97 |
| Nemotron-3-Nano-30B-A3B | FP8 KV | 88 |
| Nemotron-3-Super-120B | FP8 KV | 24 |
| Gemma-4-26B-A4B | default | 67 |
| Gemma-4-31B | `--max-batch-size 2` | 9 |
| Mistral-Small-4-119B | NVFP4 | 33 |
| Qwen3.5-27B (dense hybrid) | FP8 KV | 13 |

We compete with vLLM and TensorRT-LLM on the same GB10. On Qwen3.5-35B-A3B with MTP speculative decoding, Atlas decodes faster than the same model under NVIDIA's own vLLM build on the same hardware — meaningfully faster, on numbers we can hand you the script for. We will not put a bigger figure in this paragraph than the one that comes off our own benchmark scripts, and we publish the vLLM baseline command alongside ours so you can verify both. If you reproduce a faster vLLM number, file an issue. We would rather be measured than congratulated.

The kernel-by-kernel comparison against PyTorch eager (35 hyperoptimized CUDA kernels, all wins on production-relevant shapes) lives in the [benchmarks chapter](book/src/operations/benchmarks.md) along with the methodology footnotes — read them; they matter.

<a id="kv-cache"></a>

## 🗜️ KV Cache Quantization

Atlas stores attention key/value state in one of six quantized formats, selected via `--kv-cache-dtype`. Lower bit-widths fit more tokens in GPU memory at the cost of precision; the Turbo family adds Walsh-Hadamard rotation and Lloyd-Max optimal codebooks to recover accuracy at the same bit rate. Mix dtypes per layer with `--kv-high-precision-layers` to keep boundary layers at BF16 while compressing the middle.

| CLI flag | Bits/element | Scale overhead | Technique | When to use |
|---|---:|---|---|---|
| `bf16` | 16 | — | Raw BF16 storage | Maximum precision; short-context or quality-critical workloads |
| `fp8` | 8 | Per-tensor FP32 scale (from checkpoint or online calibration via `--fp8-kv-calibration-tokens`) | FP8 E4M3 with static or calibrated per-tensor scale | **Default.** Safe baseline — half the memory of BF16, minimal quality loss for most models |
| `turbo8` | 8 | Per-group BF16 scale (2 bytes / 16 elements) | Walsh-Hadamard rotation → FP8 E4M3 + BF16 per-group scales | FP8-level memory with outlier suppression; recommended for many-layer models (e.g. MiniMax M2.7, 58 layers) where per-group FP8 scales compound |
| `nvfp4` | 4 | Per-group FP8 scale (1 byte / 16 elements) | E2M1 packed nibbles (NVIDIA NVFP4 format) | 4× compression vs BF16; good for long-context with `--kv-high-precision-layers auto` |
| `turbo4` | 4 | Per-group FP8 scale (1 byte / 16 elements) | Walsh-Hadamard rotation → Lloyd-Max optimal 4-bit codebook | ~2× lower MSE than NVFP4 at the same bit rate; same memory footprint |
| `turbo3` | 3 | Per-group FP8 scale (1 byte / 16 elements) | Walsh-Hadamard rotation → Lloyd-Max 3-bit codebook (8 levels, packed 8 values → 3 bytes) | Maximum compression (22% smaller than turbo4); experimental |

<a id="quick-start"></a>

## 🚀 Quick Start

The whole supported model matrix lives in one Docker image. Pull it, mount your HuggingFace cache, point Atlas at any model ID from the [model table](#models).

> **Defaults below are tuned for maximum accuracy under agentic-coding workloads** — 64K context window, BF16 MTP draft head (highest acceptance rate ⇒ highest end-to-end throughput), prefix caching for multi-turn tool loops, and FP8 KV cache with `auto`-promoted boundary layers. These are the recipes we use to drive opencode / Claude Code / Cline through Atlas on a single Spark.

<a id="run-atlas"></a>

### Recipe A — Qwen3.6-35B-A3B (FP8 hybrid MoE, ~130 tok/s)

The default daily driver — 35 B params, 3 B active, GDN + attention + 256-expert MoE, MRoPE-positioned vision tower (text-only here).

```bash
docker pull avarok/atlas-gb10:latest

sudo docker run -d --name atlas \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port 8888 \
    --max-seq-len 65536 \
    --kv-cache-dtype fp8 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.90 \
    --scheduling-policy slai \
    --enable-prefix-caching \
    --speculative \
    --num-drafts 2 \
    --tool-call-parser qwen3_coder
```

Why these flags:

- `--max-seq-len 65536` — 64K window for long agent traces, file reads, multi-step tool use.
- `--kv-cache-dtype fp8 --kv-high-precision-layers auto` — half the memory of BF16, no measurable quality loss; the `auto` heuristic keeps first/last attention blocks at BF16 where the routing distribution is most sensitive.
- `--scheduling-policy slai` — SLAi scheduler (Atlas's default) reorders concurrent sequences to keep MTP verify batches dense.
- `--enable-prefix-caching` — radix-tree prefix cache; tool-use sessions reuse the system prompt + tool-defs + earlier turns.
- `--speculative --num-drafts 2` — MTP draft head proposes 2 tokens per step. **No `--mtp-quantization` flag** ⇒ defaults to **BF16**, which gives the highest acceptance rate (lossier MTP projections lower acceptance and usually *worsen* end-to-end tok/s, despite the faster draft forward).
- `--tool-call-parser qwen3_coder` — explicit Qwen XML tool format. Atlas auto-resolves the right parser from `tool_defaults.toml` per model; pass it anyway in production scripts.

### Recipe B — Qwen3.5-35B-A3B (NVFP4, ~131 tok/s with MTP K=2)

The fastest model in the matrix on a single Spark.

```bash
sudo docker run -d --name atlas \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-35B-A3B-NVFP4 \
    --port 8888 \
    --max-seq-len 65536 \
    --kv-cache-dtype fp8 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.90 \
    --scheduling-policy slai \
    --enable-prefix-caching \
    --speculative \
    --num-drafts 2 \
    --tool-call-parser qwen3_coder
```

### Recipe C — Qwen3.5-122B-A10B (NVFP4, single Spark)

The 122B NVFP4 weights + Atlas runtime overhead leave only ~2 GB for KV cache on a 119.7 GB GB10, so this recipe sacrifices `--speculative` (the MTP draft head + draft KV costs ~1.5 GB) to keep a real 16 K context window.  Verified end-to-end: model loads, `/v1/chat/completions` answers correctly, 4-way concurrent serves cleanly.

```bash
sudo docker run -d --name atlas \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --port 8888 \
    --max-seq-len 16384 \
    --kv-cache-dtype fp8 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 \
    --scheduling-policy slai \
    --max-batch-size 1 \
    --max-num-seqs 4 \
    --oom-guard-mb 1024 \
    --ssm-cache-slots 0 \
    --tool-call-parser qwen3_coder
```

For 122B with **both** `--speculative` *and* a 64 K window, move to EP=2 across two Sparks ([`QUICKSTART.md`](QUICKSTART.md) §5). For long contexts on a single Spark, add `--high-speed-swap --high-speed-swap-dir /path/on/nvme --high-speed-swap-cache-blocks-per-seq 64` — HSS keeps a rolling 1024-token KV window in HBM and streams older blocks to NVMe through an io_uring orchestrator. The container needs `--security-opt seccomp=unconfined --ulimit memlock=-1` for io_uring access.

### Hitting the Endpoint

Atlas speaks OpenAI, Anthropic, and Responses APIs on the same port. `curl`, the OpenAI SDK, Open WebUI, opencode, Cline, Claude Code — point them at port 8888:

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"atlas",
    "messages":[{"role":"user","content":"Hello!"}],
    "max_tokens":256
  }'
```

Per-model recipes (vision input, multi-node EP=2, single-GPU 122B with the tighter budget) live in [`QUICKSTART.md`](QUICKSTART.md). Build-from-source instructions are in [`CONTRIBUTING.md`](CONTRIBUTING.md), and the kernel build pipeline is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#build-pipeline).

<a id="debugging"></a>

## 🔬 Kernel Debugging

Atlas exposes a focused set of **environment-gated diagnostic dumps** for tracking down quality regressions — magnitude drift, expert-routing skew, MoE under-counting, and the rest of the bug class where the kernels run cleanly but the output slowly degrades. The dumps are zero-overhead when their env var is unset (single `var()` lookup per call, no GPU sync, no copy) so leaving the production binary instrumented is safe.

**For the full diagnostic playbook** — including the cheapest-signal-first elimination ladder, how to build a byte-exact HF CPU oracle, the per-layer divergence comparator, and the methodological reversals that cost us hours — see [**`DEBUGGING_METHODOLOGY.md`**](DEBUGGING_METHODOLOGY.md). What follows is the env-var reference.

### MoE-path dumps — `ATLAS_DUMP_EXPERT_IDS=1`

Set `-e ATLAS_DUMP_EXPERT_IDS=1` on the container. The MoE prefill paths (both `forward_prefill_fp8.rs` and `forward_prefill.rs`) emit the following per-fire log lines, scoped to the **last token of the chunk** so the values are directly comparable to a single-pass reference forward at the same position:

| Log marker | Fires | What it captures | Use it to localize |
|---|---|---|---|
| `ATLAS_FP8_GROUPED_KERNEL` | once (FP8 only) | v1 vs v2 grouped-GEMM selection | Confirm which kernel variant is active |
| `ATLAS_EXPERT_LOAD` | once / server | Per-expert histogram + `truncated=true/false` flag | Spot `max_m_tiles` truncation against actual routing skew |
| `ATLAS_GATE_INPUT` | per layer × chunk | post-norm router input (`\|x\|` + `first5`) | Verify the MoE block input matches the reference |
| `ATLAS_GATE_LOGITS` | per layer × chunk | top-10 `(idx, val)` + mean + std of raw gate logits | Catch gate-matmul drift before softmax/topK |
| `ATLAS_EXPERT_IDS` | per layer × chunk | top-K indices + renormalized weights + sum | Confirm routing decisions match HF |
| `ATLAS_ROUTED_ONLY` | per layer × chunk | routed sum **before** shared blend | Isolate the routed-expert contribution |
| `ATLAS_SHARED_OUT` | per layer × chunk | shared-expert output (pre-sigmoid) | Verify the dense FFN branch independently |
| `ATLAS_SHARED_GATE` | per layer × chunk | `dot(input, gate_weight)` + sigmoid value | Confirm shared-expert attenuation matches |
| `ATLAS_MOE_OUT` | per layer × chunk | final MoE block output (routed + blended) | The full-block ground truth vs the reference |

### SSM-path dumps

The SSM (GDN / Mamba-2) prefill in `qwen3_ssm/trait_prefill.rs` adds three pre-norm hooks under the same env var:

| Log marker | What it captures |
|---|---|
| `ATLAS_PRENORM_HIDDEN` | Residual stream entering this layer (= previous layer's output) |
| `ATLAS_PRENORM_OUTPROJ` | SSM `out_proj` output before residual add |
| `ATLAS_PRENORM_SUM` | hidden + out_proj (the input to `post_attention_layernorm`) |

Together, those plus the MoE dumps above give a complete trace of the residual stream at every layer boundary for any token in any chunk.

### Path-toggle env vars

For bisecting *which* code path is at fault, two override toggles let you swap the routed-expert dispatch at runtime without rebuilding:

| Env var | Effect |
|---|---|
| `ATLAS_FP8_MOE_COALESCED=0` | Forces the FP8 grouped-GEMM v1 kernel (default is v2; v1 has a documented numerical bug for some `(token, expert)` tiles) |
| `ATLAS_FORCE_NVFP4_MOE=1` | Routes an FP8 model's MoE through the NVFP4 path — useful for cross-validating that the bug is in one specific quant path |

### How we use these in practice — 3-step workflow

The order matters; this is the same workflow that found and fixed three compounding MoE bugs (commits `6a5fd3d`, `34626d3`, `adf39ce`, `ffdb41d`) on the Qwen3.6-A3B long-context investigation:

1. **Build an HF reference oracle.** A single-precision forward pass through HF Transformers on the same token IDs (read them back from Atlas's `/tokenize` — *do not* re-render the chat template), with `output_hidden_states=True` and per-layer hooks on `mlp.gate`, `mlp.shared_expert`, and `mlp.shared_expert_gate`. Record `\|x\|` + `first5` per layer for the last token.
2. **Spin up Atlas with `-e ATLAS_DUMP_EXPERT_IDS=1`.** Fire the same prompt. The MoE markers above give you per-layer Atlas values comparable to the oracle.
3. **Per-layer comparator.** A short script (the comparator pattern is captured in [`DEBUGGING_METHODOLOGY.md` §4](DEBUGGING_METHODOLOGY.md#4-per-layer-divergence-comparator)) prints `ratio = |Atlas| / |HF|` and `overlap = |top-K_Atlas ∩ top-K_HF|` per layer. The first layer where the ratio falls outside `[0.95, 1.05]` or overlap drops below 6/8 is your first-divergent layer — start drilling there.

For the 2026-05-20 MoE bug hunt this localized the issue from "16K context produces gibberish" to "L0 MoE output magnitude 3.4× too large because of three compounding bugs: v1 grouped-GEMM, missing zero-init, broken `max_m_tiles` heuristic" within a few iterations. After all three fixes, all 40 layers landed in `[0.977, 1.021]` of HF baseline — at the FP8 quantization noise floor.

<a id="new-hardware"></a>

## 🔌 Adding a New Hardware Target

The full recipe is in [`docs/HARDWARE.md`](docs/HARDWARE.md#adding-a-new-hardware-target). The short version: implement two traits (`ComputeTarget` for the build-time compiler, `GpuBackend` for the runtime), drop kernel sources into `kernels/<your-hw>/`, add one match arm in the registry. There is a `MockGpuBackend` in `spark-runtime` that lets you write and test the entire scaffold without owning the hardware — every layer above the GPU trait is hardware-agnostic, so unit tests can run on a laptop. We bolted the project from "single CUDA target" to "trait-pluggable across vendors" specifically so that the AMD, Apple, and Intel ports stop being our problem and start being yours.

<a id="new-model"></a>

## 🧬 Adding a New Model

Same story, smaller surface. Implement `ModelWeightLoader` (one struct, the existing `Qwen3AttentionLayer`/`MoeLayer`/`Qwen3SsmLayer`/`NemotronMamba2Layer` primitives cover most architectures), add one line to the factory dispatch, optionally drop a `MODEL.toml` for sampling defaults and behavior knobs. Kernels are reused; the scheduler is untouched; the server is oblivious. The step-by-step cookbook is in [`docs/HARDWARE.md`](docs/HARDWARE.md#adding-a-new-model-family). Once your loader produces coherent output on the integration coherence prompt, you are done — file the PR.

<a id="citations"></a>

## 📚 Citations

We did not invent the kernels we ship. We picked the right ideas from the right papers, fused them together, and tuned them for one chip until they pinned the bandwidth ceiling. Atlas owes a direct intellectual debt to:

- **FlashAttention-2** — Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) — tiled online softmax, Q/K/V SMEM staging, causal masking. Foundation of our prefill kernel.
- **FlashAttention-4** — Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. *FlashAttention-4: Taming the Hardware.* 2025. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451) — conditional softmax rescaling and software polynomial `sw_exp` (3 FMA + `ldexpf` instead of going through the SFU). Both shipped in our GQA-fused paged Flash Attention.
- **FlashInfer** — Ye, Chen, Lai, Zhao, Zheng, Shao, Hou, Jin, Zuo, Yin, Chen, Ceze. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.* MLSys 2025 (Best Paper). [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) — block-sparse paged KV cache, page index prefetch to SMEM, the gather-SMEM-MMA pattern for scattered pages. Informed our paged attention design.
- **SageAttention 3** — Zhang, Huang, Zhang, Wei, Zhu, Chen. *SageAttention3: Microscaling FP4 Attention on Blackwell GPUs.* NeurIPS 2025 Spotlight. [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) — FP4 attention with FP8 per-block microscales. On the SM121 roadmap once silicon-level FP4 MMA arrives upstream.
- **LeanAttention** — Roy, Vassilieva, Willke, Mendis. *LeanAttention: Hardware-Aware Scalable Attention for LLM Inference.* 2024. [arXiv:2405.10480](https://arxiv.org/abs/2405.10480) — stream-K tile scheduling for near-100% SM occupancy in split-K decode attention. Planned next.
- **TurboQuant** — Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* arXiv preprint, April 2025. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — Randomized Hadamard Transform + Lloyd-Max codebook for KV cache compression. The implementation in our `kernels/gb10/common/wht_bf16.cu` + `reshape_and_cache_turbo.cu` follows the **TurboQuant+** extensions (matched-norm L2 correction, sparse V dequant, asymmetric K/V, InnerQ per-channel equalisation) collected at [`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus) (research umbrella) with the llama.cpp engine reference at [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant); per-feature reproduction and prior-art chain in [`docs/turboquant-plus.md`](docs/turboquant-plus.md).

If you wrote one of these papers and you spot a misattribution or a wrong technique credit on our side, open an issue. We would rather be corrected than wrong.

<a id="license"></a>

## ⚖️ License and Enterprise Edition

Atlas operates under a **dual-license** model. Both are real, both are intentional, and neither is a teaser for the other.

1. **[Community Edition](LICENSE) — AGPLv3.** Free, open, copyleft. Use it for yourself to run inference on your own hardware, research, hobby projects, side-projects, and/or hosted demos, as examples. If you want to make money from Atlas, purchase a commercial license.
2. **Enterprise Edition — commercial license.** If you need to ship Atlas inside a closed-source product, run it as a SaaS backend without inheriting the AGPLv3 source-disclosure obligation, or simply want a support relationship with the people who wrote the kernels, contact sales. Enterprise customers also receive prioritized model and hardware ports.

This split exists for a single reason: a permissive license keeps us building Atlas full-time, and the AGPL community license keeps the project honest. What is in this repository is what we run.
