# Changelog

All notable changes to Atlas are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For per-release deep dives — kernel-level wins, the engineering history
behind specific subsystems — see the
[Atlas Spark Journey](docs/ATLAS_SPARK_JOURNEY.md).

## [Unreleased]

### Fixed

- `--gpu-memory-utilization` now enforces a hard ceiling on total GPU
  memory (weights + buffers + KV cache + reserves), matching the vLLM /
  sparkrun convention.  Previously the fraction was applied only to
  post-weight free memory, causing the KV cache to over-allocate by
  20-27 GB when values below the ~0.88 default were used.  This blocked
  multi-service co-residency on shared-memory systems (e.g. DGX Spark
  GB10).  The flag now behaves as documented: `0.50` on a 120 GB device
  caps Atlas at ~60 GB total.  (#170)

## [0.1.0] — 2026-05-06

Initial public release. Atlas is a pure-Rust LLM inference engine
targeting NVIDIA GB10 (DGX Spark, SM121) with twelve hand-tuned
(Hardware × Model × Quantization) targets.

### Added

- Pure-Rust runtime — no Python, no PyTorch — for hybrid Attention +
  SSM/GDN/Mamba-2 architectures with NVFP4 / FP8 / BF16 quantization.
- 35 hyperoptimized CUDA kernels per target, compiled to PTX and
  embedded in the binary at build time. Multi-model image dispatches
  the right kernel set at startup from `config.json`.
- OpenAI- and Anthropic-compatible HTTP API (`/v1/chat/completions`,
  `/v1/responses`, `/v1/messages`, `/v1/models`, `/v1/conversations`,
  `/tokenize`, `/detokenize`, `/health`, `/metrics`).
- Tool calling with grammar-constrained decoding (Hermes,
  Qwen3-Coder, Mistral, MiniMax-XML formats).
- MTP speculative decoding (K=2 pipelined verify), self-speculative
  layer-skipping, and N-gram speculative decoding.
- Prefix caching: radix-tree (RadixAttention) + SSM snapshot cache
  (Marconi-style). 10× warm-cache TTFT reduction.
- KV cache dtypes: BF16, FP8, NVFP4, turbo3, turbo4. Optional
  per-layer high-precision overlay (`--kv-high-precision-layers`).
- Multi-GPU expert parallelism (EP=2 over RoCEv2) for models that
  exceed a single GB10's weight budget (122B-class, MiniMax M2.7).
- Vision encoder (Qwen3-VL, Qwen3.6 ViT).
- High-speed NVMe KV swap (sliding-window, io_uring) for
  long-context decoding past the HBM cap.
- Bearer-token authentication (`--require-auth` +
  `--auth-tokens-file`), constant-time validated. Default bind is
  `127.0.0.1`; `--bind 0.0.0.0` warns when used.
- Twelve supported (GB10, model, quant) targets across Qwen3.5 /
  Qwen3.6 / Qwen3-Next / Qwen3-VL / Gemma-4 / Mistral-Small-4 /
  MiniMax-M2.7 / Nemotron-H families.
- mdBook documentation at `book/src/`, rustdoc at `target/doc/`,
  Docker image `avarok/atlas-gb10:latest`.

### Engineering notes

For the kernel-level perf history — long-context regression sweeps,
the parking_lot migration, the libcuda + libnccl CI stubs, the
multi-stage scheduler refactor — see
[`docs/ATLAS_SPARK_JOURNEY.md`](docs/ATLAS_SPARK_JOURNEY.md) and the
[`book/`](book/) chapters under `deep-dives/`.

[Unreleased]: https://github.com/Avarok-Cybersecurity/atlas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Avarok-Cybersecurity/atlas/releases/tag/v0.1.0
