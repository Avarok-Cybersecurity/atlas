# Changelog

All notable changes to Atlas are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For per-release deep dives — kernel-level wins, the engineering history
behind specific subsystems — see the
[Atlas Spark Journey](docs/ATLAS_SPARK_JOURNEY.md).

## [Unreleased]

### Added
- `spark benchmark <list|run|history>` — the dashboard's benchmark suite as a
  headless subcommand, driving the same executor. Machine-readable output on
  stdout, progress on stderr; exit codes separate a broken harness (1) from a
  failed gate (2).
- `--version`, sourced from the packaged version so a build cannot report a
  version it was not packaged as.

### Fixed
- **Benchmark runs no longer overwrite each other.** History files were named by
  whole seconds, so two runs of the same benchmark within the same second
  silently destroyed the first. Records are now keyed by nanosecond with an
  explicit collision guard.
- Run history records the parameters, target, source and version alongside the
  result. Previously only the result was stored, so a number could not be
  attributed to a configuration or reproduced. Pre-existing files still load.

### Changed
- **The tensor-core GDN chunked-prefill family is ON by default on Hopper.**
  `kernels/hopper/HARDWARE.toml` `[defaults] gdn_prefill_tc = true` — the state
  spine and both Hopper prefill remnant twins. H100 round 13 measured it on one
  binary against a same-round control: C=1 TTFT 269.1 → 162.4 ms on 1193/256 and
  889.3 → 491.5 ms on 4593/512, C=16 aggregate +21.5% / +31.4%, coherency 4/4,
  determinism 8/8 identical over three runs, and nsys pricing the two twins at
  4.28× (`chunk_fwd_o_hopper`) and 1.60× (`recompute_wu_hopper`) with the shared
  spine kernel unchanged at 0.99× as the internal control. `kernels/gb10` and
  `kernels/b200` keep `false` — this is an H100 receipt. `ATLAS_GDN_PREFILL_TC=0`
  turns the whole family off and `ATLAS_NO_GDN_PREFILL_TC_REMNANTS=1` keeps the
  spine while pinning the twins to their parents; both print on the serve's
  `target defaults (hopper): …` line. Numbers: `GDN-PREFILL-ATTRIBUTION.md`.
- **Serving defaults are now per-hardware-target and live in the repository.**
  `kernels/<hw>/HARDWARE.toml` gained a `[defaults]` table, baked into the
  binary by `build.rs` as `atlas_kernels::TARGET_DEFAULTS`. Every kernel-path
  lever that differs between GB10 and H100 — the cuBLASLt scope, the M16
  tensor-core tiers, the BF16 head's batched-GEMV band, the batched GDN
  recurrence — resolves from that declaration FIRST and the environment second.
  An H100 serve therefore reproduces its measured configuration with no
  `ATLAS_*` prefix at all, and a serve prints one `target defaults (<hw>): …`
  line naming every resolved value and which of them came from the
  environment. GB10's declaration restates the previous hardcoded defaults
  exactly, asserted as an equality in `atlas-kernels/tests/target_defaults.rs`,
  so GB10 behaviour is unchanged.
- **`ATLAS_*=0` now means OFF for the kernel-path toggles** it previously left
  ON (`ATLAS_ATTN_M16_TC`, `ATLAS_FFN_M16_TC`, `ATLAS_M16_TC`,
  `ATLAS_LM_HEAD_M16_TC`, `ATLAS_ATTN_NCOL_GEMV`,
  `ATLAS_SSM_BATCHED_RECURRENT`). They were presence-gated, which cannot
  express "off" — and once a target's default can be ON, an operator needs to
  be able to turn one off without editing a launch script. `VAR=1` is unchanged
  everywhere, and the `ATLAS_NO_*` kill switches stay presence-gated.
  `ATLAS_GDN_PREFILL_TC` joins that list: it is now `[defaults] gdn_prefill_tc`
  (and the kernel it reaches stays shared), so `=0` means off there too.
- **A hardware target declares which kernels it OWNS**, in
  `kernels/<hw>/HARDWARE.toml` `[kernels] overrides`. `kernels/hopper` and
  `kernels/b200` are otherwise pure symlink mirrors of `kernels/gb10`; a
  Hopper-tuned kernel is now a real file under `kernels/hopper/common` that
  either REPLACES a gb10 namesake (gb10 keeps its own copy — four other targets
  compile it) or ADDS a stem gb10 does not have. Declared rather than merely
  tolerated: an undeclared regular file in a mirror is a silent fork of a
  shared kernel, and nothing on disk tells the two apart.
  `scripts/check_kernel_shadows.py` RULE 3 and
  `atlas-kernels/tests/inherited_overrides.rs` check it from the one table.

### Removed
- **The split-K W8A16 decode GEMV** (`w8a16_gemv_splitk`,
  `ATLAS_FFN_DOWN_SPLITK`, `layers::ops::w8a16_decode_gemv`). It was written
  for the down projection's ~1.2-wave grid on a 132-SM H100 and measured a null
  there (58.9 us staged scalar vs 61.8 us split-K), and 1.5x slower on the
  short-N k/v shape it was also aimed at. Nothing dispatched it by default.

### Added

- DeepSeek-V4-Flash support on GB10: native MXFP4 (E8M0) routed-expert
  loading (transcode-free — no MXFP4→BF16→NVFP4 double-quant) plus the
  Phase-K E8M0 GEMM kernels, end-to-end. (#293)
- `/v1/completions` legacy-API parity: `echo`, integer `logprobs` (four
  parallel-array `CompletionLogprobs` block), `n`, `stream_options`, and
  accepted-but-ignored `user`/`suffix`/`best_of`; prompt-position logprob
  collection during prefill. (#291)
- Native U8 NVFP4 loading for pre-quantized checkpoints. (#257)
- Holo-3.1-35B-A3B / Holo-3.1-0.8B / Ornith-1.0-9B model support on GB10
  (sm_121): hybrid Gated-DeltaNet + full-attention + (256-expert MoE | dense
  FFN) + Qwen3-VL vision tower. Brings CUTLASS Sm120 NVFP4 grouped MoE, FLA
  chunked-scan GDN prefill + wmma DV-block decode, cuBLASLt/CUTLASS attention
  projections, kernel-batched co-dispatch prefill, radix-KV + Marconi
  SSM-snapshot prefix caching, and self-relative auto KV budget. (#203)
- GEMM-based Qwen3-VL ViT attention kernel (tensor-core SDPA replacing the
  warp-per-query kernel) + tensor-core ViT block GEMMs + batched multi-image
  forward — ~2× image-request TTFT on GB10. (#202)

### Fixed

- SSM snapshot eviction is now recency-only: the hit-weighted score was
  pinning fossil anchors and inflating warm TTFT; the pure-LRU/winner-only
  policy restores warm-TTFT parity with llama.cpp. (3d8130d0)
- 35B agentic-wall recipe: SSM tail-protect brings webserver_ok
  Σ(wall_time) from 2765s to 1364s (<1500s gate). (#278)
- Weight-only NVFP4 (W4A16) checkpoints now load. llm-compressor
  `nvfp4-pack-quantized` with `input_activations: None` ships no static
  activation scale; the loader previously required `input_global_scale` and
  failed (e.g. `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`). The
  field is loaded-but-unused (activations are quantized dynamically), so it is
  now optional. W4A4/W4A8 checkpoints are unaffected. (#203)
- `--gpu-memory-utilization` now enforces a hard ceiling on total GPU
  memory (weights + buffers + KV cache + reserves), matching the vLLM /
  sparkrun convention.  Previously the fraction was applied only to
  post-weight free memory, causing the KV cache to over-allocate by
  20-27 GB when values below the ~0.88 default were used.  This blocked
  multi-service co-residency on shared-memory systems (e.g. DGX Spark
  GB10).  The flag now behaves as documented: `0.50` on a 120 GB device
  caps Atlas at ~60 GB total.  (#180)

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
