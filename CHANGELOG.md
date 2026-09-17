# Changelog

All notable changes to Atlas are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For per-release deep dives — kernel-level wins, the engineering history
behind specific subsystems — see the
[Atlas Spark Journey](docs/ATLAS_SPARK_JOURNEY.md).

## [Unreleased]

### Added
- **`kernels/r9700`, an AMD Radeon AI PRO R9700 (gfx1201, RDNA 4) SCALE
  target.** A structural mirror of `kernels/strix` — same SCALE 1.7.1
  toolchain through `targets/gfx1201`, same curated 99-entry `common/` reached
  by relative symlink, same `qwen3.6-27b` kernel tree — plus `qwen3.8-27b`
  through `kernel_source`. It landed **UNVERIFIED ON SILICON**, inheriting
  strix's gfx1151 bring-up decisions (the 64 KB LDS `BR64 32` prefill pin and
  the `serve-amd.sh` runtime knobs) unexamined, with RDNA 4's native FP8 WMMA
  and LDS cap named as the first thing to probe. No `BENCH.toml` and no
  `[benchmarks.limits]`: nothing about this class has been measured, so it
  cannot be campaigned. The kernel set has since compiled on real gfx1201
  silicon (97/97 `.cu`, 94-kernel
  `spark-server` build green), which settled two of those inherited
  decisions as gfx1201 facts rather than carry-overs: RDNA 4 has the same
  64 KB per-workgroup LDS cap, so the `BR64 32` prefill pin is required, and
  SCALE emits no e4m3 MMA codegen there, so `ATLAS_W4A16_VARIANT=v1` is
  required. Coherent generation is still unobserved.
- **An `r9700` entry in `hardware_id_from_gpu_name`**, reached both by the
  `gfx1201` arch string and by the `Radeon AI PRO R9700` marketing name,
  because `lspci` on that board reports only a numeric device id. A bench
  receipt from this card now names its class instead of keying itself by the
  punctuation-stripped GPU string while the registered `r9700` baseline slot
  sits unused. Strix stays unmapped on purpose.

### Changed
- **Free GPU memory comes from amdgpu sysfs, not the CUDA driver, on SCALE
  builds.** Measured 2026-09-17 on an AMD Radeon AI PRO R9700 (gfx1201, SCALE
  1.7.1, ROCm 7.2.0): loading `unsloth/Qwen3.8-27B-NVFP4` through the fast
  loader, Atlas logged `Shard 1/2 done, GPU memory: 31.56 GB used, 0.05 GB
  free` while `mem_info_vram_used` peaked at 22.9 GB of a 31.86 GB board with
  22.57 GB of tensors on the allocation ledger. A two-loop repro in one
  program isolates it as an allocation-COUNT effect: 56 x 512 MiB tracks
  sysfs within 1 percent, 2000 x 11 MiB reports 60 MiB free at 16500 MiB
  allocated against sysfs used 17227 MiB of 32624 (about 15 GB genuinely
  free), stays at 64 MiB free through 22000 MiB allocated and never recovers
  after every allocation is freed, and native HIP `hipMemGetInfo` in the same
  loop is honest. It is a SCALE runtime reporting defect, roughly 16 MiB of
  phantom usage charged per allocation to its own accounting, with real VRAM
  use unaffected. Since every memory guard in Atlas keys off that number (the
  fast loader's OOM guard, the pre-flight estimate, the KV sizer, the OOM
  watchdog, the TUI gauge), `free_memory`, `device_free_memory` and the
  watchdog poll now read `mem_info_vram_total` minus `mem_info_vram_used`
  from the board's `/sys/class/drm/card*/device`, auto-detected by matching
  its total against the driver's within 5 percent. Those counters are the
  kernel's own accounting across every process, so they also see the desktop
  compositor. `total` stays on the driver, which was correct.
  `ATLAS_MEMINFO_SOURCE=driver|sysfs|sysfs:<dir>` overrides either way.
  NVIDIA is untouched: without `cfg!(atlas_scale)` the source resolves to the
  driver without so much as scanning `/sys`.
- **`build-amd.sh` and `serve-amd.sh` take their hardware from
  `ATLAS_TARGET_HW`** (default `strix`, so an unset environment builds and
  serves what it always did) and read the SCALE arch from
  `kernels/$ATLAS_TARGET_HW/HARDWARE.toml` rather than hardcoding `gfx1151`
  in four places each, so `ATLAS_TARGET_HW=r9700 ./build-amd.sh` and
  `ATLAS_TARGET_HW=r9700 ./serve-amd.sh unsloth/Qwen3.8-27B-NVFP4` drive the
  gfx1201 board with the same two scripts. `ATLAS_TARGET_MODEL`, the served
  model and `GPU_UTIL` follow the hardware. `serve-amd.sh` now exports
  `ATLAS_W4A16_VARIANT=v1` (every target) and, for r9700 only as a
  first-serve default pending the gfx1201 bisect, `ATLAS_NO_GDN_FP8_PREFILL=1`:
  `ATLAS_FORCE_GLOBAL_GDN` and `ATLAS_NO_FP8_PREDEQUANT` have no reader
  anywhere in the tree, so exporting them advertised a control that does not
  exist.
- **`atlas_scale` and `atlas_hip` are driven by `[hardware].vendor`, not by
  the target's name.** `spark-model/build.rs` and `spark-runtime/build.rs`
  tested `ATLAS_TARGET_HW.starts_with("strix")`, which was correct only while
  every SCALE target was named strix-something. The kernel side is not name-
  keyed — `prefill_paged_compute.cuh` pins `BR64 32` under `__SCALE__` for
  every SCALE target — so a second one under another name would have compiled
  32-row prefill kernels and launched them with the 64-row host grid stride,
  silently dropping query rows 32..63 of every band with no build error.
  Behaviour is byte-identical for `strix` (`amd`), `strix-hip` (`hip`), the
  NVIDIA targets and an unset `ATLAS_TARGET_HW`.

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

### Added
- **Six Hopper-owned decode kernels under `kernels/hopper/common`**, declared in
  that target's `[kernels] overrides`. Three REPLACE their GB10 namesakes — the
  W8A16 M=1 decode GEMV family, which on an H100 is 71% of the single-stream
  decode step and was LSU-bound on a shared-memory E4M3 LUT gather rather than
  bandwidth-bound. The override decodes with `cvt.rn.f16x2.e4m3x2` and keeps
  four chunk loads in flight: **C=1 TPOT 17.87 → 14.14 ms (−20.9%)** on
  1×H100 80 GB with `Qwen/Qwen3.8-27B-FP8`, 1.66–2.05× per shape, 2,689 GB/s on
  the fused gate+up — and **bit-identical**, `unequal=0` on all seven production
  shapes. GB10's own sources are untouched and still compiled by gb10, b200,
  strix and strix-hip.
  Three are ADDITIONS with new stems: `w8a16_gemm_m16.cu`,
  `dense_gemm_m16_bf16.cu` and `w8a16_gemv_ncol.cu`, the m16n8k16 tensor-core
  tiers for 5..32-row decode. They are **not** in `kernels/gb10`, so a GB10
  build does not compile a kernel it has no receipt for.

### Changed
- **Five `[defaults]` rows for those tiers, and they do not all say yes.**
  `attn_m16_tc` and `lm_head_m16_tc` are ON for Hopper (+5.3% and +4.1% C=16
  aggregate), `ffn_m16_tc` is OFF on a measured LOSS (−5.2%) from the same
  kernel on a different projection family, and `attn_ncol_gemv` is OFF because
  no serving A/B exists for it on any target. Hopper's `lm_head_batchm_max`
  widens 8 → 16, in the same commit as the arm it was measured beside. Every
  row is off (or the frozen baseline) on GB10 and B200, so neither target's
  serve changes.
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
  binary by `build.rs` as `atlas_kernels::TARGET_DEFAULTS`. A kernel-path lever
  that differs between one target and another resolves from that declaration
  FIRST and the environment second, so a serve reproduces its measured
  configuration with no `ATLAS_*` prefix at all, and prints one
  `target defaults (<hw>): …` line naming every resolved value and which of them
  came from the environment. GB10's declaration restates the previous hardcoded
  defaults exactly, asserted as an equality in
  `atlas-kernels/tests/target_defaults.rs`, so GB10 behaviour is unchanged. The
  first lever to differ is `ssm_batched_recurrent`, which `kernels/hopper`
  declares ON.
- **`ATLAS_SSM_BATCHED_RECURRENT=0` now means OFF.** It was read as `== "1"`,
  so `=0` was indistinguishable from absent — which cannot express "off" once a
  target's default can be ON, leaving an operator no way to turn a lever off
  without editing a launch script. `VAR=1` is unchanged, and the `ATLAS_NO_*`
  kill switches stay presence-gated. `ATLAS_GDN_PREFILL_TC` joins it as
  `[defaults] gdn_prefill_tc`: it was presence-gated, so `=0` used to mean ON
  and now means OFF. Every A/B recipe for it set `=1` and is unaffected.
- `kernels/<hw>/HARDWARE.toml` also gained `[hardware] sm_count`, cross-checked
  at boot against the driver's `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`; a
  mismatch logs one warning naming both numbers and serving continues.

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
