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
  target.** The unmodified CUDA sources recompiled through SCALE 1.7.1
  `targets/gfx1201`, with `kernels/r9700/common/` and
  `kernels/r9700/qwen3.6-27b/nvfp4/` as whole-directory relative-symlink
  mirrors of the gb10 tree, the shape `kernels/hopper` and `kernels/b200`
  already use. Four models: `qwen3.6-27b`, `qwen3.8-27b`, `ornith-1.0-9b` and
  `holo-3.1-4b`, the last three through `kernel_source = "qwen3.6-27b"`, so one
  compiled kernel set serves all four. Measured on gfx1201 (Radeon AI PRO
  R9700, SCALE 1.7.1, ROCm 7.2.0), 2026-09-17: 180 of the 193 gb10 `.cu`
  compile, and `unsloth/Qwen3.8-27B-NVFP4` serves coherently at 19.42 GB of
  resident weights and ~30.9 tok/s decode. Sampling and behaviour defaults are
  still gfx1151 inheritances. No `BENCH.toml` and no `[benchmarks.limits]`:
  nothing about this class has been measured, so it cannot be campaigned. See
  the r9700 section of `docs/HARDWARE.md`.
- **`AVAROK_LOAD_RELEASE_SOURCES`, release-on-consume for checkpoint tensors a
  loader has finished requantising.** `WeightStore::release_tensor` frees one
  entry's device allocation during the layer loop and marks it consumed;
  `prune_after_load`, the existing answer to this shape, runs after the whole
  load and is thirty-six layers too late on a 32 GB board. `1`/`0`, defaulting
  to `cfg!(avarok_scale)`: ON for SCALE/AMD, OFF for NVIDIA, where an unset
  variable leaves every path byte-identical. Wired into the Qwen3.5-dense
  loader at three sites (attention q/k/v/o, the GDN projections, the FP8 tail
  MLPs), worth **9.94 GiB** on `unsloth/Qwen3.8-27B-NVFP4`. A release claims a
  tensor only when its `.weight` is FP8 E4M3, which is proof rather than
  heuristic: `dense_auto` returns the store's own pointer for a BF16 tensor and
  allocates for an FP8 one, so an FP8 projection cannot reach a layer except
  through a fresh allocation. A read after release is reported as released,
  naming the knob, instead of as a missing key or a pointer into freed memory.
- **`AVAROK_LOAD_TRANSPOSED_TWINS`, a lever for the transposed second weight
  layout.** Every NVFP4 projection is kept in two layouts: the packed
  `[N, K/2]` original decode reads, and a transposed `[K, N/2]` twin the fast
  prefill GEMMs consume. On a 32 GB R9700 serving `unsloth/Qwen3.8-27B-NVFP4`
  the twins are **12.74 GiB** (8.96 dense FFN, 2.90 SSM, 0.88 attention), which
  is the difference between loading the model and not. `1` builds them (the
  default on every non-SCALE target), `0` builds none, `auto` builds them only
  if free VRAM after the checkpoint is resident clears their projected bytes
  plus a 4 GiB serve reserve; unset takes `cfg!(avarok_scale)`, which is `0` on
  SCALE and `1` elsewhere. Decided once before any layer allocates, for the
  reason `gemma4/loader_a.rs::ffn_transpose_fits` gives. The cost is not the
  same on every target: on GB10 the twins are a large prefill win (`w4a16_gemm`
  ~7.0 TFLOP/s against ~51 for `w4a16_gemm_t_m128` on Gemma-4-31B), while on
  gfx1201 the twin arm measures ~1 TFLOP/s against ~4 for the plain
  `w4a16_gemm`, so the SCALE default is `0` rather than `auto` and there is no
  trade left to weigh. Decode is untouched on every target. Dropping the SSM
  twin also drops the 1.41 GiB `out_proj` FP8 predequant and the NVFP4-MMQ
  finalize, which exist only to feed the same transposed GEMM.
- **The checkpoint's FP8 `lm_head` is released once the heads are built.**
  `unsloth/Qwen3.8-27B-NVFP4` ships `lm_head.weight` as FP8 E4M3 with a
  per-channel BF16 scale; `load_lm_head` dequantises it into a fresh BF16
  allocation and every head is built from that copy, so the checkpoint's own
  bytes have no reader. **1.18 GiB**, released by
  `lm_head_setup::release_lm_head_source` on the `AVAROK_LOAD_RELEASE_SOURCES`
  knob. Four guards, each a consumer that would otherwise still hold the
  pointer: the checkpoint's head must actually be FP8 (the proof a copy was
  made; on a BF16 or NVFP4-prepacked head the loader returns the store's own
  pointer and releasing it is a use-after-free), and none of
  `--lm-head-dtype fp8`, `--dflash` or `--speculative` may be set. It runs
  immediately after `setup_lm_heads` and BEFORE the KV sizer reads free memory,
  which is why it is not in `prune_after_load`: a release the sizer cannot see
  is a release the KV cache does not get.
- **`--text-only`: serve a multimodal checkpoint without its vision tower.**
  A declared `vision_config` binds the tower, which on
  `unsloth/Qwen3.8-27B-NVFP4` is ~1.65 GiB resident for the life of the
  process, charged against the same budget as the weights, the buffer arena and
  the KV cache. The flag clears `config.vision` before the weight store is
  built, so the tower's bytes are never read from disk, never bound and never
  resident; image and video inputs are then refused with a 400 naming the
  reason instead of being silently dropped. Default off.
- **The memory budget is itemised in the serve log.** A `KV budget itemised`
  INFO line names the weights (`WeightStore::resident_bytes`), the buffer arena
  (new `BufferArena::total_bytes`), the bytes already released (vision tower,
  `lm_head` source) and prints the unattributed remainder as `other` rather
  than folding it into "pre-KV"; `Preflight reserve` prints the reserve as
  `fixed + ring` beside the existing `slots x seqs x bytes` formula; and the
  per-component `Preflight reserve breakdown` line moves from `debug` to
  `info`, since it is once per serve and reading it used to cost a re-run under
  `RUST_LOG=debug`.
- **`docs/porting/r9700-residency.md`, the measured weight residency of
  `unsloth/Qwen3.8-27B-NVFP4` on a 32 GB R9700.** Every allocation site of the
  failing serve (2631 live allocations, 33.73 GB, dead at layer 28 of 64 on a
  167,772,160-byte request) reproduces to the tenth of a MiB from shape
  arithmetic over `MODEL.toml` plus the GDN head geometry, which is what makes
  the extrapolation trustworthy: the steady-state resident set is
  **47.07 GiB (50.54 GB)**, of which 12.74 GiB is the transposed second layout
  and 9.94 GiB is dead store. Release-on-consume is necessary and not
  sufficient; the doc ranks and costs what else would have to change.
- **An `r9700` entry in `hardware_id_from_gpu_name`**, reached both by the
  `gfx1201` arch string and by the `Radeon AI PRO R9700` marketing name,
  because `lspci` on that board reports only a numeric device id. A bench
  receipt from this card now names its class instead of keying itself by the
  punctuation-stripped GPU string while the registered `r9700` baseline slot
  sits unused. Strix stays unmapped on purpose.
- `spark benchmark <list|run|history>` — the dashboard's benchmark suite as a
  headless subcommand, driving the same executor. Machine-readable output on
  stdout, progress on stderr; exit codes separate a broken harness (1) from a
  failed gate (2).
- `--version`, sourced from the packaged version so a build cannot report a
  version it was not packaged as.

### Changed
- **`kernels/r9700` compiles gb10's whole kernel set, minus what SCALE cannot
  build for gfx1201.** The target shipped as a copy of strix's shape: a
  hand-curated 99-entry `common/` and four `qwen3.6-27b/nvfp4` shadows. A
  curated list has to be updated by hand when gb10 gains a kernel, and it was
  not: `kernels/gb10/common/` grew `dense_gemv_bf16_batch2.cu`,
  `qwen3_ssm::init` began resolving it with a hard `gpu.kernel(...)?`, and
  serving `unsloth/Qwen3.8-27B-NVFP4` on a real R9700 loaded all 21.8 GB of
  weights and then died in model build at `Module 'dense_gemv_bf16_batch2' not
  loaded`. The mirror is now whole-directory, so a kernel added to gb10 reaches
  this target without anyone remembering: `common/` is 178 entries against the
  old 99, and the model dir is 14 against the old 5. Both `KERNEL.toml`s are
  symlinks into gb10 as well, which carries the roughly 40 `[modules]` renames
  the strix copy had fallen behind on; its one SCALE-specific line, the clang
  spelling `-ffp-contract=off` of the `--fmad=false` contraction pin, moved to
  `kernels/r9700/HARDWARE.toml` `[build] extra_nvcc_flags`. What the mirror
  subtracts is a per-file SCALE 1.7.1 compile census over all 193 `.cu` of both
  gb10 directories: 13 sources failed and are not linked, along with the one
  header only nine of them include. Nine are the asymmetric-KV paged-prefill
  kernels, which die in `prefill_paged_compute_asym.cuh:99:28: error: local
  memory (70416 or 70432) exceeds limit (65536)` because that header hardcodes
  `BR64 64` and carries none of the `#if defined(__SCALE__) #define BR64 32`
  pin its symmetric sibling has; adding that pin is the follow-up that brings
  them back. The others are `gated_delta_rule_fla.cu` (`unknown opcode:
  fence.proxy.async.shared::cta`, an sm_90 async-proxy fence SCALE does not
  lower), `w4a16_fp8_ldmab.cu` and `w4a16_gemm_v2.cu` (the e4m3 MMA path SCALE
  has no codegen for on gfx1201), and `w4a4_gemm.cu`. Every entry point of the
  13 is declared `[expected_absent]` with the census error line as its reason,
  so the boot audit reports a stated absence instead of refusing to serve.
- **Free GPU memory comes from amdgpu sysfs, not the CUDA driver, on SCALE
  builds.** Loading `unsloth/Qwen3.8-27B-NVFP4` on the R9700, Atlas logged
  `GPU memory: 31.56 GB used, 0.05 GB free` while `mem_info_vram_used` peaked
  at 22.9 GB of a 31.86 GB board with 22.57 GB of tensors allocated. A two-loop
  repro isolates it as an allocation-COUNT effect: 56 x 512 MiB tracks sysfs
  within 1 percent, while 2000 x 11 MiB reports 60 MiB free at 16500 MiB
  allocated against sysfs used 17227 MiB of 32624, stays at 64 MiB free through
  22000 MiB allocated, and never recovers after every allocation is freed;
  native HIP `hipMemGetInfo` in the same loop is honest. It is a SCALE runtime
  reporting defect, roughly 16 MiB of phantom usage charged per allocation to
  its own accounting, with real VRAM use unaffected. Since every memory guard
  keys off that number, `free_memory`, `device_free_memory` and the watchdog
  poll now read `mem_info_vram_total` minus `mem_info_vram_used` from the
  board's `/sys/class/drm/card*/device`, auto-detected by matching its total
  against the driver's within 5 percent. Those counters are the kernel's own
  accounting across every process, so they also see the desktop compositor.
  `total` stays on the driver, which was correct.
  `AVAROK_MEMINFO_SOURCE=driver|sysfs|sysfs:<dir>` overrides either way. NVIDIA
  is untouched: without `cfg!(avarok_scale)` the source resolves to the driver
  without so much as scanning `/sys`.
- **`build-amd.sh` and `serve-amd.sh` take their hardware from
  `AVAROK_TARGET_HW`** (default `strix`, so an unset environment builds and
  serves what it always did) and read the SCALE arch from
  `kernels/$AVAROK_TARGET_HW/HARDWARE.toml` rather than hardcoding `gfx1151` in
  four places each, so `AVAROK_TARGET_HW=r9700 ./build-amd.sh` and
  `AVAROK_TARGET_HW=r9700 ./serve-amd.sh unsloth/Qwen3.8-27B-NVFP4` drive the
  gfx1201 board with the same two scripts. The served model, the KV
  utilisation, the batch size and the OOM guard follow the hardware.
  `serve-amd.sh` exports `AVAROK_W4A16_VARIANT=v1` on every target and, on
  r9700 only, `AVAROK_NO_GDN_FP8_PREFILL=1`, `AVAROK_NO_FP8_PREDEQUANT=1` and
  `AVAROK_LOAD_TRANSPOSED_TWINS=0`. `AVAROK_FORCE_GLOBAL_GDN` has no reader
  anywhere in the tree and is no longer exported.
- **`avarok_scale` and `avarok_hip` are driven by `[hardware].vendor`, not by
  the target's name.** `spark-model/build.rs` and `spark-runtime/build.rs`
  tested `AVAROK_TARGET_HW.starts_with("strix")`, which was correct only while
  every SCALE target was named strix-something. The kernel side is not
  name-keyed: `prefill_paged_compute.cuh` pins `BR64 32` under `__SCALE__` for
  every SCALE target, so a second one under another name would have compiled
  32-row prefill kernels and launched them with the 64-row host grid stride,
  silently dropping query rows 32..63 of every band with no build error.
  Behaviour is byte-identical for `strix` (`amd`), `strix-hip` (`hip`), the
  NVIDIA targets and an unset `AVAROK_TARGET_HW`.
- **`scripts/check_kernel_shadows.py` RULE 3 now also catches undeclared
  OMISSIONS, and covers `r9700`.** A mirrored `common/` had to declare every
  regular file it owned (`[kernels] overrides`); it may now also declare every
  origin entry it deliberately does not carry (`[kernels] absent`), and the two
  sets are checked against the tree from both directions. `r9700` joins
  `hopper` and `b200` in `MIRRORED_COMMON`, which is why the silent shrink
  above cannot recur: a gb10 kernel with no counterpart here and no declaration
  is a violation. The Rust-side `INHERITED` list stays NVIDIA-only, since its
  assertions are about the Hopper/B200 campaign's `HARDWARE.toml` and
  `MODEL.toml` parity rather than about mirroring.

### Fixed
- **A kernel module that compiled to nothing no longer takes the first launch
  down on SCALE.** `spark serve` on the R9700 loaded 21.8 GB of weights and
  died at its first kernel with `CUDA_ERROR_INVALID_IMAGE (200)` on
  `nvfp4_mmq::avarok_nvfp4_repack`. `nvfp4_mmq.cu` is entirely inside
  `#if defined(BLACKWELL_MMA_AVAILABLE)`, so on a non-Blackwell target it
  compiles to a code object with no kernel symbols at all. NVIDIA answers
  `cuModuleGetFunction` for such a name with "not found", `try_kernel` folds
  that into `KernelHandle(0)`, and the guarded use site takes another path.
  SCALE answers SUCCESS and returns a handle backed by no code, so the guard
  never fires and the launch is the first thing that notices. The registry now
  reads each binary module's ELF symbol table at load time, through
  `crates/avarok-core/src/elf_symbols.rs` (a dependency-free ELF64 walk over
  `STT_FUNC` symbols and AMDGPU `<kernel>.kd` descriptors, bounds-checked
  throughout, declining anything it cannot parse), and refuses a lookup the
  object provably cannot satisfy with `<module>::<kernel>: not defined in this
  target's code object (optional module compiled out?)`. That error degrades to
  handle 0 through the same probe NVIDIA uses. Unparsable objects and the PTX
  path are untouched. `avarok-kernels`' build script reads the same objects
  with the same code and names every empty module in the build log.
- **The `CompressedTensors` attention dequant leak is fixed unconditionally,
  and three direct store frees now go through `WeightStore::release_tensor`.**
  The leak is 200 MiB per full-attention layer, **3.12 GiB** across the sixteen
  of `unsloth/Qwen3.8-27B-NVFP4`, on every target including NVIDIA: every
  sibling site frees its BF16 dequant intermediate and this one never did. It
  is not gated on `AVAROK_LOAD_RELEASE_SOURCES`, because what byte-identical
  NVIDIA behaviour protects is which values the GEMMs read, and the leaked
  buffer has no reader: `quantize_to_nvfp4` has already consumed it and
  `AttentionWeights` keeps only the NVFP4 result and the two norm pointers.
  Separately, the GDN concat inputs, the GDN `out_proj` input and the `Bf16Raw`
  arm of `quantized_any` freed pointers the store still listed, so teardown
  freed them again, in the last case on every raw BF16 fine-tune Atlas serves.
  Residency is unchanged to the byte; what changes is that the store now
  forgets what was freed, so `contains` and `get` stop claiming memory that is
  gone and a late reader gets a named error instead of whatever the allocator
  handed out next.
- **The FP8 prefill predequant is no longer built on a target whose FP8 prefill
  GEMM does not exist.** On the R9700, `Ornith-1.0-9B` loads, builds and boots,
  then every request dies at layer 0 with `ssm prefill: out_proj GEMM failed:
  Kernel lookup w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab: Module load failed: Module
  'w4a16_fp8_ldmab' not loaded`. `predequant_for_prefill` built `out_proj_fp8`
  (and, on the loaders that call it, `q_fp8`..`o_fp8` and the MoE's gate and
  shared-expert copies) without asking whether anything could read them, and
  the prefill dispatch PREFERS those copies over both NVFP4 arms, so one absent
  module turned a working fallback chain into a hard error.
  `layers/fp8_predequant.rs` is now the load-time guard: it probes every arm
  `ops::fp8_gemm_n128` and `fp8_gemm_n128_m128` can take, not just the
  preferred one, and it restores a reader for `AVAROK_NO_FP8_PREDEQUANT`. With
  the copies absent the SSM falls to `w4a16_gemm_n128` and then `w4a16_gemm`,
  attention's `use_fp8_act` goes false, and the MoE's three `if let Some` arms
  take their NVFP4 branch. NVIDIA is unchanged: with the kernels present and
  the variable unset the guard returns "build them".
  `AVAROK_NO_FP8_PREDEQUANT=0` forces the copies back where an operator's
  environment sets the variable globally; it cannot override a kernel that is
  genuinely absent.
- **`detect_nvfp4_variant` probes `checkpoint_dtype`, not `get`.** It runs both
  before the layer loop and again after it (`load_mtp_weights`,
  `prune_after_load`, `detect_quant_format`) and has to give the same answer
  both times. Under release-on-consume it would otherwise detect a block-FP8
  compressed-tensors checkpoint as `Fp8Dequanted` on the way in and `Standard`
  on the way out, because its dtype probe reads the very projections the loader
  releases.
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
  `kernels/b200` keep `false` — this is an H100 receipt. `AVAROK_GDN_PREFILL_TC=0`
  turns the whole family off and `AVAROK_NO_GDN_PREFILL_TC_REMNANTS=1` keeps the
  spine while pinning the twins to their parents; both print on the serve's
  `target defaults (hopper): …` line. Numbers: `GDN-PREFILL-ATTRIBUTION.md`.
- **Serving defaults are now per-hardware-target and live in the repository.**
  `kernels/<hw>/HARDWARE.toml` gained a `[defaults]` table, baked into the
  binary by `build.rs` as `avarok_kernels::TARGET_DEFAULTS`. A kernel-path lever
  that differs between one target and another resolves from that declaration
  FIRST and the environment second, so a serve reproduces its measured
  configuration with no `AVAROK_*` prefix at all, and prints one
  `target defaults (<hw>): …` line naming every resolved value and which of them
  came from the environment. GB10's declaration restates the previous hardcoded
  defaults exactly, asserted as an equality in
  `avarok-kernels/tests/target_defaults.rs`, so GB10 behaviour is unchanged. The
  first lever to differ is `ssm_batched_recurrent`, which `kernels/hopper`
  declares ON.
- **`AVAROK_SSM_BATCHED_RECURRENT=0` now means OFF.** It was read as `== "1"`,
  so `=0` was indistinguishable from absent — which cannot express "off" once a
  target's default can be ON, leaving an operator no way to turn a lever off
  without editing a launch script. `VAR=1` is unchanged, and the `AVAROK_NO_*`
  kill switches stay presence-gated. `AVAROK_GDN_PREFILL_TC` joins it as
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
