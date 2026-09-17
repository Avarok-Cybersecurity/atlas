# Changelog

All notable changes to Atlas are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For per-release deep dives — kernel-level wins, the engineering history
behind specific subsystems — see the
[Atlas Spark Journey](docs/ATLAS_SPARK_JOURNEY.md).

## [Unreleased]

### Added
- **`ATLAS_LOAD_TRANSPOSED_TWINS`, a lever for the transposed second weight
  layout.** Atlas keeps every NVFP4 projection in two layouts: the packed
  `[N, K/2]` original decode reads, and a transposed `[K, N/2]` twin the fast
  prefill GEMMs consume. On a 32 GB R9700 serving
  `unsloth/Qwen3.8-27B-NVFP4` the twins are **12.74 GiB** (8.96 dense FFN, 2.90
  SSM, 0.88 attention), which is the difference between loading the model and
  not: release-on-consume and the attention dequant-leak fix take that serve
  from 47.07 GiB to 35.18, and dropping the twins takes it to 21.25. `1` builds
  them (the pre-lever behaviour byte for byte, and the default on every
  non-SCALE target), `0` builds none, `auto` builds them only if free VRAM after
  the checkpoint is resident clears their projected bytes plus a 4 GiB serve
  reserve; unset takes `cfg!(atlas_scale)`, which is `0` on SCALE and `1`
  elsewhere. Decided ONCE before any layer
  allocates, for the reason `gemma4/loader_a.rs::ffn_transpose_fits` gives.
  **The cost is written down rather than discovered**, and it is not the same
  cost on every target. On GB10 the twins are a large prefill win: `w4a16_gemm`
  measured ~7.0 TFLOP/s against ~51 for `w4a16_gemm_t_m128` on Gemma-4-31B, so
  skipping them is a 7x slower FFN prefill and every non-SCALE target still
  builds them. **On gfx1201 the sign is reversed** — the R9700 prefill
  measurement of 2026-09-17 puts the twin arm at ~1 TFLOP/s against ~4 for the
  plain `w4a16_gemm` — so the SCALE default is `0` rather than `auto`: the
  second layout costs 12.74 GiB *and* throughput there, and there is no trade
  left for the probe to weigh. The GB10 figures were never measured on SCALE.
  Decode is untouched on every target. The load line says which way it went. Every consumer
  already tolerated a `None` twin and the per-site proof is tabulated in
  `transposed_twins.rs`; nothing needed a new fallback. Dropping the SSM twin
  also drops the 1.41 GiB `out_proj` FP8 predequant and the NVFP4-MMQ finalize,
  both of which exist only to feed the same transposed GEMM. A proper fix is a
  prefill GEMM that reads the packed layout directly; this is a serve tonight.
- **`ATLAS_LOAD_RELEASE_SOURCES`, release-on-consume for checkpoint tensors a
  loader has finished requantising.** `WeightStore::release_tensor` frees one
  entry's device allocation during the layer loop and marks it consumed;
  `prune_after_load`, the existing answer to this shape, runs after the whole
  load and is thirty-six layers too late on a 32 GB board. `1`/`0`, defaulting
  to `cfg!(atlas_scale)`: ON for SCALE/AMD, OFF for NVIDIA, where an unset
  variable leaves every path byte-identical. Wired into the Qwen3.5-dense
  loader at three sites (attention q/k/v/o, the GDN projections, the FP8 tail
  MLPs), which is **9.94 GiB** on `unsloth/Qwen3.8-27B-NVFP4`. A release
  claims a tensor only when its `.weight` is FP8 E4M3, which is the proof
  rather than a heuristic: `dense_auto` returns the store's own pointer for a
  BF16 tensor and allocates for an FP8 one, so an FP8 projection cannot reach
  a layer except through a fresh allocation. A read after release is reported
  as released, naming the knob, instead of as a missing key or a pointer to
  freed memory.
- **`docs/porting/r9700-residency.md`, the measured weight residency of
  `unsloth/Qwen3.8-27B-NVFP4` on a 32 GB R9700.** Every site of the allocation
  ledger from the failing serve (2631 allocations, 33.73 GB live, dead at layer
  28 of 64 on a 167,772,160-byte request) reproduces to the tenth of a MiB from
  shape arithmetic over `MODEL.toml` plus the GDN head geometry, which is what
  makes the extrapolation trustworthy: the steady-state resident set is
  **47.07 GiB (50.54 GB)**, of which 12.74 GiB is the transposed second layout
  and 9.94 GiB is dead store. The verdict is stated rather than hedged.
  Release-on-consume is necessary and NOT sufficient; the ranked list of what
  else would have to change is in the doc and in
  `kernels/r9700/HARDWARE.toml`'s new open-questions block.
- **Two small models on `kernels/r9700`: `ornith-1.0-9b` and `holo-3.1-4b`,
  both through `kernel_source = "qwen3.6-27b"`.** Qwen3.8-27B does not fit the
  R9700's 32 GB under Atlas's two-layout residency (measured 2026-09-17), so
  coherent generation on gfx1201 has to be proved on a smaller model of the
  same architecture family first, and these two are that family at 9B and 4B
  (Gated DeltaNet plus full attention plus a dense FFN plus a Qwen3-VL ViT,
  which is `qwen3.6-27b`'s trunk). Neither mirrors its own gb10 kernel
  directory, because there is no such thing: gb10's `ornith-1.0-9b/nvfp4/`,
  `holo-3.1-4b/nvfp4/`, `holo-3.1-0.8b/nvfp4/` and `holo-3.1-35b-a3b/nvfp4/`
  are ONE byte-identical six-file fork of `qwen3.6-27b/nvfp4/` shared by four
  models spanning `hidden_dim` 1024 to 4096 and dense to MoE, and two of its
  six files (`w4a16_gemm.cu`, `moe_w4a16_grouped_gemm.cu`) fail the gfx1201
  census on e4m3 for want of the `#if defined(__SCALE__)` shims the
  `qwen3.6-27b` copies carry. That the one source set already serves four
  shapes is also the evidence that redirecting across a shape boundary is
  sound; the only `-D` naming a model dimension anywhere under `kernels/` is
  `-DHDIM=128`, in eight gb10 `KERNEL.toml`s and none of these, and all four
  models involved declare `head_dim = 256`. `[build] extra_nvcc_flags` is
  `["--fmad=false"]` on both sides character for character, and the two
  `KERNEL.toml`s otherwise differ only in `[modules]` renames, almost all of
  them modules `qwen3.6-27b` has and the fork does not ship. The redirect
  drops `fp4_mma_microtest.cu`, whose only reader in the tree is a
  `gpu-examples` sm_120 MMA microproof that has no gfx1201 lowering either
  way. No `match_names`: `validate_collision_match_names` demands needles only
  from a `(model_type, hidden_size)` pair two differently-named targets both
  declare, and `("qwen3_5", 4096)` and `("qwen3_5", 2560)` collide with
  nothing on this hardware. Their `[expected_absent]` tables are
  `qwen3.8-27b`'s verbatim, because same silicon plus same compiled tree means
  the same thirteen absences; gb10's copies declare a different set, harvested
  against the fork, and eight of its nine tables name families that resolve in
  the tree this target compiles (the `gated_delta_rule` f16 arms among them,
  which gb10's own honesty note flags as dispatched un-probed). The ninth,
  `w4a4`, really is absent here, for the unrelated reason that `w4a4_gemm.cu`
  does not compile for gfx1201, and the replacement set carries it with that
  reason. **UNVERIFIED**, like everything else on this board: nothing
  has been served, and this registers kernel targets, not loader support.
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
  required. Coherent generation is still unobserved. The curated 99-entry
  `common/` described above did not survive its first serve attempt and is
  gone: see the mirror entry under Changed, in this same unreleased set.
- **An `r9700` entry in `hardware_id_from_gpu_name`**, reached both by the
  `gfx1201` arch string and by the `Radeon AI PRO R9700` marketing name,
  because `lspci` on that board reports only a numeric device id. A bench
  receipt from this card now names its class instead of keying itself by the
  punctuation-stripped GPU string while the registered `r9700` baseline slot
  sits unused. Strix stays unmapped on purpose.

### Fixed
- **The `CompressedTensors` attention dequant leak is fixed unconditionally,
  and three direct store frees now go through `WeightStore::release_tensor`.**
  The leak (200 MiB per full-attention layer, **3.12 GiB** across the sixteen of
  `unsloth/Qwen3.8-27B-NVFP4`, on every target including NVIDIA) rode
  `ATLAS_LOAD_RELEASE_SOURCES` for one commit because that change was not
  allowed to move NVIDIA behaviour. It does not: what byte-identical protects is
  which values the GEMMs read, and the leaked buffer has no reader:
  `quantize_to_nvfp4` has already consumed it and `AttentionWeights` keeps only
  the NVFP4 result and the two norm pointers. Separately, the GDN concat inputs,
  the GDN `out_proj` input and the `Bf16Raw` arm of `quantized_any` freed
  pointers the store still listed, so teardown freed them again, on every raw
  BF16 fine-tune Atlas serves, in the last case. Residency is unchanged to the
  byte; what changes is that the store now forgets what was freed, so `contains`
  and `get` stop claiming memory that is gone and a late reader gets a named
  error instead of whatever the allocator handed out next.
- **The FP8 prefill predequant is no longer built on a target whose FP8 prefill
  GEMM does not exist.** Measured on an R9700 (gfx1201, SCALE 1.7.1):
  `Ornith-1.0-9B` loads, builds and boots, then every request dies at layer 0
  with `ssm prefill: out_proj GEMM failed: Kernel lookup
  w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab: Module load failed: Module
  'w4a16_fp8_ldmab' not loaded`. `predequant_for_prefill` built `out_proj_fp8`
  (and, on the loaders that call it, `q_fp8`..`o_fp8` and the MoE's gate and
  shared-expert copies) without asking whether anything could read them, and
  the prefill dispatch PREFERS those copies over both NVFP4 arms — so one
  absent module turned a working fallback chain into a hard error.
  `layers/fp8_predequant.rs` is now the load-time guard: it probes every arm
  `ops::fp8_gemm_n128` and `fp8_gemm_n128_m128` can take, not just the
  preferred one, and it restores a reader for `ATLAS_NO_FP8_PREDEQUANT`, which
  the Strix recipe carried until the variable was dropped as unread. With the
  copies absent the SSM falls to `w4a16_gemm_n128` and then `w4a16_gemm`,
  attention's `use_fp8_act` goes false, and the MoE's three `if let Some` arms
  take their NVFP4 branch. NVIDIA is unchanged: with the kernels present and
  the variable unset the guard returns "build them". `ATLAS_NO_FP8_PREDEQUANT=0`
  forces the copies back where an operator's environment sets the variable
  globally; it cannot override a kernel that is genuinely absent.
- **The `CompressedTensors` attention arm of the Qwen3.5-dense loader leaked
  its BF16 dequant intermediate.** Every sibling site frees its own — the
  `Standard | Fp8Dequanted` attention arm, the SSM path, `quantized_from_fp8`,
  the `Bf16Raw` arm of `quantized_any` — and this one never did. It is
  200 MiB per full-attention layer and **3.12 GiB** across the sixteen of
  `unsloth/Qwen3.8-27B-NVFP4`, on every target including NVIDIA, and it is
  visible in the R9700 ledger as 28 stale `quant_helpers.rs:98` allocations
  belonging to layers that finished building up to twenty-four layers earlier.
  The free is behind `ATLAS_LOAD_RELEASE_SOURCES` rather than unconditional
  ONLY because this change was not allowed to move NVIDIA behaviour; it should
  become unconditional once an NVIDIA serve confirms it.
- **`detect_nvfp4_variant` probes `checkpoint_dtype`, not `get`.** It runs both
  before the layer loop and again after it (`load_mtp_weights`,
  `prune_after_load`, `detect_quant_format`) and has to give the same answer
  both times. Under release-on-consume it would otherwise detect a block-FP8
  compressed-tensors checkpoint as `Fp8Dequanted` on the way in and `Standard`
  on the way out, because its dtype probe reads the very projections the
  loader releases.

### Changed
- **`kernels/r9700` compiles gb10's whole kernel set, minus what SCALE cannot
  build for gfx1201.** The target shipped as a copy of strix's shape: a
  hand-curated 99-entry `common/` and four `qwen3.6-27b/nvfp4` shadows. A
  curated list has to be updated by hand when gb10 gains a kernel, and it was
  not. `kernels/gb10/common/` grew `dense_gemv_bf16_batch2.cu`,
  `qwen3_ssm::init` began resolving it with a hard `gpu.kernel(...)?`
  (`init.rs:103`), and serving `unsloth/Qwen3.8-27B-NVFP4` on a real R9700
  loaded all 21.8 GB of weights and then died in model build at `Kernel lookup
  dense_gemv_bf16_batch2::dense_gemv_bf16_batch2: Module load failed: Module
  'dense_gemv_bf16_batch2' not loaded`. `kernels/r9700/common/` and
  `kernels/r9700/qwen3.6-27b/nvfp4/` are now whole-directory relative-symlink
  mirrors of the gb10 tree, the shape `kernels/hopper` and `kernels/b200`
  already use, so a kernel added to gb10 reaches this target without anyone
  remembering. `common/` is 178 entries (169 `.cu`, 8 `.cuh`, `KERNEL.toml`)
  against the old 99, and the model dir is 14 (12 `.cu`, the `q4k_vendor`
  directory, `KERNEL.toml`) against the old 5. The two links that used to
  reach into `kernels/strix/common/` now point straight at gb10, whose files
  they were byte-identical to. Both `KERNEL.toml`s are symlinks into gb10 as
  well, which is what carries the roughly 40 `[modules]` renames the strix
  copy had fallen behind on; its one SCALE-specific line, the clang spelling
  `-ffp-contract=off` of the `--fmad=false` contraction pin, moved to
  `kernels/r9700/HARDWARE.toml` `[build] extra_nvcc_flags`, which is where
  `build_flags.rs` puts a toolchain fact on a target whose KERNEL.tomls are
  all symlinks.

  What the mirror subtracts is a per-file SCALE 1.7.1 compile census over all
  193 `.cu` of both gb10 directories, run on the board: twelve sources failed
  and are not linked, along with the one header only they include. Nine are
  the asymmetric-KV paged-prefill kernels, which all die in
  `prefill_paged_compute_asym.cuh:99:28: error: local memory (70416 or 70432)
  exceeds limit (65536)` because that header hardcodes `BR64 64` and carries
  none of the `#if defined(__SCALE__) #define BR64 32` pin its symmetric
  sibling has; adding that pin is the follow-up that brings them back. One is
  `gated_delta_rule_fla.cu` (`unknown opcode: fence.proxy.async.shared::cta`,
  an sm_90 async-proxy fence SCALE does not lower), and two are the model
  dir's `w4a16_gemm_v2.cu` (the e4m3 MMA path) and `w4a4_gemm.cu`. All 15
  entry points are declared `[expected_absent]` in both MODEL.tomls, with the
  census error line as the reason, so the boot audit reports a stated absence
  instead of refusing to serve. The `w4a4` note in `qwen3.8-27b/MODEL.toml`
  said dispatch "falls back to `w4a4_gemm`"; that is corrected, because the
  whole `w4a4` module is absent here and both `try_kernel` lookups return
  `KernelHandle(0)`, which turns the FP4-activation prefill path off rather
  than substituting anything for it.
- **`scripts/check_kernel_shadows.py` RULE 3 now also catches undeclared
  OMISSIONS, and covers `r9700`.** A mirrored `common/` had to declare every
  regular file it owned (`[kernels] overrides`); it may now also declare every
  origin entry it deliberately does not carry (`[kernels] absent`), and the
  two sets are checked against the tree from both directions. `r9700` joins
  `hopper` and `b200` in `MIRRORED_COMMON`, which is why the silent shrink
  above cannot recur: a gb10 kernel with no counterpart here and no
  declaration is a violation. The Rust-side `INHERITED` list stays
  NVIDIA-only, since its assertions are about the Hopper/B200 campaign's
  `HARDWARE.toml` and `MODEL.toml` parity rather than about mirroring.

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
- **A kernel module that compiled to nothing no longer takes the first launch
  down on SCALE.** `spark serve` on the R9700 (gfx1201, SCALE 1.7.1) loaded
  21.8 GB of weights and died at its first kernel with
  `CUDA_ERROR_INVALID_IMAGE (200)` on `nvfp4_mmq::atlas_nvfp4_repack`.
  `nvfp4_mmq.cu` is entirely inside `#if defined(BLACKWELL_MMA_AVAILABLE)`, so
  on a non-Blackwell target it compiles to a code object with no kernel symbols
  at all. NVIDIA answers `cuModuleGetFunction` for such a name with "not
  found", `try_kernel` folds that into `KernelHandle(0)`, and the guarded use
  site takes another path. SCALE answers SUCCESS and returns a handle backed by
  no code, so the guard never fires and the launch is the first thing that
  notices. The registry now reads each binary module's ELF symbol table at load
  time, through `crates/atlas-core/src/elf_symbols.rs` (a dependency-free
  ELF64 walk over `STT_FUNC` symbols and AMDGPU `<kernel>.kd` descriptors,
  bounds-checked throughout, declining anything it cannot parse), and refuses
  a lookup the object provably cannot satisfy with `<module>::<kernel>: not defined in this
  target's code object (optional module compiled out?)`. That error degrades to
  handle 0 through the same probe NVIDIA uses. Unparsable objects and the PTX
  path are untouched. `atlas-kernels`' build script reads the same objects with
  the same code and names every empty module in the build log.
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
