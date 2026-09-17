# Adding a new hardware target or model family

Atlas's compute stack is structured around **(hardware, model, quant) tuples**.
Each tuple is a self-contained body of work: kernels are written, tuned, and
tested per-tuple. This document explains how to extend the matrix.

## Directory layout

```
kernels/
└── <hardware>/                    e.g. gb10
    ├── HARDWARE.toml              arch, sm, fp32-residual flag
    ├── <quant>/                   shared kernels for this hw + quant
    │   └── *.cu                   e.g. nvfp4/dense_gemm.cu
    └── <model>/                   per-model overrides
        ├── MODEL.toml             model_type list, sampling presets, behavior
        └── <quant>/               per-(model, quant) overrides
            └── *.cu               e.g. qwen3.6-35b-a3b/nvfp4/inferspark_prefill_h128.cu
```

The build script (`crates/atlas-kernels/build.rs`) walks this tree and
compiles every `.cu` to PTX. Model-specific files override shared
files when a name collision occurs.

## Adding a new model family

If your model is similar to an existing one (e.g., adding Qwen3.7 to the
Qwen3.5/3.6 family), the mechanical recipe is:

1. **Create the kernel target dir**:
   ```
   kernels/gb10/qwen3.7-XXB/
   ├── MODEL.toml                  copy from a similar model's MODEL.toml
   └── nvfp4/                      or fp8/ etc. depending on the quant
       └── (per-target overrides — leave empty if shared kernels suffice)
   ```

2. **Write `MODEL.toml`**:
   ```toml
   [model]
   name = "qwen3.7-XXB"
   hf_id = "Qwen/Qwen3.7-XXB"
   params = "XXB"
   active_params = "XXB"
   architecture = "Hybrid Attention + GDN + Dense FFN"

   [[model_types]]
   model_type = "qwen3_5"          # what the HF config.json says
   hidden_size = NNNN              # exact hidden dim — wins over wildcards

   # Only if ANOTHER target declares the same (model_type, hidden_size) —
   # e.g. qwen3.8-27b's config is bit-identical to qwen3.6-27b's — declare
   # checkpoint-reference needles so runtime resolution can break the tie
   # (case-insensitive substrings of the HF id / --model-name / model dir).
   # build.rs FAILS if colliding targets omit this, and a tie the needles
   # cannot break to exactly one target is a hard startup error (never a
   # build-order pick; `--kernel-target` pins explicitly). Rules + rationale:
   # crates/atlas-kernels/src/resolve.rs.
   # match_names = ["qwen3.7-XXb"]

   # Architecturally-identical sibling (zero new kernels)? Reuse another
   # target's .cu tree instead of copying it — qwen3.8-27b compiles
   # qwen3.6-27b's sources this way and ships no files of its own:
   # kernel_source = "qwen3.6-27b"

   [behavior]
   default_num_drafts = 1
   max_thinking_budget = 512
   thinking_in_tools = false

   [sampling.thinking_text]
   temperature = 0.6
   top_p = 0.95
   top_k = 20

   # ... (other sampling presets — see existing MODEL.toml files)
   ```

3. **Wire to a `WeightLoader`** in `crates/spark-model/src/factory.rs`:
   most Qwen3-family models share `Qwen35WeightLoader` for MoE and
   `Qwen35DenseWeightLoader` for dense FFN. Pick the right one based on
   whether the model has experts.

4. **Add to test sweep** (`tests/run_all_models.py`): one round per
   variant (with/without MTP, EP=2 if applicable).

5. **Build with the wildcard target**:
   ```
   ATLAS_TARGET_MODEL='*' cargo build --release -p spark-server
   ```
   The new target compiles into the binary; runtime selects it via
   `model_type` + `hidden_size` matching, with `match_names` breaking any
   tie between config-identical checkpoints (see
   `crates/atlas-kernels/src/resolve.rs`).

If your model is genuinely new (different attention pattern, novel SSM
variant, etc.), you'll also need to:
6. Write a per-architecture `TransformerLayer` impl in
   `crates/spark-model/src/layers/` (mirror the structure of
   `qwen3_attention/` or `qwen3_ssm/`).
7. Add a new `WeightLoader` in `crates/spark-model/src/weight_loader/`
   if the safetensors key naming differs from existing families.

## Adding a new hardware target

Atlas's NVIDIA targets are **GB10 (Blackwell, sm_121)**, **Hopper
(H100/H200, sm_90a)** and **B200 (B200/GB200, sm_100a)**; `strix`/`strix-hip`
(AMD gfx1151), `r9700` (AMD gfx1201) and `metal` are the non-NVIDIA sets. Adding another — say sm_120
for a consumer Blackwell board, or sm_103 for Blackwell Ultra (B300/GB300) —
requires:

1. **`kernels/<new-hw>/HARDWARE.toml`**. The keys are exactly the ones
   `crates/atlas-kernels/build.rs` reads, plus documentation:
   ```toml
   [hardware]
   name = "gb10"                   # matches the directory name
   vendor = "nvidia"               # picks the compiler: nvidia | apple | amd | hip
   arch = "sm_121f"                # forwarded verbatim to `nvcc -arch=`
   compute_capability = "12.1"     # the device CC this target serves
   memory_bandwidth_gbps = 273     # documentation / roofline input
   memory_type = "LPDDR5X"
   memory_gb = 120
   ```
   Only `arch` and `vendor` are load-bearing at build time: `arch` becomes
   `-arch=` (and reaches the registry twice — verbatim as `TargetPtxSet.ptx_arch`
   and, with any `a`/`f` feature suffix stripped, as `KernelTarget.arch`), and
   `vendor` selects the `ComputeTarget` impl in `build_target.rs` and the
   per-vendor KERNEL.toml flag key (`extra_nvcc_flags` vs `extra_metal_flags`).

   `compute_capability` has ONE reader, and it is a test:
   `crates/atlas-kernels/tests/target_hints.rs` asserts that every
   `vendor = "nvidia"` set's declared CC is what `atlas_core::arch::target_hint`
   maps back to that directory name — so the "rebuild with `ATLAS_TARGET_HW=…`"
   line an operator gets on an arch mismatch cannot drift from the tree. Get it
   right; it is no longer decoration.

   The `memory_*` keys still have **no reader anywhere in the repo** — they are
   documentation and roofline input, and `kernels/strix/HARDWARE.toml` records
   what happened to two keys that pretended otherwise.

   Get the SM number right. Hopper is **sm_90** (`sm_90a` with the
   arch-specific feature set); **sm_100** is Blackwell datacenter (B200/GB200),
   **sm_103** is Blackwell Ultra (B300/GB300), sm_120 is consumer Blackwell,
   sm_121 is GB10. PTX built for an `a`-suffixed arch does not run forward onto
   a later architecture — and these are not a ladder: sm_100a and sm_120a are
   siblings, each with instructions the other lacks (see the B200 section
   below).

   The value also has to be reachable: `crates/atlas-kernels/tests/target_hints.rs`
   asserts that `atlas_core::arch::target_hint` maps this file's
   `compute_capability` back to the directory name, so an operator whose GPU
   fails the arch preflight is told which target to rebuild.

   **Benchmark limits** (`[benchmarks.limits.{thermal,memory,timing,equivalence}]`,
   `atlas_plugin::hardware::limits`): what `spark bench certify` and the
   record policies judge a box of this class by — the chassis temperature it
   is parked at and resumed at, the chassis delta and clock/memory spreads
   under which two boxes are "one box" for a Speed record, the die ceiling a
   Speed capture is suspect above, the free-memory floor a self-start needs,
   and the serve/boot/shard/build allowances a campaign plans with. These are
   measured facts about the class (see `kernels/gb10/HARDWARE.toml` for the
   GB10's, each with its measurement beside it), never copied from another
   card: a target that declares none cannot be campaigned or self-served for a
   gate until someone measures them, and its Speed records from two boxes
   never agree. Every sub-table is required once the section exists. The file
   is a closure input, so changing a limit re-opens every gate on the target.

2. **Kernel sources**: `kernels/<new-hw>/common/` for the shared set and
   `kernels/<new-hw>/<model>/<quant>/` for per-model shadows. If the new
   target starts out compiling another target's sources unchanged, share them
   with **relative symlinks** rather than copies — see the Hopper section
   below and `kernels/strix/common/`. Where the kernels do diverge, tile
   shapes, SMEM budget and tensor-core MMA instructions are what usually
   needs tuning.

3. **`atlas-kernels/build.rs`**: usually no changes needed — the build
   script auto-discovers new `kernels/<hw>/` directories.

4. **`spark-runtime/src/cuda_backend.rs`**: if the hardware has different
   capabilities (e.g., GDS supported, no NVLink, different RDMA NIC),
   wire the relevant flags here.

5. **NCCL env**: launchers like `scripts/start-ep2.sh` hardcode
   `NCCL_SOCKET_IFNAME=enp1s0f0np0` (GB10's RDMA NIC). Update for the
   new hardware's interconnect.

6. **CI**: GitHub Actions runs on `ubuntu-latest` with `ATLAS_SKIP_BUILD=1`
   so no GPU is needed. The new target compiles via the wildcard build
   on a host with the right SM.

## The Hopper (sm_90a) target

`kernels/hopper/` is H100 and H200 — both SM 9.0. It is the worked example of
a target that ships **no kernels of its own**.

**Kernel set: inherited from gb10 by symlink.** `kernels/hopper/common/` is 188
relative symlinks into `kernels/gb10/common/` (all 178 `.cu`, the 9 `.cuh`
headers, and `KERNEL.toml`), and each of the seven model targets mirrors gb10's
`nvfp4/` directory file by file the same way. The mirror is a whole-directory
rule, not a list: a kernel added to `kernels/gb10/common/` needs a link here
or this target silently compiles a smaller inventory than GB10 does. Git stores them as symlinks
(mode 120000); nothing is copied. This works because the gb10 kernels are
written to an SM80-class instruction floor — `mma.sync.m16n8k16`, `cp.async.cg`,
with TMA and `cp.async.bulk` deliberately avoided — and carry no
`__CUDA_ARCH__` gating.

Per-file links, not a `[model] kernel_source` redirect: `kernel_source`
redirects a whole quant tree and only within one hardware set, whereas
replacing one link with one real file is how a Hopper-tuned kernel will later
shadow its gb10 origin without forking the other 180. `MODEL.toml` is a real
copy — sampling and behaviour are checkpoint properties that must stay
editable per hardware — and each copy's header says so, including that its
`[expected_absent]` tables were harvested on GB10 and **not** re-harvested on
Hopper (`spark serve --check-kernels` on a real H100/H200 is what would do
that).

`kernels/hopper/<model>/nvfp4/` despite Hopper having no NVFP4 datapath: the
runtime's weight-format gate is that an nvfp4-built kernel bundle also serves
FP8 and BF16 checkpoints, so Hopper FP8 checkpoints run through these kernels.
An `fp8/` directory would be a second name for the same files.

**Hopper is FP8/BF16-only.** The NVFP4 CUTLASS wrappers in
`crates/spark-runtime/cuda/cutlass_nvfp4_gemm.cu` are gated on
`CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED` and
compile to nothing for sm_90a. Serving an NVFP4 checkpoint on Hopper needs an
Sm90 block-scaled path that does not exist yet.

**The compile gate.** `scripts/hopper_ptx_gate.sh` answers "does each kernel
compile for this architecture" with nvcc alone — no H100 needed, no GPU of any
kind:

```bash
# On any CUDA host (nvcc need not be on PATH):
CUDA_HOME=/usr/local/cuda scripts/hopper_ptx_gate.sh --selftest
CUDA_HOME=/usr/local/cuda scripts/hopper_ptx_gate.sh \
  --hw hopper --model all --jobs 4 --out receipts/ptx_gate.json
```

It resolves each model's file set the way `build.rs` does (common/ overridden
by the model directory by file stem, and the three flag layers — HARDWARE.toml,
common/KERNEL.toml, the model's KERNEL.toml — merged least-specific-first),
runs `nvcc --ptx -arch=<arch>` then `ptxas -arch=<arch> -v`, and writes a JSON
ledger plus a markdown summary with per-model pass/fail counts, the first error
line of every failure, and the worst register/spill numbers. It exits non-zero
if anything failed.

**What it found, 2026-09-05** (CUDA 13.0.88; re-run the gate to reproduce —
`scripts/hopper_ptx_gate.sh --hw hopper --model all --strict`, see #899):
**870 of 871** kernels
across the five P0 targets emitted PTX and assembled for sm_90a on the first
pass. The one that did not was
`kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu`, which ptxas
rejected with

```
Instruction 'cvt with .e2m1x2' not supported on .target 'sm_90a'
Instruction 'mma with block scale' not supported on .target 'sm_90a'
Feature '.kind::mxf4nvf4' not supported on .target 'sm_90a'
```

— the NVFP4 block-scaled MMA path, Blackwell-only by construction, and the
same gap as the CUTLASS wrappers above reached through a hand-written kernel.

**That kernel is now 173/173.** Only its W4A4 *tail* uses those instructions
(FP4 weights AND FP4 activations: the two entry points
`moe_w4a16_fused_gate_up_t_k64_fp4` and `moe_w4a16_down_t_k64_fp4`).
Everything above them is W4A16 — 4-bit weights dequantised to BF16, plain
`mma.sync` — and assembles at the SM80 floor. The tail sits inside
`#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA`, and `kernels/hopper/HARDWARE.toml`
defines that macro in `[build] extra_nvcc_flags`, so it is compiled out here
and compiled in on GB10, whose PTX for the file is byte-identical across the
change (sha256 `137b44c2762d1996c9a1551a906a692cb067edae0b4ee4beee9098d303de4b3a`,
`nvcc --ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS`, before and after).
The two absent entry points are declared in
`kernels/hopper/qwen3.6-35b-a3b/MODEL.toml` `[expected_absent.moe_w4a16]` with
the ptxas error as the reason, so the boot audit reports them as an expected
absence rather than refusing to serve. Both are `try_kernel` lookups fired only
behind a default-off opt-in (`ATLAS_HOLO_MOE_GATEUP_FP4` /
`ATLAS_HOLO_MOE_DOWN_FP4`); what Hopper loses is the FP4 escape hatch, and the
FP8 path serves. 173/173 under `--strict`, i.e. with the
`--Werror all-warnings` the real build adds. `nemotron-super-120b-a12b`
passes `--strict` too.

**Current totals**, re-derived on 2026-09-11 (same CUDA 13.0.88, seven hopper
model targets, `--model all --strict`): **1282/1282**, 0 failed, 0 rejected
entry functions — `deepseek-v4-flash` 192, `qwen3.6-27b` and `qwen3.8-27b` 188
each, `qwen3.6-35b-a3b` 180, the other three 178. The 871 above is the
pre-guard five-target figure and is kept as the dated finding it was.

This does NOT make Hopper an NVFP4 target. It removes a compile-time
blocker; an Sm90 block-scaled path still does not exist, and nothing here has
run on H100/H200 silicon.

It runs a **self-test first, always**: one fixture that must compile for any
arch and one that must NOT compile for this one
(`scripts/fixtures/hopper_gate/`). If either verdict is wrong the gate refuses
to report results. A gate whose failure path has never executed is not
evidence.

The negative fixture is chosen **per arch**, because no single instruction is
absent from every architecture Atlas targets:

| arch under test | negative fixture | why it fails there |
|---|---|---|
| `sm_90a`, `sm_120a`, `sm_121*` | `known_bad_post_hopper.cu` | `redux.sync.max.abs.f32` exists only on sm_100a |
| `sm_100a` | `known_bad_post_blackwell_dc.cu` | warp-level `mma ... .kind::mxf4nvf4.block_scale` exists only on sm_120a/sm_121a |
| anything else | — | the gate REFUSES to run |

Each fixture carries its own measured table of which arches it passes and
fails on, and the ledger records which one was used. An arch with no
registered fixture is refused rather than waved through: a gate with no
failure path proves nothing.

Compilation is not correctness. A green gate says these kernels exist for
sm_90a; it says nothing about whether they produce the right numbers or run
well. Re-run the gate for a current ledger; it writes JSON and markdown
wherever `--out` points.

**`--hw gb10` is not yet usable as a control.** The gate takes any set under
`kernels/`, and pointing it at gb10 for the first time (2026-09-05, sm_121f,
`--strict`) gave 151/173: 22 `inferspark_prefill*` kernels are rejected at
their `_64` entry point for shared-memory size (`0x16000 bytes, 0xc000 max`),
all of them after passing `nvcc --ptx`. GB10 is the shipping target, so either
the gate's `ptxas` stage is stricter than the shipped pipeline — which emits
PTX and lets the driver JIT it, where a kernel may opt into >48 KB shared
memory at runtime — or those entry points are dead on GB10. That is open. It is
not caused by anything in this campaign: the same 22 stems fail against the
tree before it. Reproduce with
`scripts/hopper_ptx_gate.sh --hw gb10 --model qwen3.6-27b --strict`.

## The B200 (sm_100a) target

`kernels/b200/` is B200 and GB200 — both SM 10.0, datacenter Blackwell. It is
built exactly like `kernels/hopper/`: 225 relative symlinks into
`kernels/gb10/` (the 188-entry `common/` plus each of the five P0 models'
`nvfp4/`), with a real `MODEL.toml` per model whose header records that its
`[expected_absent]` tables were harvested on GB10 and **not** re-harvested on a
B200. `crates/atlas-kernels/tests/inherited_targets.rs` holds both trees to the
same assertions.

**sm_100a, and why it is not a step up from sm_121.** The `a` suffix opts into
datacenter Blackwell's arch-specific set — tcgen05, TMA, the native NVFP4
instructions. The two Blackwell architectures are **siblings, not a ladder**:

| instruction | sm_90a | sm_100a | sm_120a / sm_121 |
|---|---|---|---|
| `cvt.rn.satfinite.e2m1x2.f32` | ✗ | ✓ | ✓ |
| `mma.sync ... .kind::mxf4nvf4.block_scale` | ✗ | **✗** | ✓ |
| `redux.sync.max.abs.f32` | ✗ | ✓ | ✗ |
| `tcgen05.*` | ✗ | ✓ | ✗ |

(measured with nvcc/ptxas 13.0.88 on 2026-09-05; the first three are pinned by
the gate fixtures.) Warp-level block-scaled MMA is a consumer-Blackwell
instruction; on sm_100a the same work goes through `tcgen05.mma` against tensor
memory. So `sm_100a` PTX is not "sm_121 PTX that also runs on a B200", and
neither arch's PTX runs on the other.

**What the gate found, 2026-09-05** (CUDA 13.0.88;
`scripts/hopper_ptx_gate.sh --hw b200 --model all`, see #899):
**870 of 871** kernels across the five P0 targets emitted PTX and assembled for
sm_100a on the first pass — the same count as Hopper, and the same single
kernel failing, but for a **different reason**:

```
Instruction 'mma with block scale' not supported on .target 'sm_100a'
```

`kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu` fails on Hopper
because sm_90a has no `cvt .e2m1x2` at all; on sm_100a that conversion is fine
and the *warp-level block-scaled MMA* is what is missing.

**It is now 173/173 here too**, by the same mechanism as Hopper:
`kernels/b200/HARDWARE.toml` defines `-DATLAS_NO_WARP_BLOCKSCALE_MMA`, the
W4A4 tail of that file is compiled out, and its two entry points are declared
`[expected_absent.moe_w4a16]` in `kernels/b200/qwen3.6-35b-a3b/MODEL.toml`,
under `--strict`. Re-derived 2026-09-11 across all five b200 model targets:
**906/906**, 0 failed. One define covers both architectures because neither has the warp-level form —
Hopper for want of an NVFP4 datapath, B200 because it issues block-scaled MMA
through tcgen05 — so an arch comparison would get one of them wrong.

The remaining work is still different per architecture, and neither is done
here: Hopper would need an Sm90 MoE grouped GEMM, B200 the same math
re-expressed through tcgen05. What has changed is that the W4A16 path — which
is what these targets actually serve — is no longer blocked by a W4A4 kernel
they were never going to run. `nemotron-super-120b-a12b` passes under
`--strict` (`--Werror all-warnings`, as the real build) on both.

**Register pressure moves between the two arches, in both directions.** 574 of
871 kernels differ in max registers or spill bytes; 32 sit at the 255-register
ceiling on each, and 42 spill on each. Total spill across the set is 12,056
bytes at sm_90a against 19,400 at sm_100a, and the movement is concentrated:

| kernel | sm_90a regs/spill | sm_100a regs/spill |
|---|---:|---:|
| `gated_delta_rule_persistent` | 255 / 124 | 255 / **2256** |
| `gated_delta_rule_fla` | 255 / 324 | 254 / **0** |
| `gated_delta_rule_wy3_resident` | 255 / 204 | 255 / 16 |
| `gated_delta_rule` | 255 / 1552 | 255 / 1396 |
| `w4a16_gemm` | 168 / 624 | 168 / 492 |

`gated_delta_rule_persistent` is the one to look at first on real silicon: an
18x spill increase in the persistent GDN decode kernel is the shape of a
scheduling regression, and it is invisible to a pass/fail gate. `_fla` moving
the other way (324 bytes to none) is the same phenomenon with the opposite
sign. None of this has been timed — spill bytes are a hint, not a measurement.

**NVFP4 on B200 is a hand-kernel path only.** The CUTLASS wrappers in
`crates/spark-runtime/cuda/cutlass_nvfp4_gemm.cu` are gated on
`CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED` and
compile to nothing for sm_100a, exactly as they do for sm_90a. Porting them
needs `cutlass::arch::Sm100` collectives behind
`CUTLASS_ARCH_MMA_SM100_SUPPORTED`; that is not done here.

B300 and GB300 are **sm_103a** and are NOT this target. `sm_100a` PTX does not
run on CC 10.3, `atlas_core::arch::target_hint` returns `None` for it on
purpose, and `hardware_id_from_gpu_name` maps neither part — a B300 gets "no
shipped target" rather than a rebuild instruction that would fail the same way.

## The r9700 (gfx1201) target

`kernels/r9700/` is the **AMD Radeon AI PRO R9700** — 64 CU RDNA 4 on a
discrete PCIe board with 32 GB of dedicated GDDR6 (~640 GB/s). Like `strix`
it declares `vendor = "amd"`, so `build_target.rs` compiles it with SCALE
(scale-lang.com) through `targets/gfx1201/bin/nvcc`; unlike `strix` it is not
an APU, so none of that target's unified-memory sizing notes carry over.

**BUILDS, DOES NOT YET RUN.** The kernel set compiles on real gfx1201
silicon (an R9700 on ROCm 7.2.0, SCALE 1.7.1 `targets/gfx1201`), and nothing
has been SERVED on it, so every correctness claim past the build is open.

**Kernel set: a SUBTRACTIVE mirror of gb10, not a curated subset.** Until
2026-09-17 this target copied strix's shape: a hand-picked 99-entry `common/`
and four `qwen3.6-27b/nvfp4` shadows. A curated list is a list somebody has to
remember to update, and nobody did: `kernels/gb10/common/` gained
`dense_gemv_bf16_batch2.cu`, `qwen3_ssm/init.rs` began dispatching to it
(`init.rs:103`, a hard `gpu.kernel(...)?`), and on this target the qwen3.8-27b
model build died after loading all 21.8 GB of weights with

```
Kernel lookup dense_gemv_bf16_batch2::dense_gemv_bf16_batch2:
  Module load failed: Module 'dense_gemv_bf16_batch2' not loaded
```

So `kernels/r9700/common/` is now a whole-directory relative-symlink mirror of
`kernels/gb10/common/`, exactly as `kernels/hopper/` and `kernels/b200/` are,
and `kernels/r9700/qwen3.6-27b/nvfp4/` mirrors gb10's model directory the same
way (the `q4k_vendor` subdirectory included, as one directory symlink, which
is how hopper holds it). The rule is the DIRECTORY: a kernel added to gb10's
`common/` needs a link here, or this target silently compiles a smaller
inventory than GB10 does.

What it subtracts is the census, and only the census. Every `.cu` in both gb10
directories was compiled one at a time with the SCALE `nvcc` for gfx1201; 180
of 193 came back clean, and the mirror drops the twelve failing sources named
below plus the one header that only they include. The subtraction is
machine-checked from two sides: `kernels/r9700/HARDWARE.toml` `[kernels]
absent` lists the eleven `common/` names and `scripts/check_kernel_shadows.py`
RULE 3 fails if the set of gb10 entries this tree does not carry is anything
other than that list, and all 16 entry points of the thirteen dropped sources
are declared `[expected_absent]` in both MODEL.tomls so the boot audit reports
a stated absence instead of refusing to serve.

Counts after the change: `common/` holds 178 entries (169 `.cu`, 8 `.cuh`, and
`KERNEL.toml`, all relative symlinks into `kernels/gb10/common/`), and
`qwen3.6-27b/nvfp4/` holds 14 (12 `.cu`, the `q4k_vendor` directory, and
`KERNEL.toml`). The two entries that used to reach into `kernels/strix/common/`
(`dequant_fp8_blockscaled_bf16.cu`, `lora_bgmv.cu`) now link straight into
gb10: the files are byte-identical (`cmp` clean), and a link through a third
target's tree only hid which tree an edit would land in.

`HARDWARE.toml` and the four `MODEL.toml` files stay real files, because
sampling, behaviour and the per-hardware `[expected_absent]` harvest must stay
editable per hardware. Both `KERNEL.toml`s are now symlinks into gb10 instead:
the model one was a strix copy whose `[modules]` map had fallen behind gb10's
by roughly 40 renames, which the newly mirrored kernels need to register under
the module names the dispatch asks for. Its one SCALE-specific line, the
clang spelling `-ffp-contract=off` of the `--fmad=false` contraction pin, moved
to `kernels/r9700/HARDWARE.toml` `[build] extra_nvcc_flags`, which is where
`build_flags.rs` says an architecture-or-toolchain fact belongs on a target
whose KERNEL.tomls are all symlinks. `qwen3.8-27b` is gb10's `MODEL.toml` with
`kernel_source = "qwen3.6-27b"` kept, so it compiles this target's own 3.6
tree. There is no `BENCH.toml` and no `[benchmarks.limits]`: a target nobody
has measured cannot be campaigned, which is the correct state for it.

**Models served: four, and all four compile one kernel tree.**

| model | `[[model_types]]` | kernel tree | why |
|---|---|---|---|
| `qwen3.6-27b` | (`qwen3_5`, 5120), (`qwen3_6_moe`, 5120) | its own `nvfp4/`, the subtractive gb10 mirror | the only target here that owns a kernel directory |
| `qwen3.8-27b` | (`qwen3_5`, 5120) | `kernel_source = "qwen3.6-27b"` | bit-identical config to 3.6; carries `match_names` because the two collide on that exact pair |
| `ornith-1.0-9b` | (`qwen3_5`, 4096) | `kernel_source = "qwen3.6-27b"` | added 2026-09-17; see below |
| `holo-3.1-4b` | (`qwen3_5`, 2560) | `kernel_source = "qwen3.6-27b"` | added 2026-09-17; see below |

Qwen3.8-27B does not fit this board's 32 GB under Atlas's two-layout residency
(measured 2026-09-17), so coherent generation on gfx1201 has to be proved on a
smaller model of the same architecture family first.
[Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) and
[Holo-3.1-4B](https://huggingface.co/Hcompany/Holo-3.1-4B) are that family at
9B and 4B: Gated DeltaNet plus full attention plus a dense FFN plus a Qwen3-VL
ViT, which is `qwen3.6-27b`'s trunk.

Both **redirect** rather than mirror their own gb10 directory, and that choice
is load-bearing enough to state here. gb10's `ornith-1.0-9b/nvfp4/`,
`holo-3.1-4b/nvfp4/`, `holo-3.1-0.8b/nvfp4/` and `holo-3.1-35b-a3b/nvfp4/` are
**one byte-identical directory** (`diff -rq` is silent between all four), an
older six-file fork of gb10's `qwen3.6-27b/nvfp4/`. Two of those six,
`w4a16_gemm.cu` and `moe_w4a16_grouped_gemm.cu`, fail the gfx1201 census on
e4m3 because the fork carries only some of the `#if defined(__SCALE__)` shims
the `qwen3.6-27b` copies have.

The sources are shape-generic, so compiling the 3.6 tree for a 4096-wide and a
2560-wide model is sound rather than a shape mismatch: one unchanged source
set already serves `hidden_dim` 1024, 2048, 2560 and 4096, `q_heads` 8 and 16,
`kv_heads` 2 and 4, dense FFN and MoE, on gb10 today. The only `-D` naming a
model dimension anywhere under `kernels/` is `-DHDIM=128`, in eight gb10 model
`KERNEL.toml`s, none of them these; all four models involved declare
`head_dim = 256`. `[build] extra_nvcc_flags` is `["--fmad=false"]` on both
sides, character for character, and the two `KERNEL.toml`s otherwise differ
only in `[modules]` renames, almost all of them modules `qwen3.6-27b` has and
the fork does not ship.

The redirect drops one file, `fp4_mma_microtest.cu`, which the fork has and
`qwen3.6-27b` does not. It is not a serving kernel: its only reader is
`crates/spark-model/examples/fp4_mma_microproof.rs`, a proof of a hand-written
sm_120 block-scaled `mma.sync` against the CUTLASS collective, and that PTX has
no gfx1201 lowering either way.

Neither needs `match_names`. `validate_collision_match_names` demands needles
only from a `(model_type, hidden_size)` pair that two differently-named targets
both declare, and 4096 and 2560 collide with nothing on this hardware. Their
`[expected_absent]` tables are `qwen3.8-27b`'s, verbatim: same silicon, same
compiled tree through the same `kernel_source`, therefore the same thirteen
absences. gb10's copies declare a different set, harvested against the fork,
and those nine tables are dropped rather than merged because eight of them
name families that resolve in the tree this target actually compiles: the
`gated_delta_rule` f16 arms, the `w4a16` p3/k64/m128 twins,
`gated_delta_rule_wy17`, `q4k_mmq`, `q2_0_mmq`, `q4k_quantize`, `nvfp4_mmq`,
`gated_delta_rule_snap` and `gdn_verify_fused_conv_kn_f32`. The ninth,
`w4a4`, is genuinely absent here for the unrelated reason that
`w4a4_gemm.cu` does not compile for gfx1201, and the replacement set carries
it with that reason.

**UNVERIFIED, like everything else here.** Neither model has been loaded on
this board, and no loader work is claimed by this entry: what is registered is
the kernel target.

**Kernels absent on gfx1201.** Thirteen sources, 16 entry points, all declared:

| source | module | why |
|---|---|---|
| `inferspark_prefill_paged_bf16k_turbo{2,3,4}v.cu`, `_fp8k_turbo{2,3,4}v.cu`, `_turbo3k_turbo8v.cu`, `_turbo4k_turbo3v.cu`, `_turbo4k_turbo8v.cu` (9 files, 2 entry points each) | `prefill_paged_*` | `prefill_paged_compute_asym.cuh:99:28: error: local memory (70416 or 70432) exceeds limit (65536)`. RDNA 4 has RDNA 3.5's 64 KB per-workgroup LDS cap, and this header has no `#if defined(__SCALE__) #define BR64 32` pin (it hardcodes `BR64 64` at line 449). FOLLOW-UP: adding that pin and the matching 32-row host grid stride brings all nine back. |
| `prefill_paged_compute_asym.cuh` | (header) | Not compiled on its own, and nothing left in this tree includes it. Returns with the nine above. |
| `gated_delta_rule_fla.cu` (14 entry points) | `gated_delta_rule_fla` | `114:19: error: unknown opcode: fence.proxy.async.shared::cta`, an sm_90 async-proxy fence SCALE does not lower, issued unconditionally. No follow-up pending: this needs SCALE codegen or a fence-free rewrite, not a tile-size pin. GDN prefill stays on the `gated_delta_rule` / `_wy*` kernels. |
| `w4a16_fp8_ldmab.cu` | `w4a16_fp8_ldmab` | `107:9: error: this implementation does not provide a declaration of type 'fragment<nvcuda::wmma::accumulator, 16, 8, 32, float, void>'`, the e4m3 `mma.sync` path. `gemm_fp8_prefill.rs` looks `fp8_fp8_gemm_ldmab` up with `?`, so a native-FP8 prefill on this target is a hard error; NVFP4 targets requantize their FP8 projections at load and never reach it. |
| `w4a16_gemm_v2.cu` | `w4a16_v2` | `241:5: error` on the e4m3 MMA path, the same gap that makes `ATLAS_W4A16_VARIANT=v1` required here. |
| `w4a4_gemm.cu` | `w4a4` | `41:8: error: unknown opcode`. The whole `w4a4` module is therefore absent, so BOTH `try_kernel` lookups (`dense_ffn.rs:445` `w4a4_gemm`, `qwen3_attention/init.rs:749` `w4a4_gemm_mfast`) return `KernelHandle(0)` and every `w4a4_gemm_k.0 != 0 && ... && fp4_prefill` guard is false. Nothing falls back to `w4a4_gemm` here, because there is no `w4a4_gemm`: the FP4-activation prefill path is simply unavailable and prefill runs the ordinary W4A16/BF16 route. |

The census arithmetic closes: 193 files minus 180 clean is 13 failures, and
the thirteen dropped sources are exactly those 13 (the last one found was
`w4a16_fp8_ldmab.cu`, e4m3 fragment at `107:9`). Every file not named above
compiled, including all four files the old curated tree held as strix shadows,
`gated_delta_rule_snap.cu`, `gdn_verify_fused_conv_kn_f32.cu`,
`inferspark_prefill_paged_indirect.cu`, `nvfp4_mmq.cu`, `q2_0_mmq.cu`,
`q4k_mmq.cu`, `q4k_quantize.cu` and `vision_encoder.cu`.

**What the first compile settled.** Two inherited gfx1151 decisions are now
gfx1201 facts rather than carry-overs:

* **The `BR64 32` prefill pin is required here.** RDNA 4 exposes the same
  64 KB per-workgroup LDS cap as RDNA 3.5: a 73728-byte allocation is
  rejected outright ("local memory (73728) exceeds limit (65536)"). So
  `kernels/gb10/common/prefill_paged_compute.cuh` halving the block-row tile
  under `#if defined(__SCALE__)`, and `ops/prefill_attn_main_{a,b}.rs`
  selecting the matching 32-row host grid stride from `cfg!(atlas_scale)`,
  stay exactly as they are. The prefill throughput that costs is a hardware
  limit on this board, not an unexamined assumption.
* **SCALE has no e4m3 MMA codegen on gfx1201**, which is why
  `ATLAS_W4A16_VARIANT=v1` is required rather than merely inherited: there is
  no `fragment<nvcuda::wmma::accumulator, 16, 8, 32, float, void>`
  declaration, and inline `cvt.rn.satfinite.e4m3x2.f32` is rejected. What
  does compile is `__nv_cvt_float_to_fp8` from `cuda_fp8.h` and BF16
  `mma.sync m16n8k16`, which is the path `v1` takes. RDNA 4's native FP8 WMMA
  is a property of the silicon the toolchain does not reach.

**What the first bring-up must still probe.** Coherent generation above all;
nothing has been served on this board. Beyond that:

* **The four runtime knobs** `serve-amd.sh` exports for r9700:
  `ATLAS_W4A16_VARIANT=v1` is pinned by the missing e4m3 codegen above and is
  not a candidate to probe OFF until SCALE grows that path.
  `ATLAS_NO_GDN_FP8_PREFILL=1` is only a conservative first-serve default:
  strix never set it and serves coherently with the native FP8 SSM prefill on,
  so on r9700 it is the first knob to bisect OFF once a coherent baseline
  exists. `ATLAS_NO_FP8_PREDEQUANT=1` is a belt over a probe, not a pin; see
  "The FP8 prefill predequant" below. `ATLAS_LOAD_TRANSPOSED_TWINS=0` was the
  residency lever and is now simply the faster arm as well; see "The transposed
  second weight layout" below. `ATLAS_FORCE_GLOBAL_GDN`, which the script used to export, has
  no reader anywhere in the tree and was dropped rather than carried here.
* **`qwen3.6-27b/MODEL.toml` `[behavior] thinking_in_tools = false`** and the
  retuned sampling block, which are gfx1151 observations (a post-`</think>`
  content collapse on that silicon) carried over with the tree.

**The FP8 prefill predequant.** Measured on this board, 2026-09-17:
`Ornith-1.0-9B` loads, builds and passes the boot audit, and then every request
dies at layer 0 with `ssm prefill: out_proj GEMM failed: Kernel lookup
w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab: Module load failed: Module
'w4a16_fp8_ldmab' not loaded`. `predequant_for_prefill` built an NVFP4-to-FP8
copy of the SSM `out_proj` (and, on the loaders that call the other two, the
attention q/k/v/o and the MoE gate + shared expert) without asking whether the
GEMM that reads it exists, and the prefill dispatch PREFERS those copies over
both NVFP4 arms, so one absent module turned a working fallback chain into a
hard error. `w4a16_fp8_ldmab.cu` is the thirteenth gfx1201 census failure and is
not in this target's tree at all.

`crates/spark-model/src/layers/fp8_predequant.rs` is now the load-time guard. It
probes every arm the dispatch can take rather than only the preferred one.
`ops::fp8_gemm_n128` chooses between the `ldmatrix.x4` kernel and
`w4a16::fp8_gemm_t` on `K % 32`, which is a property of the projection, and the
caller chooses between it and `fp8_gemm_n128_m128` on the token count, which is
a property of the request, so a partial answer is a serve that works on short
prompts and dies on long ones. With the copies absent the SSM falls to
`w4a16_gemm_n128` and then `w4a16_gemm`, attention's `use_fp8_act` goes false,
and the MoE's three copies take their NVFP4 branch. `ATLAS_NO_FP8_PREDEQUANT=1`
forces the same outcome and is exported for r9700 so the serve log names the
decision in words; `=0` forces the copies back where an operator's environment
sets the variable globally, and cannot override a genuinely absent kernel.

**The transposed second weight layout.** Atlas keeps every NVFP4 projection in
TWO layouts: the packed `[N, K/2]` original decode reads, and a transposed
`[K, N/2]` twin the fast prefill GEMMs (`w4a16_gemm_t_m128` and its v2/k64
siblings) consume. On this board that is the difference between loading a 27B
and not. `docs/porting/r9700-residency.md` has the measured ledger; the summary:

| | GiB | fits the ~27.9 GB weight budget? |
|---|---|---|
| as of 2026-09-17 | 47.07 | no |
| release-on-consume (`ATLAS_LOAD_RELEASE_SOURCES`) | 38.31 | no |
| ... and the attention BF16 dequant leak fixed | 35.18 | no |
| ... and no transposed twins | 21.25 | **yes** |

`ATLAS_LOAD_TRANSPOSED_TWINS` is `1` (build them; the pre-lever behaviour byte
for byte, and the default on every non-SCALE target), `0` (build none) or `auto`
(build them only if free VRAM after the checkpoint is resident exceeds their
projected bytes plus a 4 GiB reserve for the KV cache, the buffer arena and the
vision encoder's working set). **Unset takes `cfg!(atlas_scale)`, and as of
2026-09-17 that means `0` on SCALE**, not `auto` — see "What skipping costs"
below: on gfx1201 the twin arm measured SLOWER than the arm it replaces, so
there is no residency-versus-speed trade for the probe to weigh. NVIDIA is
untouched and still builds them. The decision is made ONCE, before any layer
allocates, for the reason `gemma4/loader_a.rs::ffn_transpose_fits` gives:
`free_memory()` shrinks as layers load, so a per-layer probe transposes the
early layers and skips the late ones and leaves prefill straddling two dispatch
arms.

The projected twins, from the model's own dimensions (the same arithmetic that
reproduces the measured ledger, pinned by `transposed_twins_tests.rs`):

| model | dense FFN | SSM | attention | total |
|---|---|---|---|---|
| `qwen3.8-27b` / `qwen3.6-27b` | 8.96 GiB | 2.90 GiB | 0.88 GiB | **12.74 GiB** |
| `ornith-1.0-9b` | 2.53 GiB | 0.84 GiB | 0.25 GiB | **3.62 GiB** |

**What skipping costs, measured on this board rather than inherited.** Nothing;
it pays. The R9700 prefill measurement of 2026-09-17 (Ornith-1.0-9B with the
twins against Qwen3.8-27B without) puts `w4a16_gemm_t_m128` at **~1 TFLOP/s and
the plain `w4a16_gemm` at ~4 TFLOP/s** on gfx1201. The twin arm is the slower
one here.

⚠️ The **~7.0 vs ~51 TFLOP/s** figures this section used to quote are the
Gemma-4-31B numbers from **GB10** and were never measured on SCALE. They are why
every non-SCALE target still builds the twins by default, and they must not be
quoted for gfx1201. The SSM and attention sides remain unmeasured on both.
Decode is untouched on every target: it reads the packed original either way.
Dropping the SSM twin also drops the 1.41 GiB `out_proj` FP8 predequant, which
exists to feed the same transposed GEMM. The NVFP4-MMQ finalize is skipped with
them, because it is residency-neutral only while there are `_t` copies for it to
free.

**Reading the memory budget off the serve log.** Three INFO lines now carry the
whole arithmetic, so a serve that will not fit batch 4 can be diagnosed without
re-running it under `RUST_LOG=debug`:

```
Preflight reserve: inference=2098 MB = fixed 886 MB + ring 1212 MB, buffer_arena=1945 MB ...;
  ring: 8 slots x 1 seqs x 151.5 MB/seq = 1.18 GB
Preflight reserve breakdown: ssm_pool=... ssm_snapshot=... gdn_two_phase=... cuda_headroom=...
KV cache: 31.9 GB total x 90% util = 28.7 GB budget; 23.0 GB pre-KV + 2.0 GB reserve -> 3.7 GB for KV ...
KV budget itemised: pre-KV 23.00 GB = weights 19.42 (store, resident now) + buffer arena 1.90
  + other 1.68 (CUDA context, driver, co-tenants); already released before this point:
  vision 1.65, lm_head source 1.18; reserve 2.05 GB ... + DFlash 0.00 GB -> KV 3.70 GB
```

The two terms that used to be opaque are the ones that matter here. **"pre-KV"**
is `total - free`, a subtraction that lumps the weights, the buffer arena, the
CUDA context and any desktop co-tenant into one number — the itemised line names
each and prints the remainder as `other` rather than hiding it. **The reserve**
arrives at the KV sizer as a single `usize` from a different crate; the preflight
line now prints it as `fixed + ring`, and the ring is `slots x seqs x bytes`,
which at batch 1 on this board is 8 x 1 x 151.5 MiB = 1.18 GB of a 2.05 GB
reserve. That is the term to attack for more batching headroom, not the KV
arithmetic.

**The checkpoint's `lm_head`, once the heads are built.** `unsloth/Qwen3.8-27B-NVFP4`
ships `lm_head.weight` as FP8 E4M3 with a per-channel BF16 scale. `load_lm_head`
dequantises it into a fresh BF16 allocation and every head — NVFP4, FP8 or the
BF16 skip — is built from that copy, so the checkpoint's own bytes have no
reader. **1.18 GiB**, released since 2026-09-17 by
`lm_head_setup::release_lm_head_source`, on the same `ATLAS_LOAD_RELEASE_SOURCES`
knob as the loader's other release sites (ON under `cfg!(atlas_scale)`).

Two flags keep it: `--lm-head-dtype fp8` and `--dflash` both reach
`native_fp8_lm_head_share`, which binds `lm_head.weight` ZERO-COPY on purpose,
and with either set nothing is released. A third guard is less obvious and is
the one that matters: the release only fires when the checkpoint's `lm_head` is
FP8, because that is the proof a copy was made. On a BF16 or NVFP4-prepacked
head the loader hands the store's pointer straight through. The release runs
immediately after `setup_lm_heads` and before the KV sizer, so the 1.18 GiB
reaches the KV budget rather than being freed after it was already spent.

**`--text-only`: the vision tower, for a serve that only ever sends text.**
`unsloth/Qwen3.8-27B-NVFP4` ships a BF16 vision tower and Atlas binds it,
because the checkpoint declares a `vision_config`. That is **~1.65 GiB** of the
store, resident for the life of the process, charged against the same budget as
the weights, the buffer arena and the KV cache — and on this board a sequence's
KV at 4096 tokens is only ~0.3 GB, so the tower is worth several batch slots.
`--text-only` clears `config.vision` before the weight store is built, so the
tower's bytes are never read from disk, never bound and never resident (the load
log prints the GB it did not read). Image and video inputs are then refused with
a **400** naming the reason, rather than silently dropped. Default OFF: a VL
checkpoint serves with vision unless this says otherwise.

The same flag closes a spelling gap worth knowing about. The load-time skip and
`build_model`'s unbound-tower reclaim both match `model.visual.*`,
`model.vision*` and `visual.*`, but `Qwen35WeightLoader::load_vision_encoder`
also probes **`model.language_model.visual.*`**, which every
`AutoModelForImageTextToText` re-quant uses. A checkpoint in that layout had a
tower that was read from disk, never bound and never freed. Both sites now call
one predicate, `fast_weights::is_vision_tensor`, and it carries the fourth
spelling.

**A proper fix is a kernel, not this knob**: a prefill GEMM that reads the
packed `[N, K/2]` layout directly with a transposed tile walk, the way
`w4a16_gemm` already does at scalar speed but with the 128x128 `cp.async`
tiling the `_t` kernels have. Then there is no second layout to build or skip,
on any target, and the 27B fits without paying 7x for prefill. Until that
exists, this lever buys a serve and nothing else.

**Free-memory source on AMD boards.** Atlas does NOT trust SCALE's
`cuMemGetInfo` for free VRAM on this target. Measured 2026-09-17 on the R9700
(SCALE 1.7.1, ROCm 7.2.0): the first shard of `unsloth/Qwen3.8-27B-NVFP4`
logged `GPU memory: 31.56 GB used, 0.05 GB free` while
`/sys/class/drm/card1/device/mem_info_vram_used` peaked at 22.9 GB of a
31.86 GB board and the allocation ledger held 22.57 GB of tensors. A two-loop
repro in one program pins it to the allocation COUNT, not the byte count: 56
allocations of 512 MiB track sysfs within 1 percent (free 3704 MiB at
28672 MiB allocated, sysfs used 29379 MiB), while 2000 allocations of 11 MiB
report free 60 MiB at 16500 MiB allocated against sysfs used 17227 MiB of
32624 MiB, stay pinned at 64 MiB free through 22000 MiB allocated, and never
recover after every allocation is freed (22064 MiB free). Native HIP
`hipMemGetInfo` in the same loop is honest. It is a **SCALE runtime reporting
defect**, roughly 16 MiB of phantom usage charged per allocation to its own
accounting (a pool granularity rather than real VRAM), and the clean repro
exists for Spectral. Real VRAM use is fine; only the reported number is wrong.

So on a SCALE build (`cfg!(atlas_scale)`, vendor-driven) the free leg comes
from `mem_info_vram_total` minus `mem_info_vram_used` in
`/sys/class/drm/card*/device`, auto-detected by matching the node's total
against the driver's within 5 percent. That is the kernel's own TTM
accounting, so it also counts the desktop compositor's 0.4 to 1.6 GB, which a
per-context driver query never sees. `total` stays on the driver, which read
32624 MiB, the board's true capacity. `ATLAS_MEMINFO_SOURCE` overrides:
`driver` forces the old behaviour for an A/B, `sysfs` forces detection on a
non-SCALE build, and `sysfs:/sys/class/drm/cardN/device` names the node when
auto-detection cannot (two boards of the same size, an unusual DRM layout).
Nothing about this reaches an NVIDIA build, which resolves to the driver
without even scanning `/sys`. See
`crates/spark-runtime/src/cuda_backend/meminfo_source.rs`.

**Optional modules on SCALE targets.** Atlas carries kernel modules that only
some targets compile: `kernels/gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu` is
entirely inside `#if defined(BLACKWELL_MMA_AVAILABLE)`, so on anything that is
not Blackwell it compiles, successfully, to a code object with no kernels in
it. On NVIDIA that costs nothing: `cuModuleGetFunction` answers "not found",
`spark-model`'s `try_kernel` folds the error into `KernelHandle(0)`, and every
use site is already guarded. Observed on SCALE 1.7.1 / gfx1201 the driver does
not answer that way. It returns SUCCESS for a name the object does not define,
hands back a handle backed by no code, and the first launch through it dies
with `CUDA_ERROR_INVALID_IMAGE (200)`, which is how `spark serve` on the
R9700 died at its very first kernel, `nvfp4_mmq::atlas_nvfp4_repack`, with
21.8 GB of weights already resident. So the registry no longer takes the
driver's word for it: at load time it reads each binary module's ELF symbol
table (`crates/atlas-core/src/elf_symbols.rs`, a dependency-free ELF64 walk
that counts `STT_FUNC` symbols and the AMDGPU `<kernel>.kd` descriptors) and
refuses a lookup the object provably cannot satisfy, with
`<module>::<kernel>: not defined in this target's code object (optional module
compiled out?)`. That is an ordinary error, so the optional kernel degrades to
handle 0 exactly as it does on NVIDIA. A module whose bytes do not parse as
ELF64 is not guarded at all (the driver still decides), and the PTX path is
untouched. `atlas-kernels`' build script reads the same objects with the same
code and prints `cargo:warning=atlas-kernels: <module> compiled to a code
object with no kernels on <arch> (optional module compiled out)`, so the empty
modules are named in the build log before anyone serves the target.

**Build and serve.** `build-amd.sh` and `serve-amd.sh` take the hardware
target from `ATLAS_TARGET_HW` (default `strix`) and read the SCALE arch from
`kernels/$ATLAS_TARGET_HW/HARDWARE.toml`, so this target needs no separate
script:

```bash
export SCALE_HOME=$HOME/scale171/scale-1.7.1-Linux
ATLAS_TARGET_HW=r9700 ./build-amd.sh
ATLAS_TARGET_HW=r9700 ./serve-amd.sh unsloth/Qwen3.8-27B-NVFP4
```

`ATLAS_TARGET_MODEL` defaults to `qwen3.8-27b` for `r9700` (and stays
`qwen3.6-27b` for `strix`); `ATLAS_TARGET_QUANT` defaults to `nvfp4` for both.
The two small models are named the same way, and the 27B not fitting the
board's 32 GB is the reason they exist:

```bash
ATLAS_TARGET_HW=r9700 ATLAS_TARGET_MODEL=ornith-1.0-9b \
  ATLAS_TARGET_QUANT=nvfp4 ./build-amd.sh
# or ATLAS_TARGET_MODEL=holo-3.1-4b
```

`GPU_UTIL` defaults to 0.75 here against strix's 0.70, because this is a
discrete board that may also be driving a desktop session. What the scripts do
by hand:

```bash
export CUDA_PATH="$SCALE_HOME/targets/gfx1201"
export CUDA_HOME="$CUDA_PATH"
export PATH="$SCALE_HOME/targets/gfx1201/bin:/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$SCALE_HOME/targets/gfx1201/lib:$LD_LIBRARY_PATH"
export ATLAS_TARGET_HW=r9700
export ATLAS_TARGET_MODEL=qwen3.8-27b   # or qwen3.6-27b
export ATLAS_TARGET_QUANT=nvfp4
export CUDARC_CUDA_VERSION=12080
cargo build --release -p spark-server --no-default-features --features cuda

# serve: SCALE libs FIRST so /opt/rocm cannot shadow the bundled ROCm, then
# the required knob and the conservative first-serve default (see above).
export LD_LIBRARY_PATH="$SCALE_HOME/targets/gfx1201/lib:$SCALE_HOME/lib"
export ATLAS_W4A16_VARIANT=v1
export ATLAS_NO_GDN_FP8_PREFILL=1     # bisect candidate, not a pin
export ATLAS_NO_FP8_PREDEQUANT=1      # belt; the loader probes for this anyway
export ATLAS_LOAD_TRANSPOSED_TWINS=0  # fits 32 GB; the plain arm is also faster here
# --text-only drops the checkpoint's ~1.65 GiB vision tower before load. Add it
# for a text deployment; leave it off if images are ever sent (400 otherwise).
target/release/spark serve unsloth/Qwen3.8-27B-NVFP4
```

**`atlas_scale` is vendor-driven.** `spark-model/build.rs` and
`spark-runtime/build.rs` set `atlas_scale` (and `atlas_hip`) from
`kernels/<hw>/HARDWARE.toml` `[hardware].vendor`, not from the target's name.
They used to test `hw.starts_with("strix")`, which was correct only for as
long as every SCALE target was called strix-something: the kernel side pins
`BR64 32` under `__SCALE__` for EVERY SCALE target, so a second one under any
other name would have compiled 32-row kernels and launched them with a 64-row
stride — silently, with no build error and no failing test, just dropped query
rows. Vendor is the same signal `atlas-kernels/build.rs` already picks the
compiler with.

## Adding a new quantization scheme

Atlas supports NVFP4 (E2M1 + FP8 scales), FP8 block-scaled, BF16 raw.
To add a new scheme (e.g., MX4, INT4):

1. **`crates/atlas-core/src/config.rs`**: extend the quant detection
   logic to recognize the new format from `quantization_config` in
   `config.json`.
2. **`crates/spark-model/src/weight_map/`**: add a loader function
   that produces the right `QuantizedWeight` variant.
3. **Per-model kernels**: write `*.cu` for the new quant under
   `kernels/gb10/<model>/<new-quant>/`. The build script auto-picks them.
4. **Dispatch**: `crates/spark-model/src/layers/<layer>/` per-quant
   branches in the forward path.

## PTX arch suffix and the static shared-memory ceiling

The arch string in `HARDWARE.toml` decides more than which GPU the PTX runs
on. `ptxas` enforces a per-target ceiling on **static** `__shared__`
allocations, and on Blackwell that ceiling depends on the suffix, not the
chip. Measured with CUDA 13.0.88 (`nvcc --ptx` then `ptxas`) by bisecting a
single `__shared__ char buf[N]` at 48 KiB, 100 KiB and 228 KiB:

| arch string | static `__shared__` ceiling |
|---|---|
| `sm_120a`, `sm_121a` (arch-specific) | `0x18c00` = 99 KiB |
| `sm_120`, `sm_121`, `sm_121f` (plain or family) | `0xc000` = 48 KiB |
| `sm_90a` | `0x38c00` = 227 KiB |

`nvcc --ptx` accepts an oversized static array on every target; only `ptxas`
(or the driver JIT at `cuModuleLoadData`) rejects it, with `Entry function
'...' uses too much shared data`. A green `cargo build` therefore does not
prove a kernel assembles: the build runs `nvcc --ptx` only. The dynamic
opt-in (`MAX_DYNAMIC_SHARED_SIZE_BYTES`) does not rescue a fixed-size array;
only `extern __shared__` allocations use it.

Consequence for gb10: its `sm_121f` target keeps one build portable across
future 12.x parts, and pays for it with the 48 KiB ceiling. The BR=64
prefill variants (`inferspark_prefill*`, `prefill_paged_compute*.cuh`) declare
70 to 90 KiB of static shared memory and are rejected for `sm_121f` while the
same source assembles for `sm_121a` and `sm_120a`. Moving gb10 to `sm_121a`
is a portability trade, not a bug fix, so it is a deliberate decision rather
than something to change while adding a target.

## Testing a new target

### PTX emission is not assembly validation

`crates/atlas-kernels/build.rs` runs `nvcc --ptx` only (`NvidiaTarget::compile`
in `build_target.rs`); nothing assembles the PTX until the runtime hands it to
`cuModuleLoadData`. A green `cargo build` therefore proves that every kernel
*emits* PTX, not that every entry *assembles* for the target. Static
`__shared__` allocations above the target's static limit are the known case:
`nvcc --ptx` accepts them and `ptxas` rejects the entry (`uses too much shared
data`). The limit is per target: `ptxas` reports 48 KiB (`0xc000`) for
`sm_121f` and 227 KiB (`0x38c00`) for `sm_90a`, so a kernel can assemble for
Hopper and fail for gb10. The
`MAX_DYNAMIC_SHARED_SIZE_BYTES` opt-in does not rescue a fixed-size array.

`scripts/hopper_ptx_gate.sh` closes that gap for a hardware set by running
`ptxas` on every emitted module. On 2026-09-05 it found 22 of 173 gb10
`qwen3.6-35b-a3b` modules (42 entry functions, the BR=64 prefill variants and
their BR=32 siblings) rejected for `sm_121f` on CUDA 13.0.88. That is a
pre-existing gb10 finding, independent of the Hopper and B200 targets; the
inventory, receipts and runtime trace live with the campaign notes in
Avarok-Cybersecurity/atlas#899.

### Device validation

Once compiled:

```bash
# Smoke test
docker run --gpus all --ipc=host -p 8888:8888 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-gb10:latest \
  serve <new-model-hf-id> --max-seq-len 4096 --max-batch-size 1

curl http://localhost:8888/v1/chat/completions -d '{"model":"...","messages":[{"role":"user","content":"hi"}]}'

# Coherence + tool calls + long context
python3 tests/single_gpu_suite.py --url http://localhost:8888 --model <new-model-hf-id>

# Regression sweep
python3 tests/run_all_models.py
```

The sweep harness saves per-model JSONs to `tests/all_models_results/`
that you can diff against the pre-merge baseline (`tests/all_models_results.pre-refactor/`).

## Reference implementations

When in doubt, copy from a model with similar arch:

| New model is... | Look at |
|---|---|
| Hybrid SSM + attention MoE | `qwen3.5-35b-a3b/`, `qwen3.6-35b-a3b/` |
| Hybrid SSM + attention dense | `qwen3.5-27b/`, `qwen3.6-27b/` |
| Pure Mamba2 + MoE | `nemotron-3-nano-30b-a3b/` |
| Pure attention + MoE | `mistral-small-4/`, `minimax-m2-229b/` |
| Pure attention dense | `gemma-4-31b/` |
| Vision-language | `qwen3-vl-30b-a3b/` |
