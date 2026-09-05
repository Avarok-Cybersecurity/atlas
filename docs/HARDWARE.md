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
(AMD gfx1151) and `metal` are the non-NVIDIA sets. Adding another — say sm_120
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

**Kernel set: inherited from gb10 by symlink.** `kernels/hopper/common/` is 181
relative symlinks into `kernels/gb10/common/` (all 171 `.cu`, the 9 `.cuh`
headers, and `KERNEL.toml`), and each of the five model targets mirrors gb10's
`nvfp4/` directory file by file the same way. Git stores them as symlinks
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

**What it found, 2026-09-05** (CUDA 13.0.88, receipts in
`docs/campaigns/hopper-atlas-vs-vllm-2026-09/receipts/`): **870 of 871** kernels
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
FP8 path serves. Receipt:
`receipts/ptx_gate_hopper_qwen36_w4a4guard_2026-09-05.*` — 173/173 under
`--strict`, i.e. with the `--Werror all-warnings` the real build adds.
`nemotron-super-120b-a12b` passes `--strict` too.

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
well. Receipts live in `docs/campaigns/`.

**`--hw gb10` is not yet usable as a control.** The gate takes any set under
`kernels/`, and pointing it at gb10 for the first time (2026-09-05, sm_121f,
`--strict`) gave 151/173: 22 `inferspark_prefill*` kernels are rejected at
their `_64` entry point for shared-memory size (`0x16000 bytes, 0xc000 max`),
all of them after passing `nvcc --ptx`. GB10 is the shipping target, so either
the gate's `ptxas` stage is stricter than the shipped pipeline — which emits
PTX and lets the driver JIT it, where a kernel may opt into >48 KB shared
memory at runtime — or those entry points are dead on GB10. That is open. It is
not caused by anything in this campaign: the same 22 stems fail against the
tree before it. Receipts:
`receipts/ptx_gate_gb10_qwen36_w4a4guard_2026-09-05.*` and
`receipts/ptx_gate_gb10_qwen36_preguard_control_2026-09-05.*`.

## The B200 (sm_100a) target

`kernels/b200/` is B200 and GB200 — both SM 10.0, datacenter Blackwell. It is
built exactly like `kernels/hopper/`: 218 relative symlinks into
`kernels/gb10/` (the 181-entry `common/` plus each of the five P0 models'
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

**What the gate found, 2026-09-05** (CUDA 13.0.88, receipts in
`docs/campaigns/hopper-atlas-vs-vllm-2026-09/receipts/ptx_gate_b200_2026-09-05.*`):
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
`[expected_absent.moe_w4a16]` in `kernels/b200/qwen3.6-35b-a3b/MODEL.toml`.
Receipt: `receipts/ptx_gate_b200_qwen36_w4a4guard_2026-09-05.*`, `--strict`.
One define covers both architectures because neither has the warp-level form —
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

## Testing a new target

### PTX emission is not assembly validation

`crates/atlas-kernels/build.rs` runs `nvcc --ptx` only (`NvidiaTarget::compile`
in `build_target.rs`); nothing assembles the PTX until the runtime hands it to
`cuModuleLoadData`. A green `cargo build` therefore proves that every kernel
*emits* PTX, not that every entry *assembles* for the target. Static
`__shared__` allocations above 48 KiB are the known case: `nvcc --ptx` accepts
them and `ptxas` rejects the entry (`uses too much shared data`). The
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
