# CUDA Kernel Engineering

This is the chapter you read when you're about to write a kernel. It covers the conventions every Atlas kernel follows, the tools that matter on GB10 SM121, and the workflow that takes an idea from a profile to a merged PR.

## The kernel inventory

A default `(GB10, <model>, <quant>)` leaf ships ~30–40 kernels. Canonical roles:

| Role | File example | What it does |
|---|---|---|
| Prefill attention | `inferspark_prefill_v47.cu` | Flash Attention v2, `cp.async` pipelining, `mma.sync.aligned.m16n8k16` tensor cores, 2 CTAs/SM |
| Decode attention | `paged_decode_attn_turbo3_128.cu` | Online softmax, split-K, adaptive split count |
| Prefill attention (FP8 KV) | `inferspark_prefill_fp8kv.cu` | Same but with FP8 KV read path |
| KV append | `kv_cache_append.cu` | Per-token K/V write into paged cache |
| MoE prefill | `moe_prefill.cu` | Fused dequant + grouped GEMM, 256 experts, topk=10 |
| MoE decode — shared expert | `moe_shared_expert_fused_fp8.cu` | Shared-expert path with fused FP8 GEMM |
| MoE decode — expert | `moe_expert_relu2_down_shared.cu` | Token-level MoE for decode |
| Dense GEMM | `dense_gemm_bf16.cu`, `w8a16_gemv.cu` | Non-MoE FFN GEMMs |
| SSM — preprocess | `ssm_preprocess.cu` | Fused QKVZ deinterleaving + GDN gate (softplus + sigmoid) |
| SSM — Gated Delta Rule | `gdr.cu` | Mamba/delta-net SSM (prefill + decode) |
| SSM — causal conv1d | `causal_conv1d.cu` | Mamba's 1D convolution |
| Primitive — RMSNorm | `rms_norm.cu` | Single-block tree reduction |
| Primitive — SiLU×Mul | `silu_mul.cu`, `silu_mul_quant.cu` | SwiGLU with optional fused NVFP4 quant |
| Primitive — RoPE | `rope.cu` | GQA-aware rotary embedding |
| Primitive — argmax BF16 | `argmax_bf16.cu` | Single-block tree reduction, 4-byte result |
| E2M1 conversion | `e2m1_branchless.cu` | Software FP32 → E2M1 conversion for SM121 |
| MoE gating | `topk.cu`, `softmax.cu` | Expert selection pre-dispatch |
| WHT | `wht_bf16.cu` | Walsh-Hadamard for TurboQuant KV |
| Element-wise | `bf16_add.cu`, `transpose.cu` | Small utilities |

Kernels that differ between models (e.g., Nemotron's Mamba-2 vs Qwen3.5's GDN) live in different `(model, quant)` leaves but share file names when the shapes match.

## SM121 hardware budget (quick reference)

Grace-Blackwell GB10 / SM121 numbers you will care about when writing kernels:

| Quantity | Value |
|---|---:|
| Global memory | 119.7 GB LPDDR5X (unified) |
| Peak memory BW | 273 GB/s |
| SMs | 32 (typical, configurable) |
| Warps per SM | 48 max concurrent |
| Registers per SM | 65,536 |
| Shared memory per SM | 100 KB effective |
| L2 cache | ~ 32 MB (large enough that benchmark reports up to 599 GB/s achieved BW at small sizes — the "L2 cache effect" in the kernel tables) |
| Tensor core throughput (BF16) | high, SM121-specific |
| `cp.async` | supported |
| Native FP4 MMA | **not available on SM121** — see below |

The **native FP4 MMA** caveat is load-bearing: SM120/SM121 does not expose the `cvt.rn.satfinite.e2m1x2.f32` instruction or native FP4 tensor-core paths. Every NVFP4 kernel on GB10 uses software E2M1 conversion (the "branchless" kernel) and dequantises-to-BF16 for the MMA. This is *not* a performance bug — it is the silicon. The community benchmarks that cite "native FP4 throughput" on newer Blackwell parts do not transfer. The [NVFP4 deep dive](./nvfp4.md) walks the workaround in detail.

## Conventions every Atlas kernel follows

- **SPDX header line 1.** `// SPDX-License-Identifier: AGPL-3.0-only`. Enforced by the `license-headers` job in CI.
- **`extern "C" __global__`** entry points with a stable name. The name is what `GpuBackend::kernel(module, func)` looks up.
- **All pointer args are typed at the right level** — `const __nv_bfloat16*`, `const int8_t*`, not `const void*`. The BF16 + E2M1 types come from `<cuda_bf16.h>` and `<cuda_fp8.h>`; module-local aliases are fine but don't hide the precision.
- **Grid/block dimensions are passed from the Rust side.** Never compute block dims from runtime GPU properties inside the kernel — let `KernelLaunch::new().grid(...).block(...)` own it.
- **One kernel per file where possible.** Fused variants belong in their own files (e.g. `silu_mul_quant.cu` vs `silu_mul.cu`) so the `KERNEL.toml` [modules] override maps cleanly.
- **No `<iostream>`, no `printf` in hot paths.** Use `#if 0` stubs during development, strip before merging. `nvcc` warns on printf inside `__device__` code when it bloats the PTX.
- **Shared-memory layouts are always annotated.** A comment near each `__shared__` alias documents the row/column order and any padding added to avoid bank conflicts.

## The profiling workflow

1. **Start with `nsys profile`** against the live server running a benchmark. The first question is always *which kernel is the bottleneck* — do not tune in the abstract.
   ```bash
   nsys profile --trace=cuda,cudnn,cublas,osrt -o atlas.qdrep \
     /path/to/spark serve <model>
   # in another terminal: drive bench load
   nsys stats --report cuda_gpu_kern_sum atlas.qdrep | head -20
   ```
2. **For the top 2–3 kernels, drill into `ncu`** (Nsight Compute). The metrics that matter on GB10:
   - `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` — SM utilisation.
   - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` — shared-memory bank conflicts.
   - `dram__bytes.sum` vs `dram__bytes_read.sum.peak_sustained` — how close to the 273 GB/s ceiling.
   - `sm__warps_active.avg.pct_of_peak_sustained_elapsed` — warp occupancy.
3. **Know which side of the roofline you're on.** GB10's compute-vs-BW roofline is steeper than a desktop GPU's (273 GB/s vs several thousand tensor-core TFLOPs). Most decode kernels are memory-bound; most prefill kernels are tensor-core-bound once `cp.async` is hiding the K/V load.

## The three performance levers (in order of payoff)

1. **Tiling + `cp.async`.** The biggest single win on SM121 is pipelining the next tile's global-memory load behind the current tile's compute. `__pipeline_commit()`/`__pipeline_wait_prior(N)` with two or three in-flight stages typically doubles attention throughput vs a naive loop. Prefill v47 uses 2 stages and two CTAs per SM.
2. **Shared-memory layout.** Bank conflicts destroy kernels silently. The usual fix is an xor-swizzle or an extra padding column. The `ncu` metric above tells you how far you are from zero conflicts.
3. **Register budget.** Hit the 255-register-per-thread cliff and you spill to local memory, which on GB10 means LPDDR. `__launch_bounds__(256, 2)` (max 256 threads, 2 blocks per SM) is the decoration that constrains nvcc's register allocator. Use it. Check with `--ptxas-options=-v`.

Things that matter less on GB10 than on an H100:
- Shared-memory capacity (100 KB is generous for these shapes).
- Warp specialisation. SM121's scheduler is good enough that explicit producer/consumer warp roles rarely pay back on kernels of the shapes Atlas runs.
- Distributed shared memory. No NVLS / multi-CTA clusters to lean on — `NVLS_ENABLE=0` is forced in the NCCL env.

## CUDA graphs

Every supported batch size gets a captured graph at startup. The engine replays graphs with `cuGraphLaunch`, eliminating per-launch `cuLaunchKernel` overhead (~microseconds each, compounded over ~300 kernels per forward pass). Graph stability requires:

- **Buffer addresses do not change.** `BufferArena` pre-allocates at startup; the engine reuses the same `DevicePtr`s for every step. No `alloc`/`free` in the hot loop.
- **Kernel launch parameters are data-dependent in a bounded way.** Per-layer kernel launches pass a handful of scalar args that vary per step (current token count, current block index). Those are captured as `CUgraphNodeParams`.

Turning graphs off (`--profile`) disables capture — useful when you are profiling under `nsys` because graphs collapse every kernel into a single graph-launch event and defeat per-kernel timing.

## Writing a new kernel — the minimum

1. **Find the `(hw, model, quant)` leaf.** Typically `kernels/gb10/<model>/<quant>/`.
2. **Drop `your_kernel.cu`** with the SPDX header and a `extern "C" __global__` entry.
3. **Decide the module name.** Default is the file stem (`your_kernel`). Override in `KERNEL.toml` if you want a short name.
4. **Call it from the layer.** In `spark-model/src/layers/<your_layer>.rs`, `gpu.kernel("your_kernel_module", "your_kernel_function")` returns a `KernelHandle`. Store it in the layer struct at load time, not per-step.
5. **Wire the launch** via the `KernelLaunch` builder in `spark-model/src/layers/ops.rs`:
   ```rust
   KernelLaunch::new(gpu, self.kernel_handle)
       .grid([num_tokens, 1, 1])
       .block([256, 1, 1])
       .shared_mem(shared_bytes)
       .arg_ptr(input)
       .arg_ptr(output)
       .arg_u32(hidden_size)
       .arg_f32(eps)
       .launch(stream)
   ```
6. **Benchmark.** Add a shape in `atlas-spark-bench` or a micro-benchmark in the relevant primitive crate. A kernel without a benchmark is not allowed to claim "faster".
7. **Verify correctness** against a PyTorch reference on a fixture tensor. Numerical diff tolerance: for BF16 outputs, abs-tol 1e-3 / rel-tol 1e-2 is a typical starting point.

## Anti-patterns

- **Don't branch on runtime flags inside the hot loop.** If a kernel needs two variants (NVFP4 vs BF16 KV), make them two kernels.
- **Don't try to be generic.** The whole point is specialisation. A kernel that works for three batch sizes is usually slower than three kernels that work for one each.
- **Don't call `cudaDeviceSynchronize` anywhere inside a kernel launch path.** `GpuBackend::synchronize(stream)` exists for explicit syncs. Random device-sync calls break CUDA graph capture and defeat pipelining.
- **Don't mutate `__constant__` memory at runtime.** Upload it once at load time; use it forever.

## What to read next

- Per-format specifics: [NVFP4](./nvfp4.md), [FP8](./fp8.md)
- Per-op specifics: [Attention & Paged KV Cache](./attention.md), [MoE](./moe.md), [SSM](./ssm.md)
- Per-feature specifics: [Speculative Decoding](./mtp.md), [XGrammar](./xgrammar.md)
- Measuring results: [Benchmarking](../operations/benchmarks.md)
- Authoritative designs: [`docs/design/`](https://github.com/Avarok-Cybersecurity/atlas/tree/main/docs/design) in the repo
