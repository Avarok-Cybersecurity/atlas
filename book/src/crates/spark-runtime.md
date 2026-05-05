# spark-runtime

**Role:** everything that touches the GPU directly. `GpuBackend` trait + CUDA implementation, KV cache, prefix cache, buffer arena, sampler, fast weight loader.
**Key files:** `gpu.rs`, `cuda_backend.rs`, `kv_cache.rs`, `prefix_cache.rs`, `radix_tree.rs`, `buffers.rs`, `sampler.rs`, `fast_weights/mod.rs`, `weights.rs`.

## The load-bearing trait: `GpuBackend`

27 methods across five concerns:

- **Memory** — `alloc`, `free`, `copy_h2d`/`d2h`/`d2d`, `memset`, `total_memory`, `free_memory`, `alloc_host_pinned`.
- **Kernel launch** — `kernel(module, func)` returns a `KernelHandle`; `launch(handle, grid, block, shared, stream, args)` fires the launch.
- **Streams** — `default_stream`, `create_stream`, `synchronize`, `bind_to_thread`.
- **CUDA graphs** — `begin_capture`, `end_capture` → `GraphHandle`, `launch_graph`.
- **Events** — `create_event`, `record_event`, `stream_wait_event`.

Required methods have no default; optional methods have default panics/no-ops so a partial backend (e.g. a Metal backend without CUDA-graph support yet) still compiles. The trait's [SBIO role](../architecture/sbio.md) is discussed in Part II.

## The production impl: `AtlasCudaBackend`

In `cuda_backend.rs`. Built on `cudarc` (Rust bindings over the CUDA driver API). On construction:

1. `cudarc::driver::CudaDevice::new(ordinal)` acquires a driver context.
2. For each `PtxModule` in the provided target set: `cuModuleLoadData` uploads the PTX, then `cuModuleGetFunction` caches one `CUfunction` per exported kernel.
3. Kernel handles are stored in a `HashMap<(module: &str, func: &str), KernelHandle>`.

Per-launch the backend unpacks the kernel args into `void*[]`, sets the stream, and calls `cuLaunchKernel`. The launch-side ceremony — converting `DevicePtr`/scalar args into a `void*` pointer array — is abstracted one level up by the `KernelLaunch` builder pattern in `spark-model::layers::ops`.

## The paged KV cache (`kv_cache.rs`)

Atlas uses paged attention (à la vLLM) with block-level allocation. Core types:

```rust
pub enum KvCacheDtype {
    Bf16,    // 2 bytes/element — unquantized baseline
    Fp8,     // 1 byte/element — E4M3 + per-tensor scale
    Nvfp4,   // 0.5 bytes + per-group FP8 scale — maximum compression
    Turbo4,  // 4-bit WHT + Lloyd-Max (TurboQuant) — lower MSE than NVFP4
    Turbo3,  // 3-bit WHT + Lloyd-Max — smallest
    Turbo8,  // WHT + FP8 — outlier-resistant FP8
}
```

`PagedKvCache` holds a pool of fixed-size blocks (configurable, typically 16 tokens per block). `KvCacheConfig` derives pool sizing from `ModelConfig` + `--max-seq-len` + `--max-batch-size`. Allocation is O(1) from a free list; eviction is handled by the scheduler.

The **TurboQuant** family (`turbo3`, `turbo4`, `turbo8`) is specific to Atlas: Walsh-Hadamard rotation followed by Lloyd-Max quantization to optimal Gaussian codebook levels. For the same bit rate, turbo4 has ~2× lower MSE than NVFP4 on the kinds of activations transformers produce, because WHT flattens outliers before quantization. See `docs/design/turboquant-nightjob-2026-03-31.md` and [FP8](../deep-dives/fp8.md) / [NVFP4](../deep-dives/nvfp4.md) chapters.

## Prefix caching (`prefix_cache.rs`, `radix_tree.rs`)

RadixAttention: the system prompt shared by every request can be KV-cached once, reused forever. Implementation:

- **`radix_tree.rs`** — in-memory radix tree keyed on token sequences. Each node owns the KV pages for its token prefix.
- **`prefix_cache.rs`** — the orchestration layer. When a request arrives, the scheduler calls `prefix_cache.lookup(tokens)` which walks the tree to the deepest matching node. The KV pages for that prefix are already resident on the GPU.

Hit rates are high in practice — system prompts and few-shot examples dominate, and chat agents reuse most of their tool schemas across turns. TTFT drops ~10× on warm-cache hits. This is the feature enabled by `--enable-prefix-caching`.

**Marconi (SSM snapshots)** extends the idea to SSM layers: a full SSM state is ~GB on a 35B model, so prefix cache hits for hybrid models also need a snapshotted SSM state to be genuinely equivalent. That machinery lives partly here and partly in `spark-model`. See `docs/design/` for the SSM-snapshot-cache design.

## Buffer arena (`buffers.rs`)

One `BufferArena` per serve. Allocates every scratch buffer *once* at startup, sized for the worst-case batch × seq_len combination allowed by CLI flags. Includes:

- `hidden_states`, `residual`, `norm_output` — residual stream and its post-norm staging
- `qkv_output`, `attn_output` — attention projection outputs
- `gate_logits`, `moe_output` — MoE intermediates
- `logits` — the final `[M, vocab_size]` output
- `ssm_qkvz_scratch` — Mamba/GDN projections (sized for 3× positions for MRoPE on models that need it)
- `expert_outputs` — sized for `max(k_max, max_batch_tokens)` to cover both speculative decode (K=3) and batched MoE prefill

The arena never reallocates during serving. This is one of the invariants that makes CUDA graph capture viable — buffer addresses are graph-stable.

## Sampler (`sampler.rs`)

`SamplingParams` — `temperature`, `top_p`, `top_k`, `top_n_sigma`, `min_p`, `repetition_penalty`, `presence_penalty`. The sampler:

1. Applies penalties (presence, repetition) in-place on the logits buffer.
2. Applies `top_n_sigma` (entropy-based filter; see `docs/design/` for the theory).
3. Applies `top_p` + `top_k` + `min_p`.
4. Softmax.
5. Multinomial sampling or argmax (if `temperature == 0`).

A known bug with `temperature=0 && repetition_penalty=0` was fixed in wave-8 of the bug sweeps; the sampler now has explicit div-by-zero guards. `--adaptive-sampling` toggles an entropy-gated greedy path that avoids the full softmax+sample when the logits are effectively one-hot.

## Fast weight loader (`fast_weights/`)

Atlas's production weight loader. Modeled on `scitix/InstantTensor`:

- Each safetensors shard is opened with `O_DIRECT` (bypasses the page cache — critical on GB10 where the page cache shares physical memory with the GPU).
- One reader thread pre-fetches the next tensor's bytes into a page-aligned buffer.
- The main thread `copy_h2d`'s the current tensor while the next one is being read.
- A per-shard heuristic auto-picks between `O_DIRECT` and buffered reads: shards with > 5000 tensors pay too much per-tensor syscall overhead for `O_DIRECT`, and kernel readahead wins there.
- If the filesystem rejects `O_DIRECT` (tmpfs, overlayfs), falls back to mmap automatically.

Cold-load speedups measured vs mmap (with `posix_fadvise(DONTNEED)` between runs):

| Model | Cold mmap | Cold fast | Speedup |
|---|---:|---:|---:|
| Qwen3.5-27B (1 shard, 2.4k tensors) | 19s | 9s | **2.05×** |
| Qwen3.5-35B-A3B (2 shards, 125k tensors) | 62s | 34s | **1.84×** |
| Qwen3-Next-80B-A3B (11 shards, 298k tensors) | 166s | 110s | **1.51×** |

On by default. `--no-fast-load` reverts to mmap.

## `MockGpuBackend` and testing

The test double lives in `gpu.rs` next to the trait. Records every launch; returns success for every op. Enables the ~80% of the test suite that doesn't need a real GPU — see [SBIO](../architecture/sbio.md).

## What's explicitly not here

- **No model layers.** That's `spark-model`.
- **No HTTP.** That's `spark-server`.
- **No collective ops.** That's `spark-comm`.

`spark-runtime` is the bottom of the "things that move bits on a GPU" stack and the top of the "things a layer is allowed to call directly" stack.
