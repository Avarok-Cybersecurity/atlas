# SBIO: Business Logic vs I/O

**SBIO** — *Separation of Business logic and I/O* — is the user-level naming of a specific pattern the Atlas codebase applies aggressively: business logic never performs I/O directly. It calls a trait. Real I/O is implemented behind that trait. Tests swap in mock implementations.

The payoff is concrete. The Atlas test suite runs ~80% of the code without a GPU. You can verify the scheduler's fairness properties, the OpenAI/Anthropic protocol parsers, the sampler's numeric behavior, the tokenizer's template expansion, and every weight loader's shape checks on a vanilla Linux laptop. The only tests that need a GB10 are the ones that exercise a real CUDA kernel end-to-end.

## The I/O surfaces in Atlas

Four kinds of I/O happen at runtime. Each goes through a dedicated trait:

| I/O surface | Trait | Crate | Real impl | Mock impl |
|---|---|---|---|---|
| GPU memory, kernel launch, streams, events, graphs | `GpuBackend` (27 methods) | `spark-runtime` | `AtlasCudaBackend` (via `cudarc`) | `MockGpuBackend` — records launches, does not execute |
| Collective comms (all-reduce, broadcast, send/recv) | `CommBackend` | `spark-comm` | `NcclBackend` | `SingleGpuBackend` — every op is a no-op |
| Weight-blob loading | `WeightStore` (implicit — wraps safetensors) | `spark-runtime::weights` | `fast_weights` (`O_DIRECT` + pipelined) or mmap fallback | `WeightStore` directly against an in-memory map |
| HTTP | `axum::Router` handlers | `spark-server` | `axum::serve(...)` over TCP | `axum::Router::into_make_service()` tested via `tower::ServiceExt::oneshot` |

Everything above these traits — the scheduler, the engine, the layer assembly, the sampler, the tokenizer, the tool parser, the rate limiter — is pure Rust. It contains no `cudaXxx`, no socket call, no file-open, no `mpi_*`.

## What the trait boundary looks like

`GpuBackend` is the most load-bearing of the four. Simplified shape (full trait in `crates/spark-runtime/src/gpu.rs`):

```rust
pub trait GpuBackend: Send + Sync {
    // Memory
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;
    fn free(&self, ptr: DevicePtr) -> Result<()>;
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()>;
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()>;
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()>;
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()>;
    fn total_memory(&self) -> Result<u64>;
    fn free_memory(&self) -> Result<u64>;

    // Kernel launch
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle>;
    fn launch(
        &self, func: KernelHandle, grid: [u32; 3], block: [u32; 3],
        shared_mem: u32, stream: u64, params: &mut [*mut c_void],
    ) -> Result<()>;

    // Streams
    fn default_stream(&self) -> u64;
    fn synchronize(&self, stream: u64) -> Result<()>;

    // Optional (default no-op impls)
    fn begin_capture(&self, stream: u64) -> Result<()> { Ok(()) }
    fn end_capture(&self, stream: u64) -> Result<GraphHandle> { unimplemented!() }
    fn launch_graph(&self, graph: GraphHandle, stream: u64) -> Result<()> { unimplemented!() }
    // ... events, host-pinned alloc, thread binding, etc.
}
```

A layer's forward pass calls `gpu.launch(...)`. It does not know, and cannot know, whether `gpu` is `AtlasCudaBackend` or `MockGpuBackend`. That opacity is the whole point.

## The mock backend

`MockGpuBackend` is in `crates/spark-runtime/src/gpu.rs` alongside the trait. It does not talk to a GPU — it keeps a bump-allocator of fake `DevicePtr` values, records every launch in a `Vec<LaunchRecord>`, and returns `Ok(())` from every op. Typical test:

```rust
#[test]
fn engine_runs_correct_layer_sequence() {
    let gpu = MockGpuBackend::new();
    let cfg = ModelConfig::fixture_qwen3_5_small();
    let engine = InferenceEngine::build_for_test(&cfg, &gpu).unwrap();

    engine.decode_step(&mut ctx).unwrap();

    let launches = gpu.drain_launches();
    assert_eq!(launches.len(), cfg.num_hidden_layers * KERNELS_PER_LAYER);
    assert_eq!(launches[0].module, "attention");
    assert_eq!(launches[0].function, "prefill_attn_v47");
}
```

No GPU, no `nvcc`, no `cudarc` ever opens the driver. The test verifies a behavioral property of the *business logic* — the number and order of kernel launches — without depending on the kernel actually executing correctly.

## The single-GPU CommBackend

`SingleGpuBackend` in `crates/spark-comm/src/lib.rs` is the distributed-comms analogue:

```rust
pub struct SingleGpuBackend;

impl CommBackend for SingleGpuBackend {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> { Ok(()) }
    fn all_gather(&self, _send: u64, _recv: u64, _bytes: usize) -> Result<()> { Ok(()) }
    fn reduce_scatter(&self, _s: u64, _r: u64, _b: usize) -> Result<()> { Ok(()) }
    fn broadcast(&self, _ptr: u64, _bytes: usize, _root: usize) -> Result<()> { Ok(()) }
    fn rank(&self) -> usize { 0 }
    fn world_size(&self) -> usize { 1 }
    // ...
}
```

Every collective op is a no-op. The single-GPU serving path holds a `Box<dyn CommBackend>` that happens to be `SingleGpuBackend`; the multi-GPU path holds an `NcclBackend`. The layer code never knows which one is live.

## Business logic that benefits

The SBIO pattern makes the following blocks fully testable on CI without any GPU:

- **Scheduler** (`spark-server/src/scheduler.rs`): SLAI deadline logic, chunked-prefill budget enforcement, KV page allocation and eviction. Tested with a `MockGpuBackend` standing in for the KV cache.
- **Engine** (`spark-model/src/engine.rs`): layer ordering, speculative-decode verify + accept logic, sampler integration. The layer trait objects hold mock kernel handles.
- **Tool parsers** (`spark-server/src/tool_parser.rs`): Hermes, Qwen3-coder, Mistral formats. Input is a plain string of model output; tested with fixtures.
- **Rate limiter** (`spark-server/src/rate_limiter.rs`): token-bucket arithmetic. Pure CPU.
- **Refusal / citation extraction** (`spark-server/src/refusal.rs`): post-processing regex over plain strings. Pure CPU.
- **Weight loader** per family: shape/name checks, quantization-scheme dispatch. Tested with fixture safetensor files and `MockGpuBackend`.

The things that still require a GPU:

- Kernel correctness vs a PyTorch/reference implementation (covered by `atlas-spark-bench` and the integration tests in `tests/`).
- End-to-end model coherence (covered by `tests/run_all_models.py`).
- Multi-node collective ops (covered by `scripts/test-minimax-ep2.sh` and the EP=2 test harness).

## The `ATLAS_SKIP_BUILD` gate

The matching idea at build time: `ATLAS_SKIP_BUILD=1` makes `atlas-kernels/build.rs` emit a stub `target_ptx.rs` with empty constants. The workspace compiles, `cargo clippy` and `cargo fmt` both work, unit tests run. `nvcc` is not on the `PATH` of the GHA runner that runs the `ci.yml` workflow, and that is on purpose — CI catches type and lint regressions without needing a GPU CI pool.

The only test category that *requires* CUDA is the ones marked `#[ignore]` in the `cargo test` run, which the integration CI (not currently in this repo) would run on a real GB10 host.

## Anti-patterns we don't use

- **No global `cuda::init()`.** Every GPU op takes `&dyn GpuBackend`. There is no lazily-initialised global driver context that tests would need to mock.
- **No `cfg(test)` swaps.** The same code path runs in production and in tests — test doubles are first-class `impl`s, not conditional-compilation ghosts.
- **No `Result<T, Cuda*>` leaking upward.** Every GpuBackend method returns `anyhow::Result<T>`. Callers can't depend on which driver error surfaced.

SBIO is a discipline, not a mechanism. But because the trait surface is small, the discipline is easy to enforce in review: if you see `cudaXxx` or `socket` or `fs::open` inside `spark-model` or `spark-server`, that is a bug.

## When you're adding code

The rule is: **I/O goes through a trait; business logic goes nowhere near a syscall.** Three concrete checks when you're about to merge:

1. The file you edited — does it import `cudarc`, `nccl_sys`, `std::net`, or `std::fs`? If it's inside `spark-model`, `spark-server`'s non-handler code, or any `atlas-*` primitive crate, the answer should be *no*. Route through the matching trait.
2. Is the function unit-tested *without* `#[ignore]`? If yes, it's on the SBIO side of the line. If no, either move it over or explain in the PR why not.
3. If you're adding a new trait method, ask: is there a no-op mock impl that makes sense? If not, the method is probably the wrong shape.

Read [Kernel Dispatch](./dispatch.md) next to see how SBIO composes with the runtime dispatch path — or [spark-comm](../crates/spark-comm.md) for the collective-ops version of the same pattern.
