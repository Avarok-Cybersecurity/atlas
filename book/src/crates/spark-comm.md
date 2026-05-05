# spark-comm

**Role:** the multi-GPU collective-ops abstraction. One trait, two impls: NCCL for real distributed runs and a no-op backend for single-GPU.
**Key file:** `src/lib.rs` (`CommBackend` trait), `nccl.rs` (raw NCCL FFI), `nccl_backend.rs` (the `NcclBackend` impl).

## Why this is its own crate

Multi-GPU in Atlas is **Expert Parallelism (EP)** — the MoE experts of models beyond one GB10's weight budget (122B, 119B, 229B) are split across two nodes connected via RoCEv2. Token dispatch between ranks goes through NCCL all-reduces and send/recv.

`spark-comm` isolates the NCCL surface so that:

1. Single-GPU deployments never link against NCCL (the binary loads a `SingleGpuBackend`).
2. Tests for the scheduler and the layer code can run against the no-op impl on CI.
3. Porting to a different collective-ops library (RCCL for AMD, Metal MPS, oneCCL) is a new `CommBackend` impl, nothing else changes.

## The trait

```rust
pub trait CommBackend: Send + Sync {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()>;
    fn all_gather(&self, send: u64, recv: u64, bytes: usize) -> Result<()>;
    fn reduce_scatter(&self, send: u64, recv: u64, bytes: usize) -> Result<()>;
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()>;
    fn send(&self, ptr: u64, bytes: usize, peer: usize) -> Result<()>;
    fn recv(&self, ptr: u64, bytes: usize, peer: usize) -> Result<()>;
    fn barrier(&self) -> Result<()>;
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    fn stream(&self) -> u64;
    fn set_stream(&mut self, stream: u64);
}
```

All pointer arguments are `u64` (matching CUDA's `CUdeviceptr`) to avoid coupling the crate to `spark-runtime`'s `DevicePtr`. Every op is stream-associated — collectives and kernel launches can be pipelined through CUDA graph capture.

## `SingleGpuBackend` — the no-op

```rust
pub struct SingleGpuBackend;

impl CommBackend for SingleGpuBackend {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> { Ok(()) }
    fn all_gather(&self, _s: u64, _r: u64, _b: usize) -> Result<()> { Ok(()) }
    // ...
    fn rank(&self) -> usize { 0 }
    fn world_size(&self) -> usize { 1 }
}
```

This is what runs in single-GPU serving. The expert-parallel code paths in `spark-model::layers::moe` still call `comm.all_reduce(...)` — the op is a no-op under `SingleGpuBackend` and an actual NCCL call under `NcclBackend`. The caller never branches on `world_size`.

## `NcclBackend` — the real impl

In `nccl_backend.rs`. Uses the unsafe NCCL FFI in `nccl.rs`. Construction flow:

1. The `master` rank (0) calls `ncclGetUniqueId` and publishes the id to the scheduler's rendezvous port (`--master-addr`, `--master-port`, default 29500).
2. Every rank (including master) dials the rendezvous, receives the id.
3. All ranks call `ncclCommInitRank(world_size, id, rank)` in parallel; the call is collective and blocks until every rank has joined.

The NCCL env layer is fussy on GB10 — the scripts in `scripts/start-ep2.sh` + `scripts/start-minimax-ep2.sh` pin the critical vars:

| Variable | Value | Reason |
|---|---|---|
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | Forces the InfiniBand/RoCE interface, not the mgmt ethernet |
| `NCCL_IB_DISABLE` | `0` | IB transport enabled |
| `NCCL_NET_GDR_LEVEL` | `5` | GPUDirect RDMA — skip the host bounce |
| `NCCL_NVLS_ENABLE` | `0` | NVLink-SHARP would crash on GB10; force off |
| `NCCL_IB_HCA` | `mlx5_0` | The RoCE HCA device |
| `GLOO_SOCKET_IFNAME` | `enp1s0f0np0` | Same ifname for Gloo fallback paths |

These are worth the paragraph — a mis-set `NCCL_SOCKET_IFNAME` on GB10 will silently fall back to the 1 GbE management interface and drop EP=2 throughput by an order of magnitude.

## The EP=2 throughput path

For Qwen3.5-122B-A10B NVFP4 at EP=2:

- 128 experts per rank (256 total).
- Token dispatch: the gate runs on every rank, top-k expert IDs are selected, tokens destined for remote experts are `reduce_scatter`'d to the owning rank.
- Expert compute happens locally.
- Expert outputs are `all_gather`'d back.
- Result: ~46 tok/s sustained on 600-token decodes (see [Multi-GPU](../operations/multi-gpu.md)).

The bandwidth pressure is all in the dispatch + gather, which is why RoCEv2 with GDR matters. A plain TCP NCCL falls off by 3×.

## The critical MTP-flag symmetry rule

A subtle footgun: when the head (rank 0) runs with `--speculative --mtp-quantization nvfp4 --num-drafts N`, the worker **must** be started with the same flags. If not, the MTP verify command from the head lands in the worker's SSM layer without intermediate buffers allocated and you get an SSM intermediate-buffer error. `scripts/start-ep2.sh` handles this; a manual two-command launch does not, and it has bit multiple contributors. See the [Multi-GPU chapter](../operations/multi-gpu.md).

## NCCL safety in tests

The unit tests for the expert-parallel layer code do not instantiate `NcclBackend`. They hold a `Box<dyn CommBackend> = Box::new(SingleGpuBackend)` and verify the code path by checking that the layer calls `all_reduce` at the right moment — the launch recorder in `MockGpuBackend` plus a trace in `SingleGpuBackend` is enough. The real NCCL path is validated by `scripts/test-minimax-ep2.sh` against a live two-node cluster.

## What's explicitly not here

- **No kernel code.** The actual token-dispatch kernels for EP=2 live in `kernels/gb10/<model>/<quant>/moe_dispatch.cu`.
- **No scheduler logic.** That's `spark-server::scheduler`.
- **No RDMA-specific code.** Atlas talks through NCCL; NCCL talks through `libibverbs`/`librdmacm`. We do not bypass.

Adding a new collective-ops library is a single `impl CommBackend` in a new module here plus a selection arm in `spark-server::main` that picks the right backend given the vendor.
