# GPU backend extensions: graph templates, exec update, batched copies (lane L08)

The `GpuBackend` trait (`crates/spark-runtime/src/gpu.rs`) gains these
extension methods: CUDA graph templates with a
topology signature, in-place `cuGraphExecUpdate`, `cuGraphUpload`, batched async
host/device copies over `cuMemcpyBatchAsync`, non-blocking `cuEventQuery`, and
`cuMemHostRegister` registration recorded as tier-owned in `pinned_hosts`. Every method
has a default body, so the mock and the Metal backend compile unchanged, and nothing on
an existing path calls them: with the port rows off, every target behaves as at
`52d6b0e9a`.

## What came from where

### Cinference (translation, Apache-2.0)

- Upstream: https://github.com/satellitedown/cinference, commit `383e5db`, itself derived
  from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer).
- File and lines: `src/core/decode_graph.cpp:62-80`,
  `DecodeGraphDefinition::topology_signature`.
- What it does there: walks a captured `cudaGraph_t` and hashes each node's type and,
  for kernel nodes, the kernel function pointer (FNV-1a with a kernel mix constant),
  returning the node count beside the hash. Cinference uses it to group CUDA Graph
  captures into topology classes.
- Atlas files:
  - `crates/spark-runtime/src/gpu/port_ext.rs` (`topology_signature`, the hash over a
    backend-neutral `(node type, function)` list);
  - `crates/spark-runtime/src/cuda_backend/gpu_impl_graph_update.rs`
    (`graph_topology_signature_cu`, the node walk over the driver API).
- Changes for Atlas: translated from C++ over the CUDA runtime API to Rust over the
  driver API (`cuGraphGetNodes`, `cuGraphNodeGetType`, `cuGraphKernelNodeGetParams`,
  resolved through `cuGetProcAddress`); the hash runs over a plain node list so the mock
  backend and the CPU tests use the same function; the node count is folded into the
  one returned `u64` instead of being returned beside it; a node with no function
  (launched from a library kernel) is identified by its kernel handle.
- License: Apache License 2.0, `LICENSES/Apache-2.0.txt`. Cinference's NOTICE paragraph
  is reproduced in `NOTICE`. Each derived file carries the attribution header under its
  SPDX line.

### LMCache and vLLM (design only, no code copied)

- LMCache `csrc/cuda/mp_mem_kernels.cu:358-849` (commit `3aefae1`, Apache-2.0): one
  `cuMemcpyBatchAsync` per transfer, the CUDA 12.8 versus 13.x signature split (the 13.x
  form drops `failIdx`), and a support probe before use.
- vLLM `csrc/cache_kernels.cu:110-190` (commit `8a36f08`, Apache-2.0): the batch entry
  point resolved through `cuGetProcAddress` so the binary runs on drivers without it,
  routing the legacy NULL stream (which the batch API rejects) to a per-copy fallback,
  and zero location hints with `CU_MEMCPY_SRC_ACCESS_ORDER_ANY` only for sources no
  stream writes concurrently.
- Atlas file: `crates/spark-runtime/src/cuda_backend/gpu_impl_batch_copy.rs` (written for
  Atlas from these descriptions; the probe is a scrambled two-half round trip compared
  byte for byte, and the first failing batch call latches the path off).

## Atlas-native parts

`cuda_backend/port_ffi.rs` (runtime resolution of every new driver entry point, with
the ABI version pinned per symbol and compile-time layout checks of the driver structs),
`pinned_hosts.rs` (tier-owned regions, which the transient `copy_h2d_async` path no longer
synchronizes on), the mock emulation (`gpu/mock_port.rs`), the tests
(`gpu/port_ext_tests.rs`) and the GPU oracle
(`crates/spark-model/examples/gpu_ext_selftest.rs`).

## Receipts owed (GPU run)

- `gpu_ext_selftest` on the RTX 4060 Ti: every gate PASS (templates, signature, exec
  update, batched round trip, event polling, host registration).
- Batched copy GB/s through L07's `pinned_probe`.
- L09's `graph_exec_update` row (greedy SHA equal at C=1 and C=8, instantiations
  flat after warmup).
