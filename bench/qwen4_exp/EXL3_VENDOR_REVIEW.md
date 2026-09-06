# EXL3 vendor review — 2026-09-05

Read-only review of Atlas PR 834 while the wider-draft work continued. No Atlas source edits. Upstream checkout: `/tmp/atlas-exl3-upstream-review`; measurement-discipline skill read before review. No throughput measurements or projected speedups claimed.

## Provenance and already incorporated changes

Upstream reviewed: [499890c75d20d8e7c9d061f37189ae611a5c9f0b, v1.4.6](https://github.com/turboderp-org/exllamav3/commit/499890c75d20d8e7c9d061f37189ae611a5c9f0b), dated September 2, 2026.

Byte comparison against Atlas `.research/exllamav3_ref` found the GEMM/GEMV/mgemm host and device sources, codebook, Hadamard, reconstruction, and blocksparse-MLP sources unchanged from that upstream revision. The one differing matched file, `ngram.cu`, changes Windows host-I/O handling, not Linux device arithmetic. This compares the reference snapshot, not Atlas's adapted vendor files: retain Atlas-specific adaptations during any refresh.

Already incorporated or not applicable:

- [fd11c82eb3aa8bf73a7930971a713ee735ea950d](https://github.com/turboderp-org/exllamav3/commit/fd11c82eb3aa8bf73a7930971a713ee735ea950d): multi-token expert range filtering preserves slot positions. Present in the reference and vendored `kernels/gb10/common/exl3_vendor/exl3_gemm_kernel.cuh`.
- Atlas's stable per-token shape/split-K replay is an additional local correctness feature (`crates/spark-model/src/layers/ops/exl3_matmul/mgemm.rs:101`), not an upstream update to fetch. Upstream still sizes the ordinary mgemm launch from the full slot batch.
- Atlas's deterministic per-slot MoE prefill reduction is a local improvement (`kernels/gb10/common/exl3_vendor/exl3_moe_kernel.cuh:41`); upstream still has the atomic output accumulation arm. Do not overwrite this adaptation.
- [7666d62f2eded360ec64654f29021e83cbf0b3f4](https://github.com/turboderp-org/exllamav3/commit/7666d62f2eded360ec64654f29021e83cbf0b3f4), the newest QSA fix, replaces Python attribute reads with `getattr` for modules without indexers. It is not a missing numerical CUDA fix in Atlas.

## Best bounded follow-ups

1. **EXL3 block graph capture — missing host capability, medium/high integration risk.** Atlas's `crates/spark-model/src/layers/ops/exl3_dense/launch_state.rs:37` says cooperative launches are never graph-capturable, and `crates/spark-model/src/layers/moe/forward_exl3.rs:157` refuses capture. Upstream explicitly captures `run_bszN_gr` in [blocksparse_mlp.cpp:280–293](https://github.com/turboderp-org/exllamav3/blob/499890c75d20d8e7c9d061f37189ae611a5c9f0b/exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp#L280), using stream capture in [graph.cu:30–51](https://github.com/turboderp-org/exllamav3/blob/499890c75d20d8e7c9d061f37189ae611a5c9f0b/exllamav3/exllamav3_ext/graph.cu#L30), around actual cooperative launches in [exl3_gemm.cu:619](https://github.com/turboderp-org/exllamav3/blob/499890c75d20d8e7c9d061f37189ae611a5c9f0b/exllamav3/exllamav3_ext/quant/exl3_gemm.cu#L619). Start with a routed-expert-only graph at fixed K, fixed scratch and stable geometry. Do not just remove the model veto: PLE host I/O/aux snapshots, changing pointers and shared cross-stream locks/fences remain separate integration concerns. Validate actual EXL3 captured/eager outputs and state before measuring throughput.

2. **Fuse the two plain ingress staging kernels — low arithmetic risk.** `moe_decode.rs:226` and `:237` separately launch `exl3_moe_stage_routing` and `exl3_moe_replicate_a_bf16`. Their bodies at `kernels/gb10/common/exl3_matmul.cu:332` and `:352` independently write local expert IDs/FP16 probabilities and replicated FP16 activation rows. A combined adapter can retain the exact `__float2half_rn` casts, remote `-1` sentinel and token-to-slot indexing while leaving all three stable mgemm launches untouched. This removes one launch per routed block by construction; wall-clock benefit is unmeasured. Extend the existing serial-vs-batched GPU parity leg with exact staging-buffer equality, nonuniform probabilities, remote slots and K=2/3/4 before timing.

3. **Cache actual per-kernel GEMV occupancy — medium risk.** Atlas `exl3_matmul.rs:264` assumes one block/SM or uses the global `ATLAS_EXL3_GEMV_OCC` override. Upstream queries and caches occupancy separately for each instantiated kernel in [exl3_gemv.cu:123–154](https://github.com/turboderp-org/exllamav3/blob/499890c75d20d8e7c9d061f37189ae611a5c9f0b/exllamav3/exllamav3_ext/quant/exl3_gemv.cu#L123). This can inform both narrow/wide selection and the cooperative grid cap. Add a backend occupancy query, cache at model construction, and preserve matching serial/verify selection. Do not infer a speedup: selected kernel/reduction geometry can change, requiring numerical and performance A/B checks.

4. **Offline shared launch-plan tuning — lower priority, medium/high numerical risk.** Upstream's mgemm wrapper contains a cooperative shape/grid autotuner [exl3_gemm.cu:545–590](https://github.com/turboderp-org/exllamav3/blob/499890c75d20d8e7c9d061f37189ae611a5c9f0b/exllamav3/exllamav3_ext/quant/exl3_gemm.cu#L545); Atlas ports the fallback heuristic. [555ee4f685159d0cd2ad117e34469d8847025693](https://github.com/turboderp-org/exllamav3/commit/555ee4f685159d0cd2ad117e34469d8847025693) adjusts its batch-key range. If explored, tune one token's expert geometry offline and freeze that same plan for serial and verification. Independent per-K runtime tuning would reopen the exact mismatch just repaired. No performance conclusion from source inspection alone.

## Standalone cooperative capture smoke

Temporary source `/tmp/atlas-exl3-coop-capture-smoke.cu`, binary `/tmp/atlas-exl3-coop-capture-smoke`. Compiled with `/usr/local/cuda/bin/nvcc -arch=sm_121a -O2` and run once on this host, without modifying the repository. The test warms a two-block cooperative kernel, captures the same launch on a nonblocking stream, instantiates the graph, then performs three replays after zeroing its output. The kernel uses `cooperative_groups::this_grid().sync()` and checks a cross-block result on every replay.

Observed output:

```text
device=NVIDIA GB10 cc=12.1 runtime=13000 driver=13000 cooperative=1
PASS cooperative launch + grid.sync stream capture and 3 graph replays
```

This proves cooperative stream capture support on this CUDA 13.0/GB10 host. It does **not** prove that Atlas's complete EXL3 path, its driver launch wrapper, PLE, or verify rollback is graph-ready. No latency/throughput claim was measured.

## Implemented follow-up

Ingress fusion and the later concurrency/state-lifetime fixes are documented in
[MTP_PERFORMANCE.md](MTP_PERFORMANCE.md), including exact parity, measured limits,
and the remaining graph-capture boundary.
