# L2 evict-first weight loads and the fused RMSNorm FP8 producer, after Cinference (lane L15)

**What it is.** Two of the three parts of the Ada FP8 kernel family carry
something from Cinference:

1. **L2 evict-first weight loads** (row `l2_evict_first_weights`, off). The
   `_l2ef` twins of the sm_89 FP8 GEMVs (`fp8_gemv_ada_l2ef`,
   `fp8_gemv_ada_l2ef_dual`, `fp8_gemv_ada_l2ef_silu_input` in
   `kernels/ada/common/fp8_gemv_ada.cu`, and the `fp8_gemv_ada_batchm_l2ef_*`
   entries in `kernels/ada/common/fp8_gemv_ada_batchm.cu`) create an
   evict-first L2 policy and load every weight line with it. A small snippet
   adapted from Cinference.
2. **Fused RMSNorm + FP8 group quantization** (row `rmsnorm_fp8_quant_fused`,
   off). `kernels/ada/common/rmsnorm_quant_fp8_ada.cu` is an independent
   implementation of Cinference's idea for Atlas's residual-add norm and its
   1x128 activation groups. No code copied.

The third part, the batch-M verify GEMV (row `ffn_fp8_verify_gemv`), is
Atlas-native: it is `fp8_gemv_ada.cu` with M accumulators.

**Status.** Written and CPU-tested; not yet run on a GPU. Every row is off on
every target.

## Source

| | |
|---|---|
| Upstream | Cinference, https://github.com/satellitedown/cinference, commit `383e5db` |
| Derived from | NInfer by Neroued and contributors, https://github.com/Neroued/ninfer |
| License | Apache License, Version 2.0 (root `LICENSE`, no copyright line); text in `LICENSES/Apache-2.0.txt` |
| NOTICE | the first paragraph of Cinference's `NOTICE`, reproduced in Atlas's `NOTICE` |

Files read:

| Upstream file | Lines | Use in Atlas |
|---|---|---|
| `src/ops/common/memory.cuh` | 72-84 | `l2_weight_policy`: `createpolicy.fractional.L2::evict_first.b64 %0, 1.0;` for once-read weights. The same one-line PTX instruction is in `fga_l2_evict_first_policy` (`fp8_gemv_ada.cu`) and `FgbLoadEvictFirst::make` (`fp8_gemv_ada_batchm.cu`). |
| `src/ops/common/memory.cuh` | 86-99 | The note that CUDA 13.1's ptxas leaves the cache-hint operand of some hinted fills uninitialized and that Cinference issues the hint only from CUDA 13.4. Atlas compiles the hint out on 13.0 and 13.1 and requires a SASS check on any newer toolkit before the row's timing is trusted. |
| `src/ops/linear/fp8/fp8_rmsnorm_quantize.cu` | 1-98 | Idea only: an RMSNorm that writes the FP8 GEMM's quantized input in the same pass. Upstream quantizes per token with normalized pairs staged in shared memory; Atlas quantizes per 128-element group, keeps the normalized pairs in registers, and reproduces `residual_add_rms_norm` statement for statement. |

## Changes for Atlas

- The policy feeds `ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32` direct
  loads (Atlas's GEMVs read weights into registers), not `cp.async.cg` fills.
- The hint sits in separate `_l2ef` entry points; the existing entry points are
  unchanged, and the host takes a twin only with the row on
  (`crates/spark-model/src/layers/dense_ffn_ada_kernels.rs`).
- The version gate is narrower than upstream's (13.0 and 13.1 compiled out,
  13.2+ compiled in and checked), because Atlas builds with 13.3 and the gate
  must not remove the feature that is being measured.
- The fused norm writes all five outputs of the unfused pair (hidden, residual,
  BF16 output, E4M3 codes, scales), byte-identical, and is taken only where the
  dense FFN's gate and up both take the W8A8 arm
  (`crates/spark-model/src/layers/dense_ffn_w8a8_prefill/norm_fused.rs`).

## Receipts owed before any row flips (Ada only)

- `l2_evict_first_weights`: the SASS check (every hinted `LDG` carries a
  cache-policy operand that a preceding instruction writes), then
  `examples/fp8_ada_l2hint_microtest.rs` all PASS, then a `bench5.py` ABBA
  decode gain beyond the bound.
- `rmsnorm_fp8_quant_fused`: `examples/rmsnorm_quant_fp8_microtest.rs` all
  PASS, greedy token SHA equal on the FP8 FFN route, then a TTFT ABBA gain.
- `ffn_fp8_verify_gemv`: `examples/fp8_ada_batchm_microtest.rs` all PASS,
  greedy token SHA equal at `--num-drafts 3`, then verify time down.
