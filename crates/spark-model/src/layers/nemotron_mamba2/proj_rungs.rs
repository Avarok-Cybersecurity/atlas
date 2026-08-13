// SPDX-License-Identifier: AGPL-3.0-only

//! Per-quantization-arm projection-batching rungs for the Nemotron-H SSM
//! layer, and the evidence each one rests on.
//!
//! `decode_multi_seq`'s batched body is bit-identical to the per-seq default
//! loop everywhere EXCEPT the two projections, which are the only phase that
//! reorders FP accumulation. So the projections carry their own threshold,
//! and it is per arm because each arm's batched kernel is a different piece
//! of code with a different relationship to its M=1 twin.
//!
//! Standing gates (all in `crates/spark-model/examples/`), each a BYTE
//! compare at the production Nemotron projection shapes with a 1-ULP negative
//! control so a "byte-identical" verdict can never be vacuous:
//!
//!   * `w8a16_batch_bitparity_microtest.rs` — FP8 arm, PASSES → rung 2
//!   * `w4a16_batch_bitparity_microtest.rs` — NVFP4 arm, PASSES → rung 2
//!   * `bf16_batch_bitparity_microtest.rs`  — BF16 arm, FAILS; it exists to
//!     quantify the divergence that keeps that arm at rung 8

/// Projection-batching threshold on the NVFP4 (`w4a16`) arm.
///
/// **2, and PROVEN rather than assumed — same standard as the FP8 arm.**
///
/// Milestone A pinned this at 8 on two grounds, and both are now settled:
///
///   * the numeric one was RIGHT about the defect. `w4a16_gemv_batchm_impl`
///     really did diverge from `w4a16_gemv`, in three independent ways: it
///     walked `k16 = lane` stride 64 where the M=1 kernel walks `k16 = lane*2`
///     in stride-128 PAIRS into two accumulators; it pre-multiplied the FP8
///     group scale into each unpacked weight where the M=1 kernel factors it
///     out of the 16-FMA block; and it fused `acc += x*w0 + y*w1` where the
///     M=1 kernel does two separate accumulations (the `w8a16_gemv_batch4`
///     defect, a third time). Byte-comparing at the production Nemotron
///     shapes found 178 of 180 legs differing, up to 62 BF16 elements per
///     launch, max|delta| 0.0625. Milestone B fixed the kernel — the batched
///     body now mirrors `w4a16_gemv` chunk-for-chunk and FMA-for-FMA while
///     still reading the packed weight ONCE for all M rows. All 180 legs are
///     byte-identical, with 12 non-vacuity controls firing.
///     `examples/w4a16_batch_bitparity_microtest.rs` is the standing gate and
///     fails without that fix.
///
///   * the observational one is INVALID and is deleted here. It read: "rung-4
///     batching was probed and REJECTED at milestone A: 2 of 6 C=3/C=4 temp-0
///     fact probes flipped the P&P answer 1813 -> 1995." 1995 is what a
///     HALF-ZEROED prefill MoE emits — the `moe_w4a16_grouped_gemm` N-tile
///     bug, fixed on this branch (PR #474). An independent verifier
///     reproduced both "1995" and the wrong-capital answer from the UNFIXED
///     build at C=1, single request, with no batching in play at all. The
///     probe was reading a prefill defect, not a projection-batching one.
///
/// Rungs 2/4/8/16 therefore all batch bit-exactly. Above 16
/// `w4a16_gemv_batch16` silently truncates, so `batched_nvfp4_proj` branches
/// to the any-M tile GEMM, which is NOT bit-identical — but that is rung 24+,
/// i.e. `n_decode >= 17`.
///
/// COST OF THE FIX (kernel-only, `examples/batchm_bench.rs`, N=K=5120, 3 reps,
/// GB10): mirroring the M=1 order needs two accumulator arrays per row, which
/// raises batch4 48->61, batch8 54->79, batch16 70->126 registers and costs
/// one CTA/SM. batch4 M=4 71.7 -> 85.5 us, batch8 M=8 ~110 -> ~169 us,
/// batch16 M=8 ~123 -> ~204 us. Still ~3x better than the M separate
/// `w4a16_gemv` launches the sub-rung path pays (~4 x 65 us at M=4), so
/// lowering the rung is a win at every M; recovering the lost occupancy
/// (e.g. staging the second accumulator in smem) is open work.
///
/// END TO END on the model that actually takes this arm — Nano-30B-A3B-NVFP4,
/// GB10 dgx-00, 45-second fixed-concurrency windows of 400-token stories with
/// `min_tokens == max_tokens`, best of 3, sum-of-stream / aggregate tok/s.
/// LEFT = branch tip (rung 8 + the unfixed kernel), RIGHT = this commit:
///
///   C=1   69.2 / 66.1  ->  68.8 / 65.8    n=1 never enters the batched body
///   C=2   75.3 / 70.5  ->  75.5 / 73.8
///   C=4   79.3 / 78.4  ->  85.5 / 81.8    the clearest win (+7.8% / +4.3%)
///   C=8  100.3 / 99.8  -> 100.8 / 97.5    both builds batch here; the ~2%
///                                         aggregate dip is the kernel's
///                                         occupancy cost, at noise level
///
/// So the throughput case for rung 2 is modest — Nano's decode is not
/// dominated by SSM projection DRAM the way Lightning's is. The case for
/// rung 2 is the PROOF: below it, an arm that is byte-identical to the
/// reference was paying an M-fold weight read for nothing.
///
/// DETERMINISM (temp-0, 6 facts, C=1 control): 0/180 divergences at C=2 and
/// 0/180 at C=6 on BOTH builds. C=4 flakes on both at the same rate with the
/// same signatures (tip 3/180, this commit 4/180 — e.g. `1813` -> `18,813`),
/// so that flake is PRE-EXISTING family nondeterminism reached through some
/// other batched phase, not something this rung introduces: the tip does not
/// even batch projections at C=4.
pub(super) const MAMBA2_PROJ_MIN_NVFP4: usize = 2;

/// Projection-batching threshold on the native-BF16 arm.
///
/// **Stays at 8 — MEASURED not bit-exact, and not fixably so.**
///
/// Batching here does not swap in a batched GEMV; it swaps
/// `m x dense_gemv_bf16` (K split over 64 lanes, warp-shuffle reduction tree)
/// for one `dense_gemm_bf16_pipelined` (m16n8k16 tensor-core MMA marching a
/// single FP32 accumulator over K in 32-wide steps). Those are different
/// algorithms, not a reassociation that can be un-fused the way the FP8 and
/// NVFP4 GEMVs were.
///
/// `examples/bf16_batch_bitparity_microtest.rs` measures it: at the
/// production Nemotron projection shapes ([10304 x 2688], [2688 x 4096], and
/// the Super-class [18560 x 4096] / [4096 x 8192]), seeds 1/99/12345, every
/// M in {2,3,4,6,8,12,16} — 84 of 84 legs differ, 0.08%-0.50% of output
/// elements, max|delta| 0.0625, max relative delta 0.60. Not a rounding
/// wobble; a genuinely different reduction.
///
/// So this rung keeps its milestone-A value on the same logic the NVFP4 arm
/// used to: rung 8 is first reached at `n_decode >= 5`, where the family
/// already diverges, so batching there costs no determinism it had.
///
/// The clean way to lower it is a bit-exact batched BF16 GEMV
/// (`dense_gemv_bf16_batch2` is exactly that at M=2, and an M-generalised
/// `dense_gemv_bf16_batchm` exists on the Laguna multi-seq branch) rather
/// than a tile GEMM. That is a kernel port, not a rung edit, and is
/// deliberately NOT done here.
pub(super) const MAMBA2_PROJ_MIN_BF16: usize = 8;

/// Projection-batching threshold on the native block-scaled FP8 arm — the
/// arm Lightning-30B actually takes (`ATLAS_NEMOTRON_NATIVE_FP8_SSM`
/// defaults on and the checkpoint quantizes `mixer.in_proj`/`mixer.out_proj`
/// to FP8 block scales).
///
/// **2, and unlike the other two arms that is PROVEN, not assumed.**
///
/// `w8a16_gemv_batch4.cu` claimed bit-identity with `w8a16_gemv` and the
/// claim was false: its inner loop accumulated `acc += lo*w0 + hi*w1` where
/// the M=1 kernel computes `acc += lo*w0; acc += hi*w1;`, and those associate
/// differently in FP32. A byte compare at the real Lightning projection
/// shapes ([10304 x 2688] in_proj, [2688 x 4096] out_proj, seeds 1/99/12345)
/// found 1-9 differing BF16 elements per launch, up to 4.0 apart, at every
/// M in {2,3,4,8,12,16} — the first sighting of the silent FP-reordering
/// class that the NVFP4 arm above turned out to carry too. The cosine
/// microtest that
/// was supposed to catch this passed, because cos>=0.99999 cannot see a
/// handful of flipped elements in 10304.
///
/// Milestone B fixed the kernel (split the fused add) rather than working
/// around it, so all 36 legs now report `byte-identical=true` and the
/// batched tier is a pure weight-DRAM saving with no numeric consequence
/// whatsoever. `examples/w8a16_batch_bitparity_microtest.rs` is the standing
/// gate and fails without that fix.
///
/// Rungs 2/4/8/12/16 therefore all batch bit-exactly. Above 16 the ladder
/// leaves the GEMV tiers for the tile GEMM, which is NOT bit-identical — but
/// rung 24 is only reached at `n_decode >= 17`, far above the family's
/// pre-existing C>=5 divergence onset.
///
/// Measured on GB10 (dgx-00, idle), Lightning NVFP4, 400-token story sweeps,
/// sum-of-stream tok/s, best of 2 reps, milestone-B tip vs the same tip with
/// only the gates at their milestone-A values:
///
///   C=1   69.2 ->  69.2  (+0.0%)   n=1 never enters the batched body
///   C=2   69.7 ->  85.2  (+22.2%)
///   C=4   75.1 -> 103.8  (+38.2%)
///   C=8  109.1 -> 114.1  (+4.6%)   rung 8 already batched projections;
///                                  this delta is the strided conv/scan alone
pub(super) const MAMBA2_PROJ_MIN_FP8: usize = 2;
