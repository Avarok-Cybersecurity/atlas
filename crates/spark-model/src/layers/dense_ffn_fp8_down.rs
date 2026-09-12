// SPDX-License-Identifier: AGPL-3.0-only

//! The native-FP8 M=1 decode DOWN projection — which arm runs it (#928).
//!
//! WHY. nsys, 1xH100, Qwen/Qwen3.8-27B-FP8, 2026-09-11 round 7, C=1
//! steady-state decode step **21.891 ms**, GPU busy 96%:
//!
//! | kernel | shape | grid | launches | us each | ms/step | GB/s |
//! |---|---|---|---|---|---|---|
//! | `w8a16_gemv_silu_input` (down) | N=5120 K=17408 | 1280 | 64 | 103.9 | **6.65 (30.4%)** | **858** |
//! | `w8a16_gemv_dual` (gate+up) | N=17408x2 K=5120 | 4352 | 64 | 90.1 | 5.77 | **1,979** |
//! | `w8a16_gemv` | N=16384 K=5120 | 4096 | - | - | - | 1,852 |
//!
//! Same 89.1 MB of FP8 weights per layer on the first two rows. The down
//! projection reads them at **43% of the rate** the gate/up pair reads its
//! own. Two independent causes; this arm addresses the first.
//!
//! **1. The fused SwiGLU is recomputed per OUTPUT, not per CTA.** In
//! `w8a16_gemv_silu_input`, each of the `ceil(N/4)` CTAs gives each of its 4
//! outputs a 64-lane team, and every team walks the whole K computing
//! `silu(gate[k])*up[k]` for itself. That is `N*K` = 5,120 x 17,408 =
//! **89.1 M** SwiGLU evaluations per launch where the token needs K = 17,408
//! — a 5,120x redundancy — and each one is an `__expf` plus a TRUE FP32
//! division: `kernels/gb10/common/KERNEL.toml` builds with `--fmad=false` and
//! no `-use_fast_math`, so `g / (1.0f + __expf(-g))` lowers to the IEEE
//! division sequence, not a reciprocal. Counting ~20 ops per K element
//! against the dual GEMV's ~6, the fused kernel issues ~56 M warp
//! instructions per launch — ~60 us of pure issue on 132 SMs at 4
//! instructions/SM/cycle — on top of a 26.6 us weight stream. It is
//! issue-bound, not bandwidth-bound. The NVFP4 arm reached the same verdict
//! with ncu (SM 57% vs memory 23%) and has staged the activation once ever
//! since; the FP8 arm simply never got the same treatment. Staging it here is
//! what this module's default arm does.
//!
//! **2. This kernel family's grid is a pure function of N — LEFT OPEN.**
//! `w8a16_gemv.cu` puts `N_PER_BLOCK=4` outputs in a 256-thread CTA, so K
//! never enters the CTA count. ~8 such CTAs co-reside per SM, and H100 has 132
//! of them: grid 4352 is ~4.1 full waves (a ~2% tail), grid 1280 is ~1.2 waves
//! — one full wave plus a 224-CTA tail. A split-K GEMV was written for exactly
//! this and MEASURED AS A NULL on the H100 (down 61.8 us split-K vs 58.9 us
//! for the staged scalar kernel; the k/v shape came back at 0.67x), so it was
//! dropped rather than shipped behind a lever. Whatever the residual cost is,
//! wave quantisation alone does not explain it, and the next attempt should
//! start from a fresh profile rather than from that plan.
//!
//! PTXAS RECEIPT (`nvcc -cubin -Xptxas -v -arch=sm_90a --fmad=false`, CUDA
//! 13.0, taken 2026-09-11 on the gate box): `w8a16_gemv` uses **32 registers /
//! 1,056 B smem / 0 spills**, so 32 x 256 x 8 = 65,536 = exactly the SM's
//! register file — 8 CTAs/SM is not an estimate, it is the ptxas-pinned
//! ceiling. `w8a16_gemv_silu_input` uses **53 registers**, which is 13,568 per
//! CTA and therefore only **4 CTAs/SM**. The fused kernel halves resident
//! warps on top of the redundant transcendentals — a third, independent cost,
//! and one staging the activation removes for free.
//!
//! A CHILD module of `dense_ffn`, not a sibling: `dense_ffn.rs` is already at
//! the CI size cap, and the nsys attribution above needs room it does not
//! have.

/// Which arm the native-FP8 SiLU decode uses for `down_proj`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fp8DownArm {
    /// `w8a16_gemv_dual`, then `moe_silu_mul` stages `silu(gate)*up` once into
    /// `gate_out`, then the plain `w8a16_gemv` for down. DEFAULT.
    ///
    /// NOT bit-identical to `FusedSilu`: `moe_silu_mul` rounds
    /// `g*(1/(1+e^-g))*u` to BF16 before the GEMV consumes it, where the fused
    /// kernel keeps `(g/(1+e^-g))*u` in FP32 all the way into the dot product
    /// — a BF16 round of the activation plus a reciprocal-vs-divide
    /// difference. It is the numerics PREFILL already runs, which is the
    /// reason the NVFP4 arm made the same trade its default.
    SplitSilu,
    /// `w8a16_gemv_dual`, then the fused `w8a16_gemv_silu_input`. What shipped
    /// before #928; reachable via `ATLAS_NO_DECODE_SPLIT_SILU`, or when this
    /// target lacks `moe_silu_mul` / `w8a16_gemv`.
    FusedSilu,
    /// Neither fused path is usable — the 4-launch per-projection
    /// `w8a16_gemv` x2 + `moe_silu_mul` + down sequence.
    PerProjection,
}

/// The arm rule, as a pure function of the resolved handles and the lever, so
/// the CPU tests can pin every combination without a GPU. SSOT for the
/// `match` in `DenseFfnLayer::forward`.
///
/// Each `bool` is "this handle resolved on this target" (`KernelHandle(0)` on
/// a shadow that lacks the entry point) except `split_silu_lever`, which is
/// `ModelLevers::decode_split_silu` (`ATLAS_NO_DECODE_SPLIT_SILU`, presence).
pub(crate) fn fp8_down_arm(
    is_silu: bool,
    dual: bool,
    fused_silu: bool,
    act_mul: bool,
    plain_gemv: bool,
    split_silu_lever: bool,
) -> Fp8DownArm {
    // The dual GEMV feeds BOTH fused arms; without it there is no gate/up pair
    // staged in `gate_out`/`up_out` for either to consume.
    if !is_silu || !dual {
        return Fp8DownArm::PerProjection;
    }
    if split_silu_lever && act_mul && plain_gemv {
        Fp8DownArm::SplitSilu
    } else if fused_silu {
        Fp8DownArm::FusedSilu
    } else {
        Fp8DownArm::PerProjection
    }
}
