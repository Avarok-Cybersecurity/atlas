// SPDX-License-Identifier: AGPL-3.0-only

//! Whether this build may pre-dequantise NVFP4 weights to FP8 for prefill.
//!
//! **WHY.** Measured on gfx1201 (AMD Radeon AI PRO R9700, SCALE 1.7.1,
//! ROCm 7.2.0), 2026-09-17: `Ornith-1.0-9B` loads, builds and boots, and then
//! every request dies at layer 0:
//!
//! ```text
//! ssm prefill: out_proj GEMM failed: Kernel lookup
//!   w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab: Module load failed:
//!   Module 'w4a16_fp8_ldmab' not loaded
//! ```
//!
//! The chain, which is a LOAD-TIME decision producing a RUN-TIME failure three
//! files away:
//!
//! 1. `Qwen3SsmLayer::predequant_for_prefill` (`qwen3_ssm/init_fp8.rs`) builds
//!    `out_proj_fp8` unconditionally, and `Qwen3AttentionLayer`'s namesake
//!    (`qwen3_attention/prefill_weights.rs`) builds `q_fp8`..`o_fp8` the same
//!    way. Neither asks whether the GEMM that reads them exists.
//! 2. The prefill dispatch then PREFERS those copies:
//!    `qwen3_ssm/trait_prefill_helper.rs` is
//!    `if out_proj_fp8 { fp8 GEMM } else if out_proj_nvfp4_t { w4a16_gemm_n128 }
//!    else { w4a16_gemm }`, and `prefill/cache_skip.rs:126` turns the whole
//!    attention chain onto FP8 activations with `self.q_fp8.is_some()`.
//! 3. `ops::fp8_gemm_n128` (`ops/gemm_fp8_prefill.rs:48`) is DEFAULT-ON to the
//!    `ldmatrix.x4` kernel and looks it up with a hard `?`. `w4a16_fp8_ldmab.cu`
//!    is one of the 13 gfx1201 census failures and is not in this target's
//!    tree at all, so the lookup is an error rather than a fallback.
//!
//! Every arm below step 2 already handles `None`: the SSM helper falls to
//! `w4a16_gemm_n128` and then to `w4a16_gemm`, attention's `use_fp8_act` goes
//! false and its chain reaches the NVFP4 arms, and the MoE's three shared-expert
//! copies are each read behind an `if let Some`. So the fix is to not BUILD the
//! copies on a target whose FP8 prefill GEMM does not exist, and the dispatch
//! degrades on its own.
//!
//! **This is what `AVAROK_NO_FP8_PREDEQUANT=1` was for.** The Strix recipe
//! carried it, and `serve-amd.sh` dropped it once the reader was lost. This
//! module is the reader, and it adds the probe the env var should never have
//! been the only line of defence for.
//!
//! **NVIDIA is unchanged.** With the variable unset and the kernels present,
//! [`decide`] returns `None` and every caller does exactly what it did before.

use spark_runtime::gpu::GpuBackend;

/// Why the FP8 prefill copies are not being built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredequantSkip {
    /// `AVAROK_NO_FP8_PREDEQUANT` is set to something other than `0`.
    Env,
    /// A kernel the prefill dispatch would launch does not resolve on this
    /// target. Carries the `module::function` that is missing.
    KernelMissing(&'static str),
}

impl PredequantSkip {
    /// The sentence the load log prints.
    pub fn reason(self) -> String {
        match self {
            Self::Env => "AVAROK_NO_FP8_PREDEQUANT is set".to_owned(),
            Self::KernelMissing(k) => {
                format!("this target does not define {k}, which its prefill GEMM would launch")
            }
        }
    }
}

/// Which FP8 prefill kernels resolved, and whether the `ldmatrix` route is
/// armed. Taken as a struct so the decision table is pinned by a CPU test
/// rather than by the machine the test runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8PrefillKernels {
    /// `ops::fp8_gemm_n128` prefers `w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab`
    /// unless `AVAROK_FP8_LDMAB=0`. Mirrors that predicate exactly.
    pub ldmab_armed: bool,
    /// `w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab`.
    pub ldmab: bool,
    /// `w4a16::bf16_to_fp8`, which the ldmab route needs for the activation.
    pub bf16_to_fp8: bool,
    /// `w4a16::fp8_gemm_t`, the scalar arm `fp8_gemm_n128` still takes when
    /// `K % 32 != 0`, ldmab or not.
    pub scalar: bool,
    /// `w4a16::fp8_gemm_t_m128`, which `fp8_gemm_n128_m128` launches directly
    /// for the large-M prefill and which never goes through ldmab.
    pub m128: bool,
}

/// The decision. Pure: no environment reads, no lookups.
///
/// EVERY arm the dispatch could take must resolve, not just the preferred one.
/// `fp8_gemm_n128` picks between ldmab and scalar on `K % 32`, which is a
/// property of the projection and not of the build, and the caller picks
/// between `fp8_gemm_n128` and `fp8_gemm_n128_m128` on the token count, which
/// is a property of the request. A partial answer here is a serve that works on
/// short prompts and dies on long ones.
pub fn decide(no_predequant: Option<&str>, k: Fp8PrefillKernels) -> Option<PredequantSkip> {
    // PRESENCE-with-an-escape-hatch, not presence: the operator action this
    // has to support is "the probe is wrong, build them anyway", and only an
    // explicit `0` can say that.
    if matches!(no_predequant, Some(v) if v != "0") {
        return Some(PredequantSkip::Env);
    }
    if k.ldmab_armed {
        if !k.ldmab {
            return Some(PredequantSkip::KernelMissing(
                "w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab",
            ));
        }
        if !k.bf16_to_fp8 {
            return Some(PredequantSkip::KernelMissing("w4a16::bf16_to_fp8"));
        }
    }
    if !k.scalar {
        return Some(PredequantSkip::KernelMissing("w4a16::fp8_gemm_t"));
    }
    if !k.m128 {
        return Some(PredequantSkip::KernelMissing("w4a16::fp8_gemm_t_m128"));
    }
    None
}

/// Probe this backend and answer the same question.
///
/// `w4a16_fp8_ldmab` is looked up with [`crate::layers::try_target_kernel`],
/// not `try_kernel`: it is a module some targets do not compile, and the boot
/// audit treats a failed lookup as a dispatch site on a silent fallback path
/// and refuses to serve. A target that never built the module has no silent
/// fallback to record, since it has the only path it has. `w4a16` is built
/// everywhere, so its three entry points are ordinary optional lookups.
pub fn skip_reason(gpu: &dyn GpuBackend) -> Option<PredequantSkip> {
    let resolves = |m: &str, f: &str| crate::layers::try_kernel(gpu, m, f).0 != 0;
    decide(
        std::env::var("AVAROK_NO_FP8_PREDEQUANT").ok().as_deref(),
        Fp8PrefillKernels {
            // Character for character `ops::gemm_fp8_prefill.rs:39`.
            ldmab_armed: std::env::var("AVAROK_FP8_LDMAB").as_deref() != Ok("0"),
            ldmab: crate::layers::try_target_kernel(gpu, "w4a16_fp8_ldmab", "fp8_fp8_gemm_ldmab").0
                != 0,
            bf16_to_fp8: resolves("w4a16", "bf16_to_fp8"),
            scalar: resolves("w4a16", "fp8_gemm_t"),
            m128: resolves("w4a16", "fp8_gemm_t_m128"),
        },
    )
}

/// `skip_reason`, with the one-line INFO the serve log is read against.
///
/// Latched on the BACKEND (`OpCache::once`) rather than a static: every layer
/// asks, and a static would mean only the first model in the process reported
/// its decision.
pub fn skip_reason_logged(gpu: &dyn GpuBackend, what: &str) -> Option<PredequantSkip> {
    let skip = skip_reason(gpu)?;
    if gpu.op_cache().once("log:fp8_predequant_skipped") {
        tracing::info!(
            "FP8 prefill predequant skipped ({what} and every sibling): {}. \
             Prefill falls back to the NVFP4 arms; decode is unaffected. \
             AVAROK_NO_FP8_PREDEQUANT=0 forces the copies back.",
            skip.reason(),
        );
    }
    Some(skip)
}

#[cfg(test)]
#[path = "fp8_predequant_tests.rs"]
mod tests;
