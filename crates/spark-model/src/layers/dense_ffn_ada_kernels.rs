// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: pick the dense FFN's FP8 GEMV and W8A8 GEMM handles, taking
//! the sm_89 drop-ins where the compiled target carries them.
//!
//! `kernels/ada/common/fp8_gemv_ada.cu` and `fp8_gemm_ada.cu` replace four
//! gb10 entry points with the SAME arguments and the SAME launch geometry, so
//! the swap is a handle and nothing else: every `ops::*` launcher and every
//! dispatch clause in `dense_ffn.rs` is unchanged.
//!
//! | role | sm_89 module::entry | everywhere else |
//! |---|---|---|
//! | M=1 decode GEMV | `fp8_gemv_ada::fp8_gemv_ada` | `w8a16_gemv::w8a16_gemv` |
//! | gate+up decode | `fp8_gemv_ada::fp8_gemv_ada_dual` | `w8a16_gemv_fused::w8a16_gemv_dual` |
//! | down with SwiGLU input | `fp8_gemv_ada::fp8_gemv_ada_silu_input` | `w8a16_gemv_fused::w8a16_gemv_silu_input` |
//! | W8A8 prefill GEMM | `fp8_gemm_ada::fp8_gemm_ada_w8a8` | `fp8_gemm_t_blockscaled::fp8_gemm_t_blockscaled` |
//!
//! Not a lever. Both kernels are bit-identical to the ones they replace (their
//! file headers give the argument), so the choice changes speed and not one
//! output bit, and the target that compiles the module is the declaration: the
//! sm_89 sources live only in `kernels/ada`, and
//! [`crate::layers::try_target_kernel`] issues no lookup where a module was
//! never built, so a gb10, Hopper or Blackwell boot audit sees exactly the
//! lookups it saw before.
//!
//! One more sm_89 kernel is resolved here without a counterpart in the table:
//! `swiglu_quant_fp8_ada::swiglu_quant_fp8_ada`
//! (`kernels/ada/common/swiglu_quant_fp8_ada.cu`) replaces a PAIR of launches,
//! `moe_silu_mul` then `per_token_group_quant_fp8`, so there is no single
//! entry to fall back to. Its handle is `0` on every other target and the
//! layer runs the pair; the rule is `dense_ffn_swiglu_quant.rs`.

use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// The fused SwiGLU + FP8 activation quantizer's `(module, entry)`.
pub const SWIGLU_QUANT_KERNEL: Entry = ("swiglu_quant_fp8_ada", "swiglu_quant_fp8_ada");

/// One `(module, entry)` pair.
pub type Entry = (&'static str, &'static str);

/// `(sm_89 drop-in, the kernel it replaces)` for each role, in table order.
pub const FP8_FFN_KERNELS: [(Entry, Entry); 4] = [
    (
        ("fp8_gemv_ada", "fp8_gemv_ada"),
        ("w8a16_gemv", "w8a16_gemv"),
    ),
    (
        ("fp8_gemv_ada", "fp8_gemv_ada_dual"),
        ("w8a16_gemv_fused", "w8a16_gemv_dual"),
    ),
    (
        ("fp8_gemv_ada", "fp8_gemv_ada_silu_input"),
        ("w8a16_gemv_fused", "w8a16_gemv_silu_input"),
    ),
    (
        ("fp8_gemm_ada", "fp8_gemm_ada_w8a8"),
        ("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled"),
    ),
];

/// The four FP8 handles the dense FFN dispatches through. A `0` handle means
/// "absent", which every dispatch site already checks.
#[derive(Clone, Copy, Debug)]
pub struct Fp8FfnKernels {
    pub gemv: KernelHandle,
    pub gemv_dual: KernelHandle,
    pub gemv_silu_input: KernelHandle,
    pub gemm_w8a8: KernelHandle,
    /// `swiglu_quant_fp8_ada` where the target compiled it, else `0` (the
    /// unfused pair runs).
    pub swiglu_quant: KernelHandle,
}

/// The drop-in where the target compiled it, else the kernel it replaces.
fn prefer(gpu: &dyn GpuBackend, (drop_in, base): (Entry, Entry)) -> KernelHandle {
    let h = crate::layers::try_target_kernel(gpu, drop_in.0, drop_in.1);
    if h.0 != 0 {
        h
    } else {
        crate::layers::try_kernel(gpu, base.0, base.1)
    }
}

impl Fp8FfnKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        let [gemv, dual, silu, gemm] = FP8_FFN_KERNELS;
        Self {
            gemv: prefer(gpu, gemv),
            gemv_dual: prefer(gpu, dual),
            gemv_silu_input: prefer(gpu, silu),
            gemm_w8a8: prefer(gpu, gemm),
            swiglu_quant: crate::layers::try_target_kernel(
                gpu,
                SWIGLU_QUANT_KERNEL.0,
                SWIGLU_QUANT_KERNEL.1,
            ),
        }
    }

    /// The M=1 decode GEMV role alone, for a layer that needs no other role
    /// (the GDN layer's `w8a16_gemv_k`): one lookup where the target has no
    /// drop-in, exactly the lookup that layer issued before.
    pub fn m1_gemv(gpu: &dyn GpuBackend) -> KernelHandle {
        prefer(gpu, FP8_FFN_KERNELS[0])
    }

    /// The W8A8 prefill GEMM role alone (the GDN layer's
    /// `fp8_gemm_t_blockscaled_k`), on the same terms as [`Self::m1_gemv`].
    pub fn w8a8_gemm(gpu: &dyn GpuBackend) -> KernelHandle {
        prefer(gpu, FP8_FFN_KERNELS[3])
    }
}

#[cfg(test)]
#[path = "dense_ffn_ada_kernels_tests.rs"]
mod tests;
