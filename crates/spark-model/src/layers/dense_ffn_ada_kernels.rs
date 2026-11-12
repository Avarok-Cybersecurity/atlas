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

use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// One `(module, entry)` pair.
type Entry = (&'static str, &'static str);

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
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_ada_kernels_tests.rs"]
mod tests;
