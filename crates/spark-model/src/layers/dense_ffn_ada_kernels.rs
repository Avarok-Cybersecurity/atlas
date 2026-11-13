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
//!
//! # Port sprint rows (all off by default, on every target)
//!
//! * `l2_evict_first_weights`: each decode GEMV role prefers its `_l2ef` twin
//!   ([`FP8_FFN_L2EF_KERNELS`]), which differs only in the L2 policy of the
//!   weight loads, then falls back to the table above.
//! * `ffn_fp8_verify_gemv`: the verify rows' batch-M GEMVs
//!   ([`verify`], `fp8_gemv_ada_batchm.cu`).
//! * `ffn_fp8_decode_gemv`: the same batch-M GEMVs on a multi-sequence decode
//!   step (`dense_ffn/fp8_decode_gemv.rs`); it primes the same handles.
//! * `ffn_fp8_m16_tc`: the 16-row tensor-core GEMM for 5 to 16 decode or
//!   verify rows (`dense_ffn/fp8_m16_tc.rs`, `fp8_gemm_ada_m16.cu`);
//!   [`Fp8FfnKernels::resolve`] primes its two handles.
//! * `rmsnorm_fp8_quant_fused`: the fused residual-add + RMSNorm + FP8 group
//!   quantizer ([`RMSNORM_QUANT_KERNEL`]) the prefill FFN-norm sites call
//!   (`dense_ffn_w8a8_prefill/norm_fused.rs`).
//!
//! All but the first have no layer field to live in, so their handles are looked up
//! HERE, at layer construction, through the backend's [`OpCache`]
//! (`spark_runtime::op_cache`): construction is on the eager build path the
//! boot kernel audit sees, and the forward-time reads of the same names are
//! then cache hits. A row that is off issues no lookup at all, so with every
//! row off this file resolves exactly what it did before the sprint.
//!
//! [`OpCache`]: spark_runtime::op_cache::OpCache

use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// The fused SwiGLU + FP8 activation quantizer's `(module, entry)`.
pub const SWIGLU_QUANT_KERNEL: Entry = ("swiglu_quant_fp8_ada", "swiglu_quant_fp8_ada");

/// The fused residual-add + RMSNorm + FP8 group quantizer's `(module, entry)`
/// (row `rmsnorm_fp8_quant_fused`).
pub const RMSNORM_QUANT_KERNEL: Entry = ("rmsnorm_quant_fp8_ada", "rmsnorm_quant_fp8_ada");

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

/// The evict-first L2 twins of the three decode GEMV roles, in
/// [`FP8_FFN_KERNELS`] order (row `l2_evict_first_weights`). Same arguments,
/// geometry and output bits as the drop-in of the same role.
pub const FP8_FFN_L2EF_KERNELS: [Entry; 3] = [
    ("fp8_gemv_ada", "fp8_gemv_ada_l2ef"),
    ("fp8_gemv_ada", "fp8_gemv_ada_l2ef_dual"),
    ("fp8_gemv_ada", "fp8_gemv_ada_l2ef_silu_input"),
];

/// The port rows that change what this table resolves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdaFp8Rows {
    /// `l2_evict_first_weights`.
    pub l2_evict_first: bool,
    /// `ffn_fp8_verify_gemv`.
    pub verify_gemv: bool,
    /// `rmsnorm_fp8_quant_fused`.
    pub rmsnorm_quant: bool,
    /// `ffn_fp8_decode_gemv`: multi-sequence decode on the batch-M GEMVs
    /// (`dense_ffn/fp8_decode_gemv.rs`). It reads the same handles as
    /// `verify_gemv`, so it primes the same lookups.
    pub decode_gemv: bool,
}

impl AdaFp8Rows {
    /// The process's resolved rows (`layers/ops/port_rows/*`).
    pub fn resolved() -> Self {
        use crate::layers::ops::port_rows;
        Self {
            l2_evict_first: port_rows::l2_evict_first_weights::row(),
            verify_gemv: port_rows::ffn_fp8_verify_gemv::row(),
            rmsnorm_quant: port_rows::rmsnorm_fp8_quant_fused::row(),
            decode_gemv: port_rows::ffn_fp8_decode_gemv::row(),
        }
    }
}

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

/// [`prefer`] with the role's L2 twin ahead of it when `twin` is `Some` (the
/// row is on). The twin is a `try_target_kernel`, so a target without the
/// module issues no lookup for it and lands on [`prefer`] unchanged.
fn prefer_hinted(gpu: &dyn GpuBackend, twin: Option<Entry>, pair: (Entry, Entry)) -> KernelHandle {
    if let Some((module, entry)) = twin {
        let h = crate::layers::try_target_kernel(gpu, module, entry);
        if h.0 != 0 {
            return h;
        }
    }
    prefer(gpu, pair)
}

impl Fp8FfnKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        // Row `ffn_fp8_m16_tc` has no layer field either: its two handles
        // (`dense_ffn/fp8_m16_tc.rs`) are looked up here, with the row on
        // only, for the reason [`prime_port_kernels`] gives.
        if crate::layers::ops::port_rows::ffn_fp8_m16_tc::row() {
            super::fp8_m16_tc::prime(gpu);
        }
        Self::resolve_with(gpu, AdaFp8Rows::resolved())
    }

    /// [`Self::resolve`] over explicit rows, so a CPU test can drive every
    /// polarity without sealing the process-wide row table.
    pub fn resolve_with(gpu: &dyn GpuBackend, rows: AdaFp8Rows) -> Self {
        let [gemv, dual, silu, gemm] = FP8_FFN_KERNELS;
        let twin = |role: usize| rows.l2_evict_first.then_some(FP8_FFN_L2EF_KERNELS[role]);
        let kernels = Self {
            gemv: prefer_hinted(gpu, twin(0), gemv),
            gemv_dual: prefer_hinted(gpu, twin(1), dual),
            gemv_silu_input: prefer_hinted(gpu, twin(2), silu),
            gemm_w8a8: prefer(gpu, gemm),
            swiglu_quant: crate::layers::try_target_kernel(
                gpu,
                SWIGLU_QUANT_KERNEL.0,
                SWIGLU_QUANT_KERNEL.1,
            ),
        };
        prime_port_kernels(gpu, rows);
        kernels
    }

    /// The M=1 decode GEMV role alone, for a layer that needs no other role
    /// (the GDN layer's `w8a16_gemv_k`): one lookup where the target has no
    /// drop-in, exactly the lookup that layer issued before. With
    /// `l2_evict_first_weights` on, the twin is tried first.
    pub fn m1_gemv(gpu: &dyn GpuBackend) -> KernelHandle {
        Self::m1_gemv_with(gpu, AdaFp8Rows::resolved())
    }

    /// [`Self::m1_gemv`] over explicit rows.
    pub fn m1_gemv_with(gpu: &dyn GpuBackend, rows: AdaFp8Rows) -> KernelHandle {
        let twin = rows.l2_evict_first.then_some(FP8_FFN_L2EF_KERNELS[0]);
        prefer_hinted(gpu, twin, FP8_FFN_KERNELS[0])
    }

    /// The W8A8 prefill GEMM role alone (the GDN layer's
    /// `fp8_gemm_t_blockscaled_k`), on the same terms as [`Self::m1_gemv`].
    pub fn w8a8_gemm(gpu: &dyn GpuBackend) -> KernelHandle {
        prefer(gpu, FP8_FFN_KERNELS[3])
    }
}

/// `module::entry` through the backend's op cache where the target compiled
/// the module, else `0` and no lookup. The first call per backend is the real
/// lookup (and its audit record); every later call is a map read.
pub fn cached_target_kernel(gpu: &dyn GpuBackend, (module, entry): Entry) -> KernelHandle {
    if !gpu.has_module(module) {
        return KernelHandle(0);
    }
    gpu.op_cache()
        .kernel(gpu, module, entry)
        .unwrap_or(KernelHandle(0))
}

/// The fused RMSNorm-quant handle on this backend, `0` where the target did
/// not compile it. Call only with `rmsnorm_fp8_quant_fused` on: that is when
/// construction primed it ([`Fp8FfnKernels::resolve`]).
pub fn rmsnorm_quant_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    cached_target_kernel(gpu, RMSNORM_QUANT_KERNEL)
}

/// Construction-time lookups of the row-gated kernels that have no layer
/// field, so every lookup stays eager. Nothing is looked up with the rows off.
fn prime_port_kernels(gpu: &dyn GpuBackend, rows: AdaFp8Rows) {
    if rows.verify_gemv || rows.decode_gemv {
        verify::prime(gpu, rows.l2_evict_first);
    }
    if rows.rmsnorm_quant {
        let _ = rmsnorm_quant_kernel(gpu);
    }
}

#[path = "dense_ffn_ada_kernels/verify.rs"]
pub(crate) mod verify;

#[cfg(test)]
#[path = "dense_ffn_ada_kernels_tests.rs"]
mod tests;
