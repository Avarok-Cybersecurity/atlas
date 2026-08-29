// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-selection helpers for [`Qwen3SsmLayer`]: W4A16 batch-m GEMV tier
//! pick, deep-K transposed-tile GEMM pick, and the multi-seq decode GEMM
//! dispatch. Split from `mod.rs` for the ≤500 LoC cap; the selection rules
//! are shared by the decode/prefill sub-modules.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

impl Qwen3SsmLayer {
    /// W4A16 batchm-GEMV handle for `m` verify/decode rows: the narrowest
    /// resolved tier in `w4a16_gemv_batch{4,5,6,7,8}` that covers `m`.
    /// 0-handle when absent or out of range — callers must check `.0 != 0` and
    /// fall back to the tile GEMMs. See `layers::w4a16_gemv_tiers`.
    pub(super) fn w4a16_batchm_kernel(&self, m: usize) -> KernelHandle {
        self.w4a16_batchm.kernel(m as u32)
    }

    /// Transposed-twin tile GEMM handle for reduction depth `k`: the deep-K
    /// `_k64` variant when the shape qualifies, else the K_STEP_T=32 default.
    /// Same selection rule as the dense-FFN and attention-QKV paths, so all
    /// three consume `W4A16_K64_MIN_K` rather than repeating the threshold.
    pub(super) fn deep_k_gemm(&self, k: u32) -> KernelHandle {
        if k >= crate::layers::w4a16_k64_min_k()
            && k.is_multiple_of(64)
            && self.w4a16_gemm_t_k64_k.0 != 0
        {
            self.w4a16_gemm_t_k64_k
        } else {
            self.w4a16_gemm_t_k
        }
    }

    /// Transposed-twin tile GEMM for the multi-seq DECODE projections
    /// (QKVZ in, out_proj out), choosing the M-tile by batch width.
    ///
    /// `deep_k_gemm`'s kernels carry a 64-row M-tile, so a launch covers
    /// `ceil(m/64)` CTA ROWS and EVERY CTA row re-streams the whole
    /// transposed weight. At a 128-wide decode batch that is a second full
    /// pass over qkvz/out_proj — pure LPDDR5X traffic on a path already at
    /// the memory wall. `w4a16_gemm_t_m128` covers 128 rows per CTA, so the
    /// weight is read ONCE; it is the kernel the SSM PREFILL arm
    /// (`trait_prefill_proj`) and the dense-FFN prefill already call on
    /// these exact two weights, so no new kernel is introduced.
    ///
    /// The M-tile is NOT free, so a second condition applies: the wider tile
    /// halves the CTA count, and `ceil(N/128)` alone must already cover the
    /// machine or the "duplicate" CTA row was buying occupancy rather than
    /// wasting bandwidth. MEASURED both ways at n=128 (nsys, same-session
    /// kill-switch A/B):
    ///   QKVZ  N=16384 -> grid.x 128 >= 48 SMs: 27.73 -> 16.77 ms/step (-39%)
    ///   out_proj N=5120 -> grid.x 40 < 48 SMs: 10.28 -> 16.94 ms/step (+65%)
    /// At grid.x=40 the m64 tile's second CTA row is what fills the last 8 SMs
    /// (80 CTAs = 1.67 waves); the m128 tile leaves 40 CTAs = 0.83 of a wave.
    ///
    /// ADDITIVE: below `super::gdn_flags::ssm_m128_min_m()`, and at any N that does not fill the
    /// machine on its own, this is the identical `deep_k_gemm` launch the path
    /// has always made.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_proj_gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(min_m) = super::gdn_flags::ssm_m128_min_m()
            && m >= min_m
            && n.div_ceil(128) >= self.sm_count
            && self.w4a16_gemm_t_m128_k.0 != 0
        {
            return ops::w4a16_gemm_n128_m128(
                gpu,
                self.w4a16_gemm_t_m128_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            );
        }
        let wide = self.deep_k_gemm(k);
        // Narrow-N deep-K twin: bit-identical, and 1.42x at the out_proj shape
        // (N=5120, K=6144 -> 40 CTAs on 48 SMs). See `layers::k64_n64_wins`.
        if wide.0 == self.w4a16_gemm_t_k64_k.0
            && self.w4a16_gemm_t_k64_n64_k.0 != 0
            && crate::layers::k64_n64_wins(m, n)
        {
            return ops::w4a16_gemm(
                gpu,
                self.w4a16_gemm_t_k64_n64_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            );
        }
        ops::w4a16_gemm_n128(gpu, wide, input, weight, output, m, n, k, stream)
    }
}
