// SPDX-License-Identifier: AGPL-3.0-only

//! Which batched-GEMV kernel the chain-verify tier gets.
//!
//! Split out of `layers/mod.rs`, which is a module list and had grown a
//! several-hundred-line function with it. The provenance below is the whole
//! point of the function, so it travels with the code rather than being
//! summarised at the new call site.

use super::try_kernel;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Resolve the M<=8 batched-GEMV kernel for the chain-verify tier.
/// provenance-id: 526f6e616c6420522e205374657369616b
///
/// Prefers the register-tiled `w4a16_gemv_batch8_rt2` (T=2 output rows per
/// thread: one activation load feeds two FMA chains, halving activation
/// traffic, load-instruction count, and exposing 2x FMA-chain ILP).
/// batchm_bench 2026-08-19 @M=8: +17-26% GB/s on every verify shape
/// (qkv/o 143->168, ffn_up 149->182, ffn_down 150->186, gdn_qkvz 149->181,
/// lm_head 143->181), BIT-EXACT vs batch8 at all M (gate 4) — per-output
/// FMA order is identical, only the thread<->output mapping changed.
///
/// Launch geometry is IDENTICAL to batch8 (grid ceil(N/4), block 256; rt2's
/// surplus blocks early-exit on `n0 >= N`), so every call site, launcher,
/// and CUDA-graph capture is unchanged. `ATLAS_NO_BATCH8_RT=1` restores the
/// classic batch8 for A/B (strict `== "1"`, matching the sibling levers).
pub(crate) fn batch8_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("ATLAS_NO_BATCH8_RT").as_deref() != Ok("1") {
        let h = try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch8_rt2");
        if h.0 != 0 {
            return h;
        }
    }
    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch8")
}
