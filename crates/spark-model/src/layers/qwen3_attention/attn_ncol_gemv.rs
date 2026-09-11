// SPDX-License-Identifier: AGPL-3.0-only

//! The N-column-blocked W8A16 decode tier for the ATTENTION projections —
//! `w8a16_gemv_batch16_ncol{2,4}`, behind `ATLAS_ATTN_NCOL_GEMV` (#927).
//!
//! WHY. Measured on 1xH100, 2026-09-11, Qwen/Qwen3.8-27B-FP8, serve H
//! (`ATLAS_FFN_NO_BATCH16=1`, graphs off, batch 16): the step is 82.3 ms, of
//! which the 16 attention layers are 18.8 ms = **1177 µs/layer**. The full
//! split is `ATTN-DECODE-ATTRIBUTION.md`; the part this module acts on:
//!
//! | attention-layer sub-phase at n=16 | µs/layer | kernel |
//! |---|---:|---|
//! | dense FFN inside the layer | ~779 | (the FFN ladder's, not ours) |
//! | Q/K/V projections (73.4 MB of weight) | ~215 | `w8a16_gemv_batch16_strided` |
//! | o_proj (31.5 MB) | ~105 | `w8a16_gemv_batch16` |
//! | RoPE + q/k norm + KV write + paged decode + gate + residual | ~78 | already one launch each |
//!
//! So the projections are 27% of the phase and they are NOT bandwidth bound.
//! #927's receipt for the same kernel reads 342 GB/s against ~3,000 GB/s of
//! HBM3, because `w8a16_gemv_batchm_impl<16>` gives each thread ONE output
//! column and therefore pays, per 16 weight bytes, 32 `uint4` activation loads
//! and 256 BF16->FP32 converts on top of the 256 FFMA that are the work —
//! ~36 ALU ops per weight byte. Every other column's thread loads and converts
//! the SAME activations. `N_COLS` adjacent columns per thread amortise both:
//! ~36 -> ~27 ops/byte at N_COLS=2, ~23 at N_COLS=4, floor 18.
//!
//! 🟢 BIT-EXACT, and that is the whole reason this tier exists next to
//! `ATLAS_FFN_M16_TC`. The lane -> k16 map, the per-accumulator operand order
//! and the reduction tree are the single-column kernel's, element for element
//! (argument in `kernels/gb10/common/w8a16_gemv_ncol.cu`); the convert is
//! hoisted out of the column loop and a BF16->FP32 widening is exact. So each
//! row stays bit-identical to the scalar `w8a16_gemv` that M=1 decode runs,
//! exactly as `w8a16_gemv_batch{4,16}` are. `w8a16_gemm_m16` buys more and
//! REASSOCIATES (<= 2 BF16 ULP); this one buys less and opens no seam.
//!
//! ⚠ UNMEASURED ON DEVICE, hence OPT-IN. The ops-per-byte arithmetic above is
//! a hypothesis until an H100 receipt exists, and it is bought with registers.
//! `ptxas -v -arch=sm_90a` (CUDA 13.0, `--fmad=false`, same flags as the
//! build), all four entry points, 0 spill bytes:
//!
//! | kernel | registers | smem | blocks/SM at 256 threads |
//! |---|---:|---:|---:|
//! | `w8a16_gemv_batch16` (today) | 64 | 1 KB | 4 (32 warps) |
//! | `..._ncol2[_strided]` | 128 | 2 KB | 2 (16 warps) |
//! | `..._ncol4[_strided]` | 214 | 3 KB | 1 (8 warps) |
//!
//! So the tier trades occupancy for ILP, which is exactly the trade a device
//! measurement has to settle: N_COLS=2 halves the resident warps, N_COLS=4
//! quarters them. That is why 2 is the default width and why nothing flips on
//! its own. The receipt is
//! `examples/native_fp8_attn_decode_batch_microtest`, which asserts
//! bit-identity and times every route at n in {2,4,8,16}.
//!
//! GRAPH-CAPTURE. Both call sites pass the ctx `n`, which IS `padded_n` (the
//! ladder in `traits::model::padded_batch_n`, which the graph cache is keyed
//! by), so branching on it bakes exactly the value the graph is keyed by — the
//! same contract the n==2 / n==3 NVFP4 branches and the FP8 QKV tier rely on.
//! Never branch on the unpadded `seqs.len()`. The lever itself is resolved
//! ONCE into a layer field at construction, so a replay cannot take a
//! different route than the capture did.
//!
//! BAND 5..=16, deliberately the band `w8a16_gemv_batch16` already owns. The
//! ALU wall is proportional to M: at m <= 4 `w8a16_gemv_batch4` pays ~10
//! ops/byte and is already near the bandwidth floor, so widening the band
//! would swap a route that is not the problem for a MAX_M=16 register array.
//! 17+ is above the kernel's MAX_M, which CLAMPS rather than erroring.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3AttentionLayer;
use crate::layers::ops;

/// Output columns one thread owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcolWidth {
    Two,
    Four,
}

impl NcolWidth {
    /// The `N_COLS` template argument, for logging and the grid divisor.
    pub(crate) fn cols(self) -> u32 {
        match self {
            NcolWidth::Two => 2,
            NcolWidth::Four => 4,
        }
    }
}

/// Whether the N-column-blocked tier claims the decode attention projections
/// on THIS target.
///
/// The compiled target declares it (`kernels/<hw>/HARDWARE.toml` `[defaults]
/// attn_ncol_gemv`) and `ATLAS_ATTN_NCOL_GEMV` overrides. BOTH current targets
/// declare it OFF: `w8a16_gemv_ncol.cu` is Hopper-tuned and is not in GB10's
/// kernel set, and on H100 the microtest exists but the serving A/B does not —
/// a default is a claim about a measurement.
///
/// `ATLAS_NO_ATTN_DECODE_BATCH` (PRESENCE, unchanged) wins over both and forces
/// the tier off: the operator escape hatch that has to stay meaningful once a
/// target can declare the tier on.
///
/// Resolution and caching are `layers::ops::target_defaults::resolved`, which
/// is `OnceLock`-backed for the reason every hot-path lever here is: the
/// selector runs per projection per layer per step, and a per-call `var_os`
/// could change the captured launch set across CUDA-graph replays.
pub fn ncol_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::layers::ops::target_defaults::resolved()
            .attn_ncol_gemv
            .value
            && std::env::var_os("ATLAS_NO_ATTN_DECODE_BATCH").is_none()
    })
}

/// `ATLAS_ATTN_NCOL_WIDTH=4` picks the 4-column instantiation; anything else
/// (including unset) keeps 2, the conservative register choice. A separate
/// variable from the on/off lever on purpose: folding the width into the
/// presence lever would make `ATLAS_ATTN_NCOL_GEMV=0` mean "on, 2 columns".
pub fn ncol_gemv_width() -> NcolWidth {
    static W: std::sync::OnceLock<NcolWidth> = std::sync::OnceLock::new();
    *W.get_or_init(|| match std::env::var("ATLAS_ATTN_NCOL_WIDTH").as_deref() {
        Ok("4") => NcolWidth::Four,
        _ => NcolWidth::Two,
    })
}

/// The whole selection rule, as a pure function of the row count, the resolved
/// width, the two handles' presence and the lever.
///
/// Pure and injected for the same reason `batch16_plan` is: the CPU tests pin
/// every edge without a `ForwardContext`, and a process-global `OnceLock`
/// cannot be toggled per test.
///
/// Returns the width to launch with, or `None` to leave the caller on
/// `w8a16_gemv_batch16` (or whatever rung it would otherwise take).
pub(crate) fn ncol_plan(
    m: usize,
    width: NcolWidth,
    ncol2_loaded: bool,
    ncol4_loaded: bool,
    enabled: bool,
) -> Option<NcolWidth> {
    if !enabled || !(5..=16).contains(&m) {
        return None;
    }
    // The chosen width must be the one the shadow actually carries; a missing
    // entry point declines rather than silently substituting the other width,
    // so a route line and an A/B always describe the kernel that ran.
    let loaded = match width {
        NcolWidth::Two => ncol2_loaded,
        NcolWidth::Four => ncol4_loaded,
    };
    loaded.then_some(width)
}

/// One line per process, the first time a projection takes the tier — the same
/// contract as the FFN tiers' route lines, so a serve log says which kernel the
/// decode projections actually ran.
pub(crate) fn log_route_once(width: NcolWidth, site: &str) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            "ATLAS_ATTN_NCOL_GEMV: decode attention projections via \
             w8a16_gemv_batch16_ncol{} (bit-exact; first site: {site})",
            width.cols(),
        );
    });
}

/// The shape of `ops::w8a16_gemv_batch{4,16}_strided` and the `_ncol*_strided`
/// entry points — the same alias `qkv_fp8_batch.rs` holds, so a tier swap is a
/// function pointer, not a second call site.
pub(crate) type StridedBatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

impl Qwen3AttentionLayer {
    /// The strided rung for the multi-seq Q/K/V projections, or `None` to stay
    /// on `w8a16_gemv_batch16_strided`. Logs the route the first time it fires.
    pub(super) fn ncol_strided_route(&self, m: usize) -> Option<(StridedBatchGemv, KernelHandle)> {
        let width = ncol_plan(
            m,
            self.attn_ncol?,
            self.w8a16_gemv_ncol2_strided_k.0 != 0,
            self.w8a16_gemv_ncol4_strided_k.0 != 0,
            true,
        )?;
        log_route_once(width, "multi-seq QKV");
        Some(match width {
            NcolWidth::Two => (
                ops::w8a16_gemv_batch16_ncol2_strided as StridedBatchGemv,
                self.w8a16_gemv_ncol2_strided_k,
            ),
            NcolWidth::Four => (
                ops::w8a16_gemv_batch16_ncol4_strided as StridedBatchGemv,
                self.w8a16_gemv_ncol4_strided_k,
            ),
        })
    }

    /// The contiguous rung for the FP8 o_proj, or `None` to stay on
    /// `w8a16_gemv_batch16`. Both rungs walk the batch in 16-row groups, so the
    /// caller's `step` is unchanged.
    pub(super) fn ncol_contiguous_route(
        &self,
        m: usize,
    ) -> Option<(ops::ContiguousBatchGemv, KernelHandle)> {
        let width = ncol_plan(
            m,
            self.attn_ncol?,
            self.w8a16_gemv_ncol2_k.0 != 0,
            self.w8a16_gemv_ncol4_k.0 != 0,
            true,
        )?;
        log_route_once(width, "o_proj");
        Some(match width {
            NcolWidth::Two => (
                ops::w8a16_gemv_batch16_ncol2 as ops::ContiguousBatchGemv,
                self.w8a16_gemv_ncol2_k,
            ),
            NcolWidth::Four => (
                ops::w8a16_gemv_batch16_ncol4 as ops::ContiguousBatchGemv,
                self.w8a16_gemv_ncol4_k,
            ),
        })
    }
}

#[cfg(test)]
#[path = "attn_ncol_gemv_tests.rs"]
mod tests;
