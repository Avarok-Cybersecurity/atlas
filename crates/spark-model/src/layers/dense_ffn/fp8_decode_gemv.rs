// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: serve MULTI-SEQUENCE DECODE (2 to 8 streams) of a
//! native-FP8 dense FFN on the sm_89 batch-M GEMVs, behind the
//! `ffn_fp8_decode_gemv` row (off by default on every target).
//!
//! # What it replaces, and why
//!
//! A decode step over `n` streams runs each layer's dense FFN over `n` rows.
//! On an FP8 layer every small-batch entry point ends in `forward_prefill`:
//!
//! | streams | SSM (GDN) layers | attention layers | prefill-ladder kernel |
//! |---|---|---|---|
//! | 2, 3 | `forward_k2` / `forward_k3` | same | `w8a16_gemv_batch4` |
//! | 4 | `try_forward_km` | same | `w8a16_gemv_batch4` |
//! | 5..8 | `forward_prefill` DIRECTLY (`ssm_ffn_prefill_min_n`) | `try_forward_km` | the W8A8 arm |
//!
//! `w8a16_gemv_batch4` reads each weight once and is at the weight-bandwidth
//! floor. The W8A8 arm (`fp8_gemm_ada_w8a8`) is not: it runs a padded
//! tensor-core tile, costs the same at every row count, and that cost is about
//! 1.6x the floor on the RTX 4060 Ti 16 GB (L07 linear sweep, 4B shapes:
//! 141 to 176 us a projection at 1 to 32 rows against 90 us for
//! `fp8_gemv_ada` at one row). The measured decode step shows the same seam:
//! 27.0 / 28.2 / 30.5 / 40.0 ms at 1 / 2 / 4 / 8 streams, so the last four
//! streams cost 9.5 ms where the first three cost 3.5 ms. It also quantizes
//! the activations to E4M3, which no narrower batch does.
//!
//! The batch-M GEMVs (`fp8_gemv_ada_batchm.cu`) read each weight once for all
//! rows and are byte-identical, row for row, to the M=1 decode GEMVs. They
//! were written for the speculative verify (`ada_kernels/verify.rs`, row
//! `ffn_fp8_verify_gemv`); this file points the decode step at the same
//! launches.
//!
//! # What it buys in time (measured)
//!
//! Less than the table suggests. `fp8_ada_batchm_microtest` on the same card
//! (4B gate/up shape, warm, 2026-11-13): 104 us at M=1, 97 at M=4 and 184 at
//! M=8; the down shape 103, 94 and 157. The kernel holds the bandwidth floor
//! to four rows and is compute bound above them, so at eight streams a
//! projection costs about what the W8A8 arm costs (1.6 to 1.8 times M=1 in
//! both measurements) and the time gain is at five and six streams. What the
//! row changes at every width from 5 to 8 is the arithmetic (Numerics, below).
//!
//! # Scope
//!
//! Decode steps only: [`ForwardContext::decode_step`] is true on the
//! single-token and batched decode forwards and false on every prefill and
//! verify forward, so a verify pass is untouched by this row and a decode
//! pass is untouched by the verify row's absence.
//!
//! Two call sites reach it:
//!
//! * `DenseFfnLayer::fp8_verify_arm` (`w4.rs`), which `forward_k2`,
//!   `forward_k3` and `forward_km` already try first. That covers 2 to 4
//!   streams in both layer kinds and 5 to 8 in the attention layers.
//! * `Qwen3SsmLayer::decode_multi_seq_inner`'s wide arm, which calls
//!   `forward_prefill` without going through `forward_km`. It asks
//!   [`FfnComponent::try_forward_fp8_decode_rows`] first.
//!
//! # Numerics
//!
//! With the row on, every stream's FFN output is the single-stream decode
//! output for the same input, bit for bit, at every width. At 2 to 4 streams
//! that is what `w8a16_gemv_batch4` computes today. At 5 to 8 streams it
//! replaces the W8A8 arithmetic, so greedy output at those widths changes
//! against the row off, toward what the same prompts produce alone.
//!
//! [`FfnComponent::try_forward_fp8_decode_rows`]: crate::layers::FfnComponent::try_forward_fp8_decode_rows

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use super::ada_kernels::verify::{self, BATCHM_MODULE, MAX_ROWS, MIN_ROWS, RowKernels};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// The whole admission rule, as a pure function so a CPU test can pin every
/// clause without a backend:
///
/// * `row_on`: the `ffn_fp8_decode_gemv` row;
/// * `decode_step`: a decode forward, never a prefill or verify forward;
/// * `native_fp8` and not `packed_q2`: the layer holds the block-scaled FP8
///   overlay and no keep-packed Q2_0 weights (those have their own arm);
/// * `rows` in `MIN_ROWS..=MAX_ROWS`: one row is the M=1 decode forward, and
///   the kernels stop at eight.
pub(crate) fn admits(
    row_on: bool,
    decode_step: bool,
    native_fp8: bool,
    packed_q2: bool,
    rows: usize,
) -> bool {
    row_on && decode_step && native_fp8 && !packed_q2 && (MIN_ROWS..=MAX_ROWS).contains(&rows)
}

impl DenseFfnLayer {
    /// The decode FFN over `rows` streams on the batch-M GEMVs, into
    /// `moe_output` (where every small-batch entry point writes), or
    /// `Ok(false)` with nothing launched: row off, not a decode step, not a
    /// native-FP8 layer, a row count outside 2..=8, or the kernels absent.
    ///
    /// The structural clauses are tested before the row table is read, so a
    /// prefill or verify pass never touches it.
    pub(crate) fn fp8_decode_arm(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let structural = admits(
            true,
            ctx.decode_step,
            self.fp8_weights.is_some(),
            self.q2_weights.is_some(),
            rows,
        );
        if !structural || !ops::port_rows::ffn_fp8_decode_gemv::row() {
            return Ok(false);
        }
        let l2ef = ops::port_rows::l2_evict_first_weights::row();
        let Some(k) = RowKernels::cached(ctx.gpu, rows, l2ef) else {
            return Ok(false);
        };
        self.fp8_decode_arm_with(true, &k, input, rows, ctx, stream)
    }

    /// [`Self::fp8_decode_arm`] over an explicit row value and explicit
    /// handles, so a CPU test drives both polarities without sealing the
    /// process-wide row table.
    pub(crate) fn fp8_decode_arm_with(
        &self,
        row_on: bool,
        k: &RowKernels,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !admits(
            row_on,
            ctx.decode_step,
            self.fp8_weights.is_some(),
            self.q2_weights.is_some(),
            rows,
        ) {
            return Ok(false);
        }
        let out = ctx.buffers.moe_output();
        let Some(route) = verify::launch_km_fp8(self, k, input, rows, out, ctx, stream)? else {
            return Ok(false);
        };
        if ctx.stats.once("log:ffn_fp8_decode_gemv") {
            tracing::info!(
                "[avarok] dense FFN multi-sequence decode: FP8 batch-M GEMV ({BATCHM_MODULE}, \
                 2..8 streams, {route:?}); each stream byte-identical to the single-stream decode \
                 FFN. AVAROK_FFN_FP8_DECODE_GEMV=0 restores the prefill ladder (the W8A8 arm at \
                 5..8 streams)."
            );
        }
        Ok(true)
    }
}

impl crate::layers::FfnComponent {
    /// The decode FFN over `rows` streams on the FP8 batch-M GEMVs
    /// ([`DenseFfnLayer::fp8_decode_arm`]), for a caller that would otherwise
    /// go straight to `forward_prefill`. `Ok(false)` = nothing launched and
    /// the caller keeps its path: always, for an MoE or absent FFN.
    pub fn try_forward_fp8_decode_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) => d.fp8_decode_arm(input, rows, ctx, stream),
            _ => Ok(false),
        }
    }
}

#[cfg(test)]
#[path = "fp8_decode_gemv_tests.rs"]
mod tests;
