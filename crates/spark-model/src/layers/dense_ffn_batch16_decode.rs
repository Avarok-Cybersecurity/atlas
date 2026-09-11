// SPDX-License-Identifier: AGPL-3.0-only

//! The 5..=32-row native-FP8 dense-FFN DECODE tier — `w8a16_gemv_batch16`.
//!
//! WHY (#927). Measured on 1xH100, 2026-09-11, Qwen/Qwen3.8-27B-FP8, tip
//! `fbbe70767`: the decode step cost 44 ms at 4 active rows and **224 ms at
//! 16** (TPOT), so raising the batch cap from 4 to 16 made C=16 aggregate
//! throughput FALL from 76 to 62 tok/s. Five extra rows cost 5x the step.
//!
//! The cliff is a dispatch gap, not a kernel one. `dense_ffn.rs`'s `w8_gemm!`
//! claimed only `(1..=4)` for `w8a16_gemv_batch4`; at m = 5..16 it fell to the
//! transposed `w8a16_gemm_n128_m128` / `w8a16_gemm_pipelined` tile GEMMs. Those
//! pad M to a 128-row MMA tile, so at M=16 seven eighths of every tile is
//! padding and the kernel turns 5-12 TFLOP/s while the FFN at decode widths is
//! purely weight-bandwidth bound. `w8a16_gemv_batch16` — the MAX_M=16
//! instantiation of the SAME template as `w8a16_gemv_batch4`, already in
//! `kernels/gb10/common/w8a16_gemv_batch4.cu` — makes ONE pass over the FP8
//! weight for up to 16 rows.
//!
//! NUMERICS. Every row is bit-identical to the scalar `w8a16_gemv` that M=1
//! decode runs: same K-iteration order, same per-row reduction tree, the
//! accumulators are independent and `M` appears in no row's operand sequence.
//! H100 receipt on #932: M=8 and M=16 both `unequal_bf16=0 max_abs=0`. So this
//! moves widths 5..=32 from a REASSOCIATING tile GEMM onto the bits decode
//! already produces at M=1 — the direction that removes a numerics seam rather
//! than adding one.
//!
//! 17..=32 runs the same kernel TWICE on contiguous row halves. The FFN
//! activations and outputs are contiguous `[m, k]` / `[m, n]`, so a half is a
//! plain byte offset — no staging, no strided variant. Two weight passes still
//! beat one M-padded MMA tile at these widths, and it means a `max_batch_size`
//! of 32 never reaches the tile GEMMs at decode either.
//!
//! ARM ORDER in `w8_gemm!` (see `dense_ffn.rs`) is deliberate and this module
//! owns the 2nd and 3rd rungs:
//!   1. `m <= 4`     -> `w8a16_gemv_batch4`
//!   2. `m` 5..=16   -> `w8a16_gemv_batch16`            (here)
//!   3. `m` 17..=32  -> `w8a16_gemv_batch16` x2 halves  (here)
//!   4. W8A8 block-scaled prefill (#917/#928)
//!   5. transposed / pipelined / base W8A16 tile GEMMs
//!
//! 🪤 CONSEQUENCE, stated because it is a real boundary move: the W8A8 prefill
//! arm's own rule (`dense_ffn_w8a8_prefill.rs`) starts at `m > 4`, so with
//! rungs 2-3 ahead of it the W8A8 path now begins at **m > 32** in practice.
//! That is the intent — 5..=32 are decode widths where one weight pass beats
//! any MMA tile — but a prefill of 5..=32 tokens (a very short prompt, or the
//! tail chunk of a chunked prefill) now takes the GEMV too. Set
//! `ATLAS_FFN_NO_BATCH16` to restore the previous routing for those widths.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// `ATLAS_FFN_NO_BATCH16` kill switch: PRESENCE (any value, including empty)
/// sends 5..=32 rows back to the pre-#927 arms.
///
/// Presence rather than `=1`, matching `ffn_w8a16_only` next door: this is an
/// escape hatch an operator reaches for while a serve misbehaves, and
/// `ATLAS_FFN_NO_BATCH16=0` meaning "batch16 is off" is a trap.
///
/// `OnceLock`-cached: the selector runs per projection per layer per step and
/// `std::env::var_os` walks the environment block on every call. Cached
/// process-wide is also what keeps the route CONSTANT across CUDA-graph
/// replays — a per-call read could change the captured launch set.
pub fn ffn_no_batch16() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("ATLAS_FFN_NO_BATCH16").is_some())
}

/// How the batch16 tier serves `m` rows, or `None` when it does not claim them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Batch16Plan {
    /// One launch covering rows `0..m` (m <= 16).
    Single,
    /// Two launches on contiguous row halves: rows `0..first`, then
    /// `first..m`. `first` is `ceil(m/2)`, so both halves are <= 16 for every
    /// m <= 32 and the FIRST half is the wider one (m=17 -> 9 + 8).
    Halves { first: u32 },
}

/// The whole batch16 selection rule, as a pure function of the row count, the
/// handle's presence and the kill switch.
///
/// Split out from the layer for the same reason `w8a8_prefill_selected` is:
/// the CPU tests pin every rung without building a `ForwardContext`, and
/// `disabled` is injected because a process-global `OnceLock` cannot be
/// toggled per test.
pub(crate) fn batch16_plan(m: u32, batch16_loaded: bool, disabled: bool) -> Option<Batch16Plan> {
    if disabled || !batch16_loaded {
        return None;
    }
    match m {
        5..=16 => Some(Batch16Plan::Single),
        // Both halves must be <= 16, the kernel's MAX_M. `div_ceil` puts the
        // odd row in the first half; either order is bit-identical per row.
        17..=32 => Some(Batch16Plan::Halves {
            first: m.div_ceil(2),
        }),
        _ => None,
    }
}

impl DenseFfnLayer {
    /// The plan for `m` rows on THIS layer — handle presence plus the switch.
    pub(crate) fn ffn_batch16_plan(&self, m: u32) -> Option<Batch16Plan> {
        batch16_plan(m, self.w8a16_gemv_batch16_k.0 != 0, ffn_no_batch16())
    }

    /// Run one dense-FFN projection through `w8a16_gemv_batch16`.
    ///
    /// `input` is `[m, k]` BF16 and `out` is `[m, n]` BF16, both CONTIGUOUS —
    /// which is what makes the `Halves` plan a pair of byte offsets rather
    /// than a strided launch. (`ops::w8a16_gemv_batch16_strided` is the tool
    /// when a caller's rows are NOT contiguous; the attention QKV path uses
    /// it.)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a16_batch16_proj(
        &self,
        ctx: &ForwardContext,
        plan: Batch16Plan,
        w: &Fp8Weight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        self.log_batch16_decode_route(ctx, plan);
        const BF16: usize = 2;
        let launch = |rows: u32, first: u32| {
            ops::w8a16_gemv_batch16(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
                input.offset(first as usize * k as usize * BF16),
                w.weight,
                w.row_scale,
                out.offset(first as usize * n as usize * BF16),
                rows,
                n,
                k,
                stream,
            )
        };
        match plan {
            Batch16Plan::Single => launch(m, 0),
            Batch16Plan::Halves { first } => {
                launch(first, 0)?;
                launch(m - first, first)
            }
        }
    }

    /// Log-once latch for the batch16 decode tier, in the same `log:ffn_*`
    /// shape the other dense-FFN route logs use. It is worth a line: this arm
    /// is what a 5..=32-row TPOT report is measuring, and its absence at a
    /// width that should have it is the first thing to check when the #927
    /// cliff appears to be back.
    fn log_batch16_decode_route(&self, ctx: &ForwardContext, plan: Batch16Plan) {
        if ctx.stats.once("log:ffn_batch16_decode") {
            let how = match plan {
                Batch16Plan::Single => "one launch",
                Batch16Plan::Halves { .. } => "two launches on contiguous row halves",
            };
            tracing::info!(
                "[atlas] dense FFN decode: native FP8 w8a16_gemv_batch16 ({how}) \
                 for 5..=32 rows — one weight pass, bit-identical per row to the \
                 M=1 w8a16_gemv. ATLAS_FFN_NO_BATCH16 restores the tile GEMMs (#927)."
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_batch16_decode_tests.rs"]
mod tests;
