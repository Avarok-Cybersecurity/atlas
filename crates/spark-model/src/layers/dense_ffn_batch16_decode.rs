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
//! rungs 2-3 ahead of it the W8A8 path begins at **m > 32** in practice. 5..=32
//! are decode widths where one weight pass beats any MMA tile — but a prefill
//! of 5..=32 tokens (a very short prompt, or the TAIL CHUNK of a chunked
//! prefill) takes the GEMV too. That consequence is what the serving A/B below
//! caught, and it is why this tier ships disarmed.
//!
//! 🔴 DEFAULT OFF — OPT-IN VIA `ATLAS_FFN_BATCH16=1`. The cliff above is
//! real and this kernel is the right shape for it, but on the one target where
//! the tier has been A/B'd end to end it is a net LOSS in serving. H100 round
//! 5, 2026-09-11, Qwen/Qwen3.8-27B-FP8, single variable — same binary, same
//! 16-way burst, the tier the only difference:
//!
//! | 1024x256, C=16      | tier ON  | tier OFF     |
//! |---------------------|----------|--------------|
//! | aggregate tok/s     | 121.4    | **128.0**    |
//! | TPOT p50            | 107.4 ms | **102.0 ms** |
//! | 28-token smoke TTFT | 150 ms   | **101 ms**   |
//!
//! Per-phase at n=16 (`ATLAS_MS_PROFILE=1`, so eager — read the ratios): with
//! the tier OFF the step goes 86.80 -> 82.32 ms, `ssm` 63.31 -> 59.91 ms and
//! `attn` 19.90 -> 18.82 ms (-5.2% to -5.4% each); `head` does not move. `ssm`
//! per layer returns to 1248 us against a pre-#927 1252 us — the tier's cost
//! is the whole of the regression it introduced, not part of it.
//!
//! WHY it loses although the kernel wins at 16 rows: it dispatches by ROW
//! COUNT, not by phase, so a chunked prefill's tail chunk lands in the band —
//! a 1193-token prompt splits `1168 + 25`, and the 25-row tail takes the GEMV.
//! That is a FIXED ~35 ms TTFT cost per request (49 ms on the 28-token smoke),
//! which no decode-rate gain at these widths pays back.
//!
//! 🚨 AND IT HAS NEVER BEEN MEASURED ON GB10. `w8a16_gemv_batch16` is an
//! instantiation in `w8a16_gemv_batch4.cu`, so the handle resolves on every
//! target that carries that module — GB10 included. A default-ON tier would
//! ship an unmeasured routing change to the target this repo serves, on the
//! strength of an H100 number that came out negative. Opt-in is the honest
//! default until a GB10 A/B exists; if one wins there, the lever to flip is
//! this file's, not the caller's.
//!
//! WHAT STAYS DEFAULT-ON, and why it is a different lever: the attention
//! `o_proj` groups-of-16 arm, the QKV band widening and the SSM MTP-verify
//! arms key off their OWN kernel handles and never read this switch. They were
//! ON in BOTH arms of the A/B above, so none of the movement in that table is
//! theirs to claim or to blame — including the `attn` phase's -5.4%, which
//! moved while they were untouched. Each is bit-identical per row to the M=1
//! `w8a16_gemv` it replaces, which is a numerics improvement that does not
//! depend on the FFN result either way.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// `ATLAS_FFN_BATCH16` opt-in: the value `1` — and only `1` — arms the
/// 5..=32-row tier. Anything else, absence included, leaves those widths on
/// the pre-#927 arms.
///
/// VALUE rather than the house PRESENCE convention (`ffn_w8a16_only` next
/// door) because the polarity is the other way round: for a switch that ARMS
/// an arm, it is `ATLAS_FFN_BATCH16=0` meaning "on" that would be the trap.
/// Same shape as `moe_grouped_decode_forced` in `layers/mod.rs`, the other
/// lever in this crate that arms rather than disarms.
///
/// `OnceLock`-cached and read ONCE PER LAYER, into `DenseFfnLayer`'s
/// `batch16_enabled`: the route must be CONSTANT across CUDA-graph replays,
/// and `std::env::var` walks the environment block on every call.
pub fn ffn_batch16_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_FFN_BATCH16").as_deref() == Ok("1"))
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
/// handle's presence and the opt-in.
///
/// Split out from the layer for the same reason `w8a8_prefill_selected` is:
/// the CPU tests pin every rung without building a `ForwardContext`, and
/// `enabled` is injected because a process-global `OnceLock` cannot be toggled
/// per test. `enabled` reads FIRST in the guard: it is the default-off gate,
/// and a reader asking "what does a stock serve do at m=8" should meet it
/// before anything about handles.
pub(crate) fn batch16_plan(m: u32, batch16_loaded: bool, enabled: bool) -> Option<Batch16Plan> {
    if !enabled || !batch16_loaded {
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
    /// The plan for `m` rows on THIS layer — handle presence plus the opt-in
    /// the layer latched at construction.
    pub(crate) fn ffn_batch16_plan(&self, m: u32) -> Option<Batch16Plan> {
        batch16_plan(m, self.w8a16_gemv_batch16_k.0 != 0, self.batch16_enabled)
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
    /// shape the other dense-FFN route logs use. It earns its line for a
    /// reason the default-on version did not have: this arm now runs only
    /// because an operator asked for it, and a serve quoting a 5..=32-row TPOT
    /// number should be able to prove from its own log which side of the A/B
    /// it ran.
    fn log_batch16_decode_route(&self, ctx: &ForwardContext, plan: Batch16Plan) {
        if ctx.stats.once("log:ffn_batch16_decode") {
            let how = match plan {
                Batch16Plan::Single => "one launch",
                Batch16Plan::Halves { .. } => "two launches on contiguous row halves",
            };
            tracing::info!(
                "[atlas] dense FFN decode: native FP8 w8a16_gemv_batch16 ({how}) \
                 for 5..=32 rows — one weight pass, bit-identical per row to the \
                 M=1 w8a16_gemv. ARMED BY ATLAS_FFN_BATCH16=1, off by default: it \
                 measured -5.4% aggregate and +50 ms TTFT on H100, and has never \
                 been measured on GB10 (#927)."
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_batch16_decode_tests.rs"]
mod tests;
