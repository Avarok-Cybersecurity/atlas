// SPDX-License-Identifier: AGPL-3.0-only

//! The 5..=32-row native-FP8 dense-FFN decode tier on TENSOR CORES —
//! `w8a16_gemm_m16`, behind `ATLAS_FFN_M16_TC` (#927).
//!
//! WHY. Measured on 1xH100, 2026-09-11, Qwen/Qwen3.8-27B-FP8, tip
//! `2962cfed7`: at a decode batch of 16 the step is **86.7 ms**, of which the
//! 48 SSM layers are 63.3 ms and the dense FFN inside them is **63%**
//! (~833 us/layer). The tier that serves those widths today,
//! `w8a16_gemv_batch16` (rungs 2-3 of `dense_ffn_batch16_decode.rs`), is
//! bit-exact but **FP32-FMA-bound** at M=16, not bandwidth-bound:
//!
//! | shape | batch16 GEMV @ M=16 | HBM3 |
//! |---|---|---|
//! | gate/up N=17408 K=5120 | 0.260 ms / **342 GB/s** | ~3,000 GB/s |
//! | down    N=5120 K=17408 | 0.330 ms / **270 GB/s** | ~3,000 GB/s |
//!
//! An 89 MB FP8 weight matrix should stream in ~30 us. The GEMV spends ~37 ALU
//! ops per weight BYTE (16 scalar FFMA across the 16 rows, 16 BF16->FP32
//! converts, a LUT lookup, a scale multiply), which caps it near 350 GB/s no
//! matter how fast the DRAM is. `w8a16_gemm_m16` replaces those 16 FFMA with
//! one `mma.sync.m16n8k16` lane-slot — the M tile IS 16 rows, so nothing is
//! padded, which is the whole difference from the tile GEMMs that pad M to 128
//! and waste 7/8 of every tile — and cuts the dequant to ~2 instructions per
//! byte. Target: >= 1,500 GB/s at M=16, >= 1,000 GB/s at M=8.
//!
//! 🔴 NUMERICS — THIS ARM REASSOCIATES; THE BATCH16 ARM DOES NOT.
//! `w8a16_gemv_batch16` reduces each output in ONE FP32 accumulator walked in
//! strict K order, which makes it bit-identical to the scalar `w8a16_gemv` that
//! M=1 decode runs. An MMA reduces 16 K-products in the tensor core's own order
//! first, so THIS arm is not. Its contract is <= 2 BF16 ULP per element, with
//! the 128-K block scale still folded once per block onto an FP32 outer
//! accumulator (the two-level fold, preserved exactly). That is a seam, and it
//! is why the lever exists and defaults OFF.
//!
//! It is not a NEW seam, though: the arm the FFN reached at these widths BEFORE
//! #927 was `w8a16_gemm_n128_m128` / `w8a16_gemm_pipelined`, both m16n8k16 MMA
//! kernels with exactly this reassociation. Turning the lever on returns 5..=32
//! to MMA numerics while keeping the ONE-weight-pass property #927 bought.
//!
//! ARM ORDER in `w8_gemm!` (`dense_ffn.rs`) with the lever ON:
//!   1. `m <= 4`     -> `w8a16_gemv_batch4`                (bit-exact)
//!   2. `m` 5..=16   -> `w8a16_gemm_m16`                   (here, MMA)
//!   3. `m` 17..=32  -> `w8a16_gemm_m16` x2 halves         (here, MMA)
//!   4. `m` 5..=16   -> `w8a16_gemv_batch16`               (bit-exact)
//!   5. `m` 17..=32  -> `w8a16_gemv_batch16` x2 halves     (bit-exact)
//!   6. W8A8 block-scaled prefill (#917/#928)
//!   7. transposed / pipelined / base W8A16 tile GEMMs
//! With the lever OFF (the default) rungs 2-3 vanish and the ladder is exactly
//! what #927 shipped.
//!
//! 17..=32 runs the kernel TWICE on contiguous row halves, for the same reason
//! `batch16_decode.rs` does: the FFN activations and outputs are contiguous
//! `[m, k]` / `[m, n]`, so a half is a plain byte offset, and two weight passes
//! still beat one M-padded MMA tile at these widths.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// `ATLAS_FFN_M16_TC`: PRESENCE (any value, including empty) turns the
/// tensor-core tier ON. SSOT for the lever across ALL of its call sites — the
/// dense FFN here, plus the multi-seq FP8 QKV tier (`qkv_fp8_batch.rs`) and the
/// FP8 o_proj tier (`attn/o_proj.rs`), which read it through this function so
/// one `ATLAS_FFN_M16_TC=1` A/Bs the whole route rather than three. Default OFF — the opposite polarity to
/// `ATLAS_FFN_NO_BATCH16` next door, and deliberately so: that one is an
/// operator's escape hatch from a shipped default, this one is an opt-in to a
/// route that trades the bit-exactness of #927 for bandwidth, and it stays off
/// until an H100 receipt says it wins. Presence rather than `=1` keeps the A/B
/// recipe a single `ATLAS_FFN_M16_TC=1` prefix with no "=0 means on" trap.
///
/// `OnceLock`-cached for the same reason the batch16 switch is: the selector
/// runs per projection per layer per step, and a per-call `var_os` could change
/// the captured launch set across CUDA-graph replays.
pub fn m16_tc_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_FFN_M16_TC").is_some())
}

/// How the tensor-core tier serves `m` rows, or `None` when it does not claim
/// them. Mirrors `Batch16Plan` so the two ladders read the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum M16TcPlan {
    /// One launch covering rows `0..m` (m <= 16 = the kernel's M tile).
    Single,
    /// Two launches on contiguous row halves: rows `0..first`, then
    /// `first..m`. `first` is `ceil(m/2)`, so both halves are <= 16 for every
    /// m <= 32 and the FIRST half is the wider one (m=17 -> 9 + 8).
    Halves { first: u32 },
}

/// The whole selection rule, as a pure function of the row count, the reduction
/// depth, the handle's presence and the lever.
///
/// `k` is part of the rule and not an `ensure!` at the call site: the kernel
/// indexes `block_scale[n_block * (K/128) + k/128]`, so a K that is not a whole
/// number of 128-wide scale blocks has no correct scale to fold and the tier
/// must DECLINE rather than launch and be wrong. Every Atlas FP8 FFN shape
/// satisfies it (Qwen3.8-27B: 5120 and 17408), but a model whose hidden size is
/// not a multiple of 128 would otherwise fall off this cliff silently.
pub(crate) fn m16_tc_plan(m: u32, k: u32, loaded: bool, enabled: bool) -> Option<M16TcPlan> {
    if !enabled || !loaded || !k.is_multiple_of(128) {
        return None;
    }
    match m {
        5..=16 => Some(M16TcPlan::Single),
        // Both halves must be <= 16, the kernel's M tile. `div_ceil` puts the
        // odd row in the first half; the split changes no row's arithmetic.
        17..=32 => Some(M16TcPlan::Halves {
            first: m.div_ceil(2),
        }),
        _ => None,
    }
}

impl DenseFfnLayer {
    /// The plan for `m` rows at reduction depth `k` on THIS layer — handle
    /// presence plus the lever.
    pub(crate) fn ffn_m16_tc_plan(&self, m: u32, k: u32) -> Option<M16TcPlan> {
        m16_tc_plan(m, k, self.w8a16_gemm_m16_k.0 != 0, self.m16_tc)
    }

    /// Run one dense-FFN projection through `w8a16_gemm_m16`.
    ///
    /// `input` is `[m, k]` BF16 and `out` is `[m, n]` BF16, both CONTIGUOUS,
    /// which is what makes the `Halves` plan a pair of byte offsets rather than
    /// a strided launch. (`ops::w8a16_gemm_m16_strided` is the tool when a
    /// caller's rows are not contiguous; the attention QKV path uses it.)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a16_m16_tc_proj(
        &self,
        ctx: &ForwardContext,
        plan: M16TcPlan,
        w: &Fp8Weight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        self.log_m16_tc_route(ctx, plan);
        const BF16: usize = 2;
        let launch = |rows: u32, first: u32| {
            ops::w8a16_gemm_m16(
                ctx.gpu,
                self.w8a16_gemm_m16_k,
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
            M16TcPlan::Single => launch(m, 0),
            M16TcPlan::Halves { first } => {
                launch(first, 0)?;
                launch(m - first, first)
            }
        }
    }

    /// Log-once latch, in the same `log:ffn_*` shape the other dense-FFN route
    /// logs use. It is worth a line because this arm is the one that is NOT
    /// bit-identical to the M=1 decode path: a TPOT report or a parity
    /// complaint at 5..=32 rows needs to say which of the two tiers ran.
    fn log_m16_tc_route(&self, ctx: &ForwardContext, plan: M16TcPlan) {
        if ctx.stats.once("log:ffn_m16_tc_decode") {
            let how = match plan {
                M16TcPlan::Single => "one launch",
                M16TcPlan::Halves { .. } => "two launches on contiguous row halves",
            };
            tracing::info!(
                "[atlas] dense FFN decode: ATLAS_FFN_M16_TC — tensor-core w8a16_gemm_m16 \
                 ({how}) for 5..=32 rows, ahead of w8a16_gemv_batch16. One weight pass, \
                 m16n8k16 MMA, so outputs are REASSOCIATED vs the scalar w8a16_gemv \
                 (<= 2 BF16 ULP), unlike the batch16 tier. Unset the lever to restore it (#927)."
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_m16_tc_tests.rs"]
mod tests;
