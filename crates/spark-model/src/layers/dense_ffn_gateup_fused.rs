// SPDX-License-Identifier: AGPL-3.0-only

//! The FUSED dense-FFN gate+up DECODE GEMM — one block-scaled FP8 cuBLASLt
//! call at `N = 2 * intermediate` in place of two at `N = intermediate`.
//!
//! # WHY (#927)
//!
//! nsys `--cuda-graph-trace=node`, 1xH100 80GB HBM3, `Qwen/Qwen3.8-27B-FP8` @
//! `3717cb05e`, round 13 cell V, median `n = 16` decode step **19.887 ms** of
//! kernel busy (`h100-r13-attribution.md` §§C.2–C.4). Resolved by grid shape,
//! the dense FFN's gate and up projections are **128 graph nodes, 5 730.5 µs =
//! 44.77 µs/node**, `K = 5120` `N = 17408` each. At 89.1 MB of E4M3 weight per
//! node that is **1 991 GB/s = 59.4 % of HBM**.
//!
//! In the SAME step, on the SAME arm, moving the SAME bytes:
//!
//! | projection | K | N | nodes | µs/node | GB/s | % HBM |
//! |---|---|---|---|---|---|---|
//! | FFN gate + up | 5120 | 17408 | 128 | 44.77 | 1 991 | **59.4 %** |
//! | FFN `down` | 17408 | 5120 | 64 | 37.25 | 2 393 | 71.4 % |
//! | SSM `in_proj_qkvz` | 5120 | 16384 | 48 | 34.21 | 2 453 | 73.2 % |
//!
//! `down` reads the same 89.1 MB as one of the gate/up nodes and is 7.5 µs
//! faster; the difference between the 59.4 % arm and the 71–73 % arms is that
//! the first issues **two launches per layer for one weight pass**. The weight
//! bytes are read once either way — this is not a traffic saving. One launch
//! of twice the N halves the per-launch fixed cost and doubles the tile count
//! per wave, which is what the 71.4 % row already demonstrates on this card.
//! At an 80 % target the pair costs `11.41 GB / (0.8 × 3.35 TB/s) = 4.26 ms`
//! against a measured 5 730.5 µs → **1 476 µs/step, 7.4 % of the step** and
//! the largest single decode kernel item in the round-13 table.
//!
//! # Numerics: a bit claim, not a tolerance
//!
//! The fused weight is the two `[inter, K]` E4M3 blocks appended along N, and
//! its `[N/128, K/128]` FP32 block-scale grid is the two grids appended along
//! N/128. Splitting N therefore produces **independent output columns over the
//! same K with the same scales**: output element `(m, j)` of the fused GEMM is
//! the same dot product, in the same order, as element `(m, j)` of gate (for
//! `j < inter`) or `(m, j - inter)` of up. Same cuBLASLt op, same epilogue.
//! `examples/native_fp8_ffn_gateup_fused_microtest.rs` asserts **byte
//! equality** of both halves at `M ∈ {5, 8, 16}` rather than a cosine.
//!
//! # Layout, and why it is N-concatenation rather than an interleave
//!
//! The fused output is `[m, 2*inter]` with gate in columns `[0, inter)` and up
//! in `[inter, 2*inter)` — a row is `[gate | up]`. Three things stay simple
//! that a tile-interleave would complicate: the loader's fused weight is a
//! straight device-to-device append, its scale grid is the same append one
//! row-block wider, and **each half remains addressable as an un-fused
//! `Fp8Weight` VIEW**, so every other rung of `dense_ffn.rs`'s `w8_gemm!`
//! ladder keeps working on the same bytes with no change at all. The consumer
//! pays a row stride instead of a flat index (`ops::silu_mul_strided`), and
//! coalescing survives it: a row half is `inter` contiguous BF16 — 34 816 B at
//! these shapes — so every warp's 128-byte segments are whole and only the
//! jump BETWEEN rows differs.
//!
//! # Residency: net zero, by construction
//!
//! A second copy of gate+up is 178.3 MB × 64 layers = **11.4 GB**, which would
//! not fit beside the bs32 KV budget. So the loader does not make one: it
//! builds the fused buffer, re-points `gate_proj` and `up_proj` at VIEWS
//! inside it, and `Qwen35DenseWeightLoader::prune_after_load` releases the two
//! source store tensors the copy consumed. Steady-state delta is zero, and
//! `predicted_residency` prices it as zero for the preflight ring fit. The
//! load-time transient is one layer's 178.3 MB at a time against the store
//! tensors that have not been pruned yet — the same shape, and the same
//! precedent, as the SSM `[QKV|Z]` concat that has shipped since #915.
//!
//! # Band
//!
//! 5..=[`GATEUP_FUSED_MAX_M`] rows.
//!
//! * **Below 5** rung 1 of `w8_gemm!` (`w8a16_gemv_batch4`) owns the width and
//!   already makes ONE pass over each weight; there is no second launch to
//!   fuse, and the W8A8 rule this arm rides on starts at `m > 4` anyway.
//! * **Above 16** the arm stops because the saving does. At the prefill widths
//!   these same two GEMMs run at **68.6 % of FP8 PEAK** (M=4576,
//!   `h100-r13-attribution.md` §A.4) — compute-bound, where a launch buys
//!   nothing measurable — and the attribution's own advice is to keep the
//!   lever scoped to decode until a prefill microtest says otherwise.
//!
//! The lever is `[defaults] ffn_gateup_fused`: **hopper `true`**, gb10 and
//! b200 `false` (no receipt, and both declare `cublas_gemm_scope = "off"`, so
//! the arm this changes is not even armed there). The strided SiLU consumer
//! lives in `kernels/hopper/common/silu_mul_strided.cu` — HOPPER-OWNED
//! (`[kernels] overrides`, an addition), so the two other NVIDIA targets do
//! not compile a kernel they can never launch and no new cross-hardware
//! symlink is created. `ATLAS_FFN_GATEUP_FUSED=0` kills it; on gb10/b200
//! `=1` arms a lever whose kernel lookup returns 0 and the arm declines.

use anyhow::Result;
use spark_runtime::buffers::GATEUP_FUSED_MAX_M;
use spark_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// Whether the compiled target arms the fused gate+up decode GEMM.
///
/// The target declares it (`kernels/<hw>/HARDWARE.toml` `[defaults]
/// ffn_gateup_fused`); `ATLAS_FFN_GATEUP_FUSED` overrides it under the
/// 2026-09-11 grammar, so `=0`/`=off`/`=false` turn it off and anything else
/// turns it on. There is no `ATLAS_NO_*` legacy spelling: the lever is new, so
/// no script predates the grammar and none can be surprised by it.
pub fn ffn_gateup_fused() -> bool {
    ops::target_defaults::resolved().ffn_gateup_fused.value
}

/// Bytes the fused `[ceil16(m), 2 * inter]` BF16 output occupies.
///
/// `ceil16` because `cublas_fp8_proj_prequant` hands cuBLASLt `ceil16(M)` and
/// the phantom rows are WRITTEN. SSOT for both the arena sizing check below
/// and the microtest's guard bands.
pub(crate) fn fused_out_bytes(m: u32, inter: u32) -> usize {
    ops::cublas_fp8_m_pad(m) as usize * 2 * inter as usize * 2
}

/// The whole fused-arm selection rule, as a pure function.
///
/// Split out from the layer for the reason `w8a8_prefill_selected` and
/// `batch16_plan` are: the CPU tests pin every clause without a
/// `ForwardContext`, and `lever` is injected because the process-global
/// `OnceLock` behind [`ffn_gateup_fused`] cannot be toggled per test.
///
/// Clauses, each load-bearing:
///
/// * `lever` — the target's declaration, environment-overridable.
/// * `gate_up_w8a8` — the W8A8 block-scaled arm was selected for BOTH gate and
///   up. This arm IS that arm with a wider N: if the ladder would have put
///   either half on a W8A16 rung, fusing would silently change the
///   ARITHMETIC of that half, not just its launch count. It also carries every
///   clause of `w8a8_prefill_selected` transitively — format, `k % 128`,
///   `n % 128`, the per-arch ceiling, both kernel handles, the shared scratch.
/// * `5..=GATEUP_FUSED_MAX_M` — the band, see the module docs.
/// * `fused_installed` — the loader built the `[2*inter, K]` weight. Absent on
///   any checkpoint or route the fusion was not built for, which is the only
///   thing that makes this arm optional at runtime.
/// * `silu_strided_loaded` — the strided SiLU consumer. Without it the fused
///   output has no reader, and a target whose kernel set lacks the entry point
///   must decline rather than launch `moe_silu_mul` over the wrong stride.
/// * `out_capacity_bytes` — the arena's `ffn_gate_up_fused` buffer holds the
///   PADDED extent. A gate and not an assert, for the reason the null-scratch
///   check in `prefill_w8a8_selected` is one: too small is a cross-buffer
///   write, and declining is always sound.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gateup_fused_selected(
    m: u32,
    inter: u32,
    lever: bool,
    gate_up_w8a8: bool,
    fused_installed: bool,
    silu_strided_loaded: bool,
    out_capacity_bytes: usize,
) -> bool {
    lever
        && gate_up_w8a8
        && fused_installed
        && silu_strided_loaded
        && (5..=GATEUP_FUSED_MAX_M as u32).contains(&m)
        && fused_out_bytes(m, inter) <= out_capacity_bytes
}

impl DenseFfnLayer {
    /// Whether THIS layer fuses gate+up for `m` rows.
    ///
    /// `gate_up_w8a8` is the caller's because `forward_prefill_inner` has
    /// already resolved it for both projections — asking again here would be a
    /// second copy of the W8A8 rule that could disagree with the one the
    /// `w8_gemm!` ladder uses.
    pub(crate) fn gateup_fused_plan(
        &self,
        ctx: &ForwardContext,
        m: u32,
        inter: u32,
        gate_up_w8a8: bool,
    ) -> Option<&Fp8Weight> {
        let fused = self.fp8_gate_up_fused.as_ref();
        // SiLU only, and not because the fusion cares: the consumer this arm
        // launches IS the SiLU·mul, so a GeLU layer would silently get the
        // wrong activation. GeLU keeps the `w8_gemm!` pair and its
        // `self.act_mul`, which is the gelu kernel. Same shape as the
        // packed-Q2 and LoRA paths, which refuse rather than assume.
        let silu = self.activation == super::FfnActivation::SiLU;
        gateup_fused_selected(
            m,
            inter,
            self.gateup_fused && silu,
            gate_up_w8a8,
            fused.is_some(),
            self.silu_mul_strided_k.0 != 0,
            ctx.buffers.ffn_gate_up_fused_bytes(),
        )
        .then_some(fused)
        .flatten()
    }

    /// gate+up in ONE GEMM, then SiLU·mul straight out of its `[m, 2*inter]`
    /// output into the contiguous `[m, inter]` the down projection reads.
    ///
    /// Allocates nothing: `fused_out` is the arena's `ffn_gate_up_fused`, and
    /// `w8a8_gemm`'s own operands are the arena's shared W8A8 scratch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a8_gate_up_fused(
        &self,
        ctx: &ForwardContext,
        a_fp8: DevicePtr,
        a_scale: DevicePtr,
        fused_w: &Fp8Weight,
        gate_out: DevicePtr,
        m: u32,
        inter: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        self.log_gateup_fused_route(ctx, m);
        let fused_out = ctx.buffers.ffn_gate_up_fused();
        let cap = ctx.buffers.ffn_gate_up_fused_bytes();
        debug_assert!(fused_out_bytes(m, inter) <= cap);
        self.w8a8_gemm(
            ctx,
            a_fp8,
            a_scale,
            fused_w,
            fused_out,
            cap,
            m,
            2 * inter,
            h,
            stream,
        )?;
        // `up` is the same buffer, one row-half along: BF16, so `inter`
        // elements is `inter * 2` bytes. The output is the CONTIGUOUS
        // `[m, inter]` every downstream consumer already expects, so nothing
        // past this launch knows the projection was fused.
        const BF16: usize = 2;
        ops::silu_mul_strided(
            ctx.gpu,
            self.silu_mul_strided_k,
            fused_out,
            fused_out.offset(inter as usize * BF16),
            gate_out,
            m,
            inter,
            2 * inter,
            inter,
            stream,
        )
    }

    /// Log-once latch, in the same `log:ffn_*` shape the other dense-FFN route
    /// logs use. Worth a line: a TPOT report at 5..=16 rows is measuring this
    /// arm, and its absence at a width that should have it is the first thing
    /// to check when the round-13 59.4 %-of-HBM figure appears to be back.
    fn log_gateup_fused_route(&self, ctx: &ForwardContext, m: u32) {
        if ctx.stats.once("log:ffn_gateup_fused") {
            tracing::info!(
                "[atlas] dense FFN decode: gate+up FUSED into ONE W8A8 \
                 block-scaled GEMM at N=2*intermediate (m={m}, band 5..={max}) \
                 — same weight bytes, one launch instead of two. Round 13 \
                 priced the un-fused pair at 5 730.5 us/step, 59.4% of HBM, \
                 against `down`'s 71.4% for the same bytes in one launch. \
                 Bit-identical per element; ATLAS_FFN_GATEUP_FUSED=0 restores \
                 the two-GEMM arm (#927).",
                max = GATEUP_FUSED_MAX_M,
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_gateup_fused_tests.rs"]
mod tests;
