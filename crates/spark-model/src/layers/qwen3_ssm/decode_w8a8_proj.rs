// SPDX-License-Identifier: AGPL-3.0-only

//! W8A8 block-scaled cuBLASLt arm for the SSM/GDN **decode** projections at
//! 5..16 concurrent rows — `in_proj_qkvz` and `out_proj`.
//!
//! WHY (#927). H100, 2026-09-11 round 7, `Qwen/Qwen3.8-27B-FP8`, batch 16,
//! steady-state n=16 decode step **43.595 ms** (idle 4.2%), nsys
//! `--cuda-graph-trace=node`. Resolved by grid shape, the two projections in
//! this file are the largest and third-largest weight reads in the step:
//!
//! * `in_proj_qkvz` (N=16384, K=5120): **48 launches × 235.3 µs = 11.29 ms =
//!   25.9% of the whole step**, moving 83.9 MB per launch at **357 GB/s** —
//!   about a ninth of the H100's 3 350 GB/s HBM3 roofline. The SAME weight
//!   read costs 45.3 µs at M=1 (1 852 GB/s): 5.2× the time for 1× the bytes,
//!   which is the `w8a16_gemv_batch16` template paying ~16 scalar FFMA per
//!   weight byte once the row count stops being 1.
//! * `out_proj` (N=5120) shares the GrdX=1280 group with the attention
//!   `o_proj`: 64 launches × 106.5 µs = 6.81 ms, of which the 48 SSM ones are
//!   ~5.11 ms.
//!
//! Meanwhile the dense FFN, at the SAME 16 rows in the SAME step, runs
//! cuBLASLt W8A8 at ~128 µs/layer for 267 MB of weights (~2 100 GB/s
//! -equivalent, coherency 4/4). This file gives the SSM projections that same
//! path: one `per_token_group_quant_fp8` over the normed input, one
//! `fp8_act_scale_to_kmajor` to the VEC128 layout cuBLASLt documents, then
//! `nvjet` against the checkpoint's own FP8 bytes and 128×128 block scales.
//! Zero device memory: every buffer is arena scratch that already exists, and
//! the weight is consumed in FP8 (the #917 BF16-dequant arm's 10.3 GiB of
//! off-ledger copies is exactly what the W8A8 route replaced in prefill).
//!
//! NUMERICS. Same trade the dense FFN already takes at these widths, and
//! vLLM's: dynamic per-token 1×128 E4M3 activation quant instead of BF16
//! activations. M=1 decode is untouched and still bit-exact. See
//! `ops::dispatch_proj_decode` for the full SSOT note.
//!
//! FALLBACK. Every clause of `ops::decode_w8a8_selected` that fails drops the
//! projection back to the `w8a16_gemv_batch16` tier it uses today — the arm is
//! additive, never a precondition.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// Which of the two SSM decode projections a route line is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SsmDecodeProj {
    Qkvz,
    OutProj,
}

impl SsmDecodeProj {
    fn label(self) -> &'static str {
        match self {
            Self::Qkvz => "in_proj_qkvz",
            Self::OutProj => "out_proj",
        }
    }

    fn log_key(self) -> &'static str {
        match self {
            Self::Qkvz => "log:ssm_qkvz_w8a8_decode",
            Self::OutProj => "log:ssm_out_proj_w8a8_decode",
        }
    }
}

/// The [`ops::CublasScope`] slice that arms this family, as ONE function so a
/// test and the dispatch site cannot disagree about which bit is read.
/// `ATLAS_CUBLAS_GEMM=ffn` must NOT reach the SSM projections — that exact
/// confusion is what cost 10.3 GiB of off-ledger BF16 weight copies in #917.
pub(super) fn ssm_decode_family_armed(scope: ops::CublasScope) -> bool {
    scope.ssm
}

impl Qwen3SsmLayer {
    /// The activation-quant scratch this layer hands the decode W8A8 arm. The
    /// same arena triple the SSM PREFILL projection uses (`fp8_act`,
    /// `fp8_act_scale`, `fp8_act_scale_kmajor`) — decode and prefill never run
    /// in the same step, and inside a step every consumer is same-stream
    /// ordered behind its producer.
    fn decode_w8a8_scratch(&self, ctx: &ForwardContext) -> ops::DecodeW8a8Scratch {
        ops::DecodeW8a8Scratch {
            act_fp8: ctx.buffers.fp8_act(),
            act_fp8_bytes: ctx.buffers.fp8_act_bytes(),
            act_scale: ctx.buffers.fp8_act_scale(),
            act_scale_bytes: ctx.buffers.fp8_act_scale_bytes(),
            act_scale_kmajor: ctx.buffers.fp8_act_scale_kmajor(),
            act_scale_kmajor_bytes: ctx.buffers.fp8_act_scale_kmajor_bytes(),
            quant_k: self.per_token_group_quant_fp8_k,
            scale_kmajor_k: self.fp8_act_scale_kmajor_k,
        }
    }

    /// Whether ONE SSM decode projection takes the W8A8 cuBLASLt arm.
    ///
    /// `rows` is the PADDED ctx `n` — the value `padded_batch_n` produced and
    /// the CUDA-graph cache is keyed by — so the branch bakes exactly what the
    /// captured graph was built for.
    fn ssm_decode_w8a8_selected(
        &self,
        ctx: &ForwardContext,
        plan: &ops::DecodeW8a8Plan,
        fp8w: &Fp8Weight,
    ) -> bool {
        ops::decode_w8a8_selected(
            ssm_decode_family_armed(ctx.dispatch.cublas),
            ops::w8a8_decode_proj_disabled(),
            plan,
            fp8w.scale_format,
            &self.decode_w8a8_scratch(ctx),
        )
    }

    /// Route one SSM decode projection through cuBLASLt W8A8 if every clause
    /// holds; `Ok(false)` means the caller's existing GEMV/GEMM tiers own it.
    ///
    /// `rows` is the PADDED ctx `n`, `n_out` the projection's output width and
    /// `k` its contract width. `out_capacity_bytes` is the ARENA size of `out`
    /// — the padded rows ARE written, so this is a bound and not a formality.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_ssm_decode_w8a8(
        &self,
        ctx: &ForwardContext,
        which: SsmDecodeProj,
        act_bf16: DevicePtr,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        out_capacity_bytes: usize,
        rows: usize,
        n_out: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let plan = ops::DecodeW8a8Plan::contiguous(rows, n_out, k, out_capacity_bytes);
        if !self.ssm_decode_w8a8_selected(ctx, &plan, fp8w) {
            return Ok(false);
        }
        self.ssm_decode_w8a8_proj(ctx, which, act_bf16, fp8w, out, &plan, stream)?;
        Ok(true)
    }

    /// Run one SSM decode projection through cuBLASLt W8A8: quantize the
    /// activation into the shared scratch, then one matmul.
    ///
    /// Two launches of our own (`per_token_group_quant_fp8`,
    /// `fp8_act_scale_to_kmajor`) plus the library's, against the single
    /// `w8a16_gemv_batch16` launch it replaces — the count goes UP and the
    /// time goes down, which is the same trade the dense FFN's
    /// quantise/reduce tail already makes (0.27 + 0.26 + 0.14 ms of tail
    /// against 8.22 ms of nvjet at n=16).
    fn ssm_decode_w8a8_proj(
        &self,
        ctx: &ForwardContext,
        which: SsmDecodeProj,
        act_bf16: DevicePtr,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        plan: &ops::DecodeW8a8Plan,
        stream: u64,
    ) -> Result<()> {
        let scratch = self.decode_w8a8_scratch(ctx);
        self.log_decode_w8a8_route(ctx, which, plan);
        ops::decode_w8a8_quant_act(
            ctx.gpu,
            &scratch,
            act_bf16,
            plan.rows as u32,
            plan.k,
            stream,
        )?;
        ops::decode_w8a8_gemm(&scratch, fp8w, out, plan, stream)
    }

    /// Say ONCE per family which arithmetic the 5..16-row decode rows took.
    ///
    /// This is not bookkeeping: the rows move from W8A16 to W8A8, a deliberate
    /// precision trade, so any coherency or quality report has to be able to
    /// state which arithmetic produced it — and the line names the kill switch
    /// that undoes it.
    fn log_decode_w8a8_route(
        &self,
        ctx: &ForwardContext,
        which: SsmDecodeProj,
        plan: &ops::DecodeW8a8Plan,
    ) {
        if ctx.stats.once(which.log_key()) {
            tracing::info!(
                "[atlas] SSM {} decode (n={} rows, N={} K={}): W8A8 block-scaled via cuBLASLt \
                 (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue; \
                 vLLM-equivalent FP8 numerics), replacing w8a16_gemv_batch16. \
                 ATLAS_NO_W8A8_DECODE_PROJ restores the GEMV tier; M=1 decode is untouched.",
                which.label(),
                plan.rows,
                plan.n,
                plan.k,
            );
        }
    }
}

#[cfg(test)]
#[path = "decode_w8a8_proj_tests.rs"]
mod tests;
