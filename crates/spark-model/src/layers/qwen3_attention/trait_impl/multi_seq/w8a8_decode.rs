// SPDX-License-Identifier: AGPL-3.0-only

//! W8A8 block-scaled cuBLASLt arm for the ATTENTION decode projections at
//! 5..16 concurrent rows — Q/K/V (strided) and O (contiguous).
//!
//! WHY (#927). H100, 2026-09-11 round 7, `Qwen/Qwen3.8-27B-FP8`, batch 16,
//! steady-state n=16 decode step **43.595 ms** (idle 4.2%), nsys
//! `--cuda-graph-trace=node`. Resolved by grid shape:
//!
//! * `w8a16_gemv_batch16_strided` GrdX 3072 — `q_proj` (N=12288, the
//!   interleaved `[Q|gate]`): 16 launches × 180.8 µs = **2.89 ms**.
//! * `w8a16_gemv_batch16_strided` GrdX 256 — `k_proj` + `v_proj` (N=1024):
//!   32 launches × 26.4 µs = **0.85 ms**.
//! * `w8a16_gemv_batch16` GrdX 1280 — the 16 `o_proj` launches inside the
//!   64-launch group, ~**1.70 ms**.
//!
//! ~5.44 ms, 12.5% of the step, at the same ~350-500 GB/s the SSM projections
//! sit at — while the dense FFN runs the SAME 16 rows through cuBLASLt W8A8 at
//! ~128 µs/layer for 267 MB (~2 100 GB/s-equivalent).
//!
//! ⚠ THE STRIDED OUTPUT IS THE WHOLE DIFFICULTY. The multi-seq QKV buffer is
//! `[n, per_seq_qkv]` — Q at offset 0 of each sequence's slot, K at
//! `q_proj_bytes`, V after it — so a contiguous `[M, N]` GEMM cannot write it.
//! `fp8_gemm_act_weight_t_blkscaled_ldc` takes the output row pitch, which is
//! exactly what cuBLASLt's D layout already has (`ldc` on a column-major
//! `[N, M]` IS a row-major `[M, N]`'s row pitch), so one extra parameter buys
//! the strided write with no staging buffer and no scatter kernel.
//!
//! ⚠ AND THE PHANTOM ROWS ARE THE TRAP. `cublas_fp8_proj_prequant` hands
//! cuBLASLt `ceil16(M) = 16` for every row count in this band, and those
//! phantom rows ARE written. With a contiguous output they land past the live
//! rows; with THIS one they land in decode slots `n..16` — slots belonging to
//! sequences that are not in this step, whose contents are re-projected before
//! anything reads them. That is correct, but only while the buffer really has
//! 16 slots, so the selector bounds the FULL padded write extent
//! (`(m_pad - 1) * ldc + n` elements, from EACH projection's own base — V's is
//! the tightest) against the arena. A buffer that cannot hold it declines to
//! the `w8a16_gemv_batch16_strided` tier rather than writing past its end.
//!
//! NUMERICS. Same trade the dense FFN already takes at these widths, and
//! vLLM's: dynamic per-token 1×128 E4M3 activation quant instead of BF16
//! activations. M=1 decode is untouched. SSOT: `ops::dispatch_proj_decode`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::layers::qwen3_attention::attn_qkv_fused;
use crate::weight_map::Fp8Weight;

/// The [`ops::CublasScope`] slice that arms this family, as ONE function so a
/// test and the dispatch site cannot disagree about which bit is read.
/// `ATLAS_CUBLAS_GEMM=ffn` must NOT reach the attention projections — that
/// exact confusion is what cost 10.3 GiB of unledgered BF16 weight copies in
/// #917 and is why the lever became a family set.
pub(super) fn attn_decode_family_armed(scope: ops::CublasScope) -> bool {
    scope.attn
}

impl Qwen3AttentionLayer {
    /// The activation-quant scratch this layer hands the decode W8A8 arm — the
    /// arena triple the attention PREFILL projections already use.
    fn decode_w8a8_scratch(&self, fwd: &ForwardContext) -> ops::DecodeW8a8Scratch {
        ops::DecodeW8a8Scratch {
            act_fp8: fwd.buffers.fp8_act(),
            act_fp8_bytes: fwd.buffers.fp8_act_bytes(),
            act_scale: fwd.buffers.fp8_act_scale(),
            act_scale_bytes: fwd.buffers.fp8_act_scale_bytes(),
            act_scale_kmajor: fwd.buffers.fp8_act_scale_kmajor(),
            act_scale_kmajor_bytes: fwd.buffers.fp8_act_scale_kmajor_bytes(),
            quant_k: self.per_token_group_quant_fp8_k,
            scale_kmajor_k: self.fp8_act_scale_kmajor_k,
        }
    }

    fn decode_w8a8_selected(
        &self,
        fwd: &ForwardContext,
        plan: &ops::DecodeW8a8Plan,
        fp8w: &Fp8Weight,
    ) -> bool {
        ops::decode_w8a8_selected(
            attn_decode_family_armed(fwd.dispatch.cublas),
            ops::w8a8_decode_proj_disabled(),
            plan,
            fp8w.scale_format,
            &self.decode_w8a8_scratch(fwd),
        )
    }

    /// The three Q/K/V plans for this step, each bounded from ITS OWN base
    /// inside `qkv_output` — Q at 0, K at `q_proj_bytes`, V after K — so the
    /// write-extent check shrinks with the offset exactly as the buffer does.
    ///
    /// `qkv_capacity` is the allocated size of `qkv_output`; `ldc` is
    /// `per_seq_qkv` in BF16 ELEMENTS, the row pitch of the slot layout.
    pub(super) fn qkv_decode_w8a8_plans(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_dim: u32,
        qkv_capacity: usize,
    ) -> [(usize, ops::DecodeW8a8Plan); 3] {
        let ldc = (c.per_seq_qkv / c.bf16) as u32;
        let kv_bytes = kv_dim as usize * c.bf16;
        let plan = |offset: usize, n_out: u32| {
            (
                offset,
                ops::DecodeW8a8Plan::strided(
                    c.n,
                    n_out,
                    c.h as u32,
                    ldc,
                    qkv_capacity.saturating_sub(offset),
                ),
            )
        };
        [
            plan(0, c.q_proj_dim),
            plan(c.q_proj_bytes, kv_dim),
            plan(c.q_proj_bytes + kv_bytes, kv_dim),
        ]
    }

    /// Route Q/K/V through cuBLASLt W8A8 if EVERY projection qualifies;
    /// `Ok(false)` leaves all three to the caller's GEMV tier.
    ///
    /// All-or-nothing on purpose: the three share one activation quantization,
    /// and a split route would pay the quantizer AND keep a full GEMV weight
    /// pass, which is the worst of both. They also share a shape — same `K`,
    /// same `ldc`, same rows — so in practice they qualify together or not at
    /// all; the only asymmetry is the write extent, and V's (the tightest) is
    /// checked like the others.
    pub(super) fn try_ms_qkv_decode_w8a8(
        &self,
        c: &MultiSeqCtx<'_>,
        q: &Fp8Weight,
        k: &Fp8Weight,
        v: &Fp8Weight,
        kv_dim: u32,
    ) -> Result<bool> {
        let fwd = c.fwd;
        let plans = self.qkv_decode_w8a8_plans(c, kv_dim, fwd.buffers.qkv_output_bytes());
        let weights = [q, k, v];
        if !plans
            .iter()
            .zip(weights)
            .all(|((_, plan), w)| self.decode_w8a8_selected(fwd, plan, w))
        {
            return Ok(false);
        }
        let scratch = self.decode_w8a8_scratch(fwd);
        self.log_decode_w8a8_route(fwd, "q/k/v", c.n, c.h as u32);
        // ONE quantization of `normed` for all three — Q, K and V read the
        // same `[n, h]` rows, so quantizing per projection would run the
        // quantizer and the VEC128 scale transpose three times per layer per
        // step. The dense FFN makes the same split for gate/up.
        ops::decode_w8a8_quant_act(
            fwd.gpu, &scratch, c.normed, c.n as u32, c.h as u32, c.stream,
        )?;
        // ── FUSED [q|k|v] at N = q_proj_dim + 2*kv_dim (#927) ──
        // The three plans above already passed, so the fused arm is the SAME
        // arithmetic with the three N ranges adjacent instead of separate, and
        // its output IS `qkv_buf`'s slot layout byte for byte (`ldc ==
        // fused_n`). Rule + receipt: `attn_qkv_fused.rs`.
        if let Some((fused_w, fused_plan)) = self.qkv_fused_plan(c, kv_dim, true) {
            self.log_qkv_fused_route(fwd, c.n, fused_plan.n);
            return ops::decode_w8a8_gemm(&scratch, fused_w, c.qkv_buf, &fused_plan, c.stream)
                .map(|()| true);
        }
        for ((offset, plan), w) in plans.iter().zip(weights) {
            ops::decode_w8a8_gemm(&scratch, w, c.qkv_buf.offset(*offset), plan, c.stream)?;
        }
        Ok(true)
    }

    /// The fused `[q|k|v]` weight and its plan for THIS step, or `None` to
    /// keep the three-GEMM arm.
    ///
    /// `three_ok` is the caller's already-resolved answer for the three
    /// separate projections, passed in rather than re-derived: asking again
    /// here would be a second copy of the W8A8 rule that could disagree with
    /// the one the loop below uses, and the fused arm must never run where the
    /// un-fused one would not have.
    ///
    /// The plan is built from the SAME `per_seq_qkv` row pitch the three use,
    /// and `decode_w8a8_selected` is re-run on it because the fused `n` is a
    /// different width with a different write extent — the clause that a
    /// `qkv_output` too small for `ceil16(m)` fused rows must decline on.
    pub(super) fn qkv_fused_plan(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_dim: u32,
        three_ok: bool,
    ) -> Option<(&Fp8Weight, ops::DecodeW8a8Plan)> {
        let fused_w = self.qkv_fp8_fused.as_ref()?;
        let k = c.h as u32;
        let ldc = (c.per_seq_qkv / c.bf16) as u32;
        let n = attn_qkv_fused::fused_n(c.q_proj_dim, kv_dim);
        let plan = ops::DecodeW8a8Plan::strided(c.n, n, k, ldc, c.fwd.buffers.qkv_output_bytes());
        attn_qkv_fused::attn_qkv_fused_selected(
            c.n,
            c.q_proj_dim,
            kv_dim,
            k,
            ldc,
            self.attn_qkv_fused,
            // "the loader built THIS weight": the shape is checked and not
            // assumed, because a checkpoint whose q/k/v widths disagree with
            // the config would otherwise be read at the wrong stride.
            fused_w.n == n && fused_w.k == k,
            three_ok && self.decode_w8a8_selected(c.fwd, &plan, fused_w),
        )
        .then_some((fused_w, plan))
    }

    /// Say ONCE that the three Q/K/V decode GEMMs became one. Worth a line for
    /// the reason the gate+up route log is: a TPOT report at 5..=16 rows is
    /// measuring this arm, and its absence at a width that should have it is
    /// the first thing to check when the round-13 9.8%-of-HBM k/v figure
    /// appears to be back.
    fn log_qkv_fused_route(&self, fwd: &ForwardContext, rows: usize, n: u32) {
        if fwd.stats.once("log:attn_qkv_fused") {
            tracing::info!(
                "[atlas] attention decode: q/k/v FUSED into ONE W8A8 block-scaled GEMM at \
                 N={n} (n={rows} rows, band 5..={max}) — same weight bytes, one launch \
                 instead of three, writing the existing [n, per_seq_qkv] slot layout \
                 unchanged. Round 13 priced k_proj+v_proj at 15.97 us/node = 328 GB/s = \
                 9.8% of HBM against q_proj's 65.3%. Bit-identical per element; \
                 ATLAS_ATTN_QKV_FUSED=0 restores the three-GEMM arm (#927).",
                max = spark_runtime::buffers::ATTN_QKV_FUSED_MAX_M,
            );
        }
    }

    /// Route the O projection through cuBLASLt W8A8; `Ok(false)` leaves it to
    /// the caller's `w8a16_gemv_batch{4,16}` group loop.
    ///
    /// Contiguous on both sides: `attn_out` is `[n, q_dim]` and `o_out` is
    /// `[n, hidden]`, so this is the plain `ldc == n` case.
    pub(super) fn try_ms_o_proj_decode_w8a8(
        &self,
        c: &MultiSeqCtx<'_>,
        o_fp8: &Fp8Weight,
        attn_out: DevicePtr,
        o_out: DevicePtr,
    ) -> Result<bool> {
        let fwd = c.fwd;
        let k = c.nq * c.hd;
        let plan =
            ops::DecodeW8a8Plan::contiguous(c.n, c.h as u32, k, fwd.buffers.moe_output_bytes());
        if !self.decode_w8a8_selected(fwd, &plan, o_fp8) {
            return Ok(false);
        }
        let scratch = self.decode_w8a8_scratch(fwd);
        self.log_decode_w8a8_route(fwd, "o_proj", c.n, k);
        ops::decode_w8a8_quant_act(fwd.gpu, &scratch, attn_out, c.n as u32, k, c.stream)?;
        ops::decode_w8a8_gemm(&scratch, o_fp8, o_out, &plan, c.stream)?;
        Ok(true)
    }

    /// Say ONCE per family which arithmetic the 5..16-row decode rows took.
    /// The rows move from W8A16 to W8A8 — a deliberate precision trade — so a
    /// coherency or quality report has to be able to state which arithmetic
    /// produced it, and the line names the switch that undoes it.
    fn log_decode_w8a8_route(&self, fwd: &ForwardContext, what: &str, rows: usize, k: u32) {
        let key = if what == "o_proj" {
            "log:attn_o_proj_w8a8_decode"
        } else {
            "log:attn_qkv_w8a8_decode"
        };
        if fwd.stats.once(key) {
            tracing::info!(
                "[atlas] attention {what} decode (n={rows} rows, K={k}): W8A8 block-scaled via \
                 cuBLASLt (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue; \
                 vLLM-equivalent FP8 numerics), replacing w8a16_gemv_batch16. \
                 ATLAS_NO_W8A8_DECODE_PROJ restores the GEMV tier; M=1 decode is untouched."
            );
        }
    }
}

#[cfg(test)]
#[path = "w8a8_decode_tests.rs"]
mod tests;
