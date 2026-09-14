// SPDX-License-Identifier: AGPL-3.0-only

//! QKVZ projection GEMM dispatch for `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC
//! cap. [`Qwen3SsmLayer::prefill_qkvz_proj`] mirrors the original step
//! 2+3 block 1:1 — same FP8 / NVFP4 / BF16 dispatch, same deinterleave,
//! same kernel launches and buffer wiring.

use super::*;

impl Qwen3SsmLayer {
    /// QKVZ projection GEMM (+ deinterleave when QKVZ is interleaved).
    ///
    /// Writes the sequential `[Q|K|V|Z]` projection into the
    /// `ssm_deinterleaved` buffer. `force_bf16` (= `ATLAS_GDN_BF16_WEIGHTS`)
    /// bypasses both the FP8 and NVFP4 weight-quant paths.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qkvz_proj(
        &self,
        normed: DevicePtr,
        deinterleaved: DevicePtr,
        k: u32,
        qkvz_size: usize,
        h: usize,
        nk: usize,
        kd: usize,
        vpg: usize,
        vd: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        // Tier-1c keep-packed Q2_0: transient-dequant the fused qkvz then dense
        // GEMM. Bonsai is `sequential_qkvz`, so `proj_dst == deinterleaved` and
        // no post-deinterleave is needed. Highest priority (all other weight
        // slots are NULL on this path).
        if self.qkvz_q2.is_some() {
            let scratch = ctx.buffers.q2_dequant_scratch();
            let act_q8 = ctx.buffers.q2_act_q8();
            self.qkvz_q2_prefill_gemm(ctx.gpu, normed, proj_dst, scratch, act_q8, k, stream)?;
            return Ok(());
        }
        // Env override: ATLAS_GDN_BF16_WEIGHTS=1 forces the BF16 dense
        // GEMM path for QKVZ — bypassing both FP8 and NVFP4 weight-quant
        // paths. Tests whether weight-quantization noise on qkvz (esp.
        // the W_z slice that feeds gnorm's silu gate) is the dominant
        // source of long-context layer-1+ drift.
        let force_bf16 = matches!(
            std::env::var("ATLAS_GDN_BF16_WEIGHTS").ok().as_deref(),
            Some("1")
        );
        // PER-ROW FP8 straight from a mixed-precision checkpoint
        // (`ATLAS_FP8_ROWWISE=1`), dequantised ONCE to BF16 — into the
        // LEDGERED `BufferSizes::ssm_rowwise_w_bf16` slab, see
        // `rowwise_bf16.rs` (#917) — and multiplied by cuBLASLt. Ahead of every arm below because it is the only one that
        // never re-quantises: FP8 E4M3 is exactly representable in BF16, so
        // the checkpoint's precision survives, where the default path
        // dequantises to BF16 and then throws half of it away again by
        // quantising to NVFP4.
        //
        // NOT the row-wise FP8 GEMM this was first written against —
        // `cublaslt::fp8_gemm_act_weight_t_rowwise` returns NOT_SUPPORTED on
        // sm_121 (measured 2026-08-15, and reproduced through the
        // block-scaled path with `ATLAS_CUBLAS_FP8=1`, so it is the GEMM and
        // not the weights). Keeping FP8 all the way needs a kernel that works
        // on this hardware; until then BF16 is what buys the precision back.
        //
        // `force_bf16` still wins, so the `ATLAS_GDN_BF16_WEIGHTS` A/B lever
        // keeps working.
        if !force_bf16 && let Some(ref fp8w) = self.qkvz_fp8w_rowwise {
            // This arm returns EARLY, so it shadows the CUTLASS / cuBLAS arms
            // below. That is deliberate — its whole point is precision, and
            // every arm it shadows consumes the NVFP4 copy, i.e. the
            // double-quantised weights this exists to avoid — but an operator
            // who set a CUTLASS flag and silently did not get it would have no
            // way to tell. Say so, once.
            if ctx.dispatch.cutlass_nvfp4_qkvz || ctx.dispatch.cutlass_gemm {
                static SHADOW_WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !SHADOW_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        "ATLAS_FP8_ROWWISE is shadowing an enabled CUTLASS/cuBLAS QKVZ \
                         prefill arm: the row-wise arm keeps the checkpoint's precision, \
                         the shadowed arms would consume the re-quantised NVFP4 copy. \
                         Unset ATLAS_FP8_ROWWISE to get the CUTLASS path back."
                    );
                }
            }
            // The BF16 weight comes from the LEDGERED slab, dequantised once
            // per layer (`rowwise_bf16.rs`); this arm allocates nothing. It
            // used to come from `ops::cublas_bf16_proj`'s by-pointer cache of
            // `gpu.alloc`s — 167772160 B per layer with no `BufferSizes`
            // entry, the #917 H100 OOM at layer 36.
            let w_bf16 = self.rowwise_qkvz_bf16(ctx, fp8w, stream)?;
            ops::cublas_bf16_proj_dense(
                normed,
                w_bf16,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
            return Ok(());
        }
        let force_w8a8 = ctx.dispatch.fp8_blockscaled_prefill;
        // NO cuBLASLt-BF16 ARM HERE, deliberately. `ATLAS_CUBLAS_GEMM=1` used
        // to route this projection to `ops::cublas_bf16_proj`, whose cached
        // FP8→BF16 weight dequant cost 167772160 B per layer (~10.3 GiB over 48
        // SSM layers) outside the buffer ledger and killed a 28-token H100
        // prefill at layer 36 on 2026-09-11. The cuBLASLt arm now lives INSIDE
        // the W8A8 branch below, consumes the FP8 weight directly, and
        // allocates nothing — see `prefill_w8a8.rs` for the full receipt.
        if ctx.dispatch.cutlass_nvfp4_qkvz
            && let Some(ref nvfp4_t) = self.qkvz_nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "ssm_qkvz_nvfp4", k, qkvz_size as u32, h as u32);
            ops::cutlass_nvfp4_proj(
                ctx,
                normed,
                nvfp4_t,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if ctx.dispatch.cutlass_nvfp4_qkvz
            && let Some(ref fp8w) = self.qkvz_fp8w
        {
            ops::log_cutlass_nvfp4_route(
                ctx.gpu,
                "ssm_qkvz_fp8pack",
                k,
                qkvz_size as u32,
                h as u32,
            );
            ops::cutlass_nvfp4_proj_from_fp8(
                ctx,
                normed,
                fp8w,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if ctx.dispatch.cutlass_gemm
            && let Some(ref fp8w) = self.qkvz_fp8w
        {
            ops::cutlass_bf16_proj(
                ctx.gpu,
                ctx.derived,
                normed,
                fp8w,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if ctx.dispatch.cublas_fp8
            && let Some(ref fp8w) = self.qkvz_fp8w
        {
            ops::cublas_fp8_rowwise_proj(
                ctx.gpu,
                ctx.derived,
                normed,
                ctx.buffers.fp8_act(),
                ctx.buffers.fp8_act_scale(),
                fp8w,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if force_bf16 {
            // cuBLASLt, NOT the hand-written `dense_gemm`. The weights are
            // already BF16 [N,K] on this path, so there is no dequant step and
            // nothing to cache — `cublas_bf16_proj_dense` exists for exactly
            // this shape.
            //
            // MEASURED 2026-08-15 on unsloth/Qwen3.8-27B-NVFP4: this lever
            // through `dense_gemm` cost 72.9% of prefill (507 -> 137 tok/s),
            // which is what made "keep the GDN weights BF16" look like a
            // quality-for-speed trade. It was never the precision — it was
            // the GEMM.
            ops::cublas_bf16_proj_dense(
                normed,
                self.ssm.in_proj_qkvz.weight,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ BF16 cuBLASLt GEMM failed (M={k}, N={qkvz_size}): {e}"
                )
            })?;
        } else if force_w8a8
            && let Some(ref fp8w) = self.qkvz_fp8w
            && self.per_token_group_quant_fp8_k.available()
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            tracing::debug!(
                "ssm prefill: QKVZ via block-scaled FP8 (W8A8+FP32-epilogue, M={k} K={h} N={qkvz_size})"
            );
            let k_dim = h;
            // Persistent arena scratch (no per-projection alloc/sync/free): the
            // quant→GEMM chain is same-stream ordered. PADDED rows, because the
            // cuBLASLt arm reads `ceil16(M)` of them.
            let m_pad = ops::cublas_fp8_m_pad(k) as usize;
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m_pad * k_dim <= ctx.buffers.fp8_act_bytes());
            debug_assert!(m_pad * k_dim.div_ceil(128) * 4 <= ctx.buffers.fp8_act_scale_bytes());
            // Per-token block FP8 quant of the activation, then block-scaled
            // FP8×FP8 GEMM folding both per-128 scales in an FP32 epilogue —
            // cuBLASLt or the in-tree kernel, chosen by `qkvz_w8a8_gemm`.
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed,
                a_fp8_buf,
                a_scale_buf,
                k,
                k_dim as u32,
                stream,
            )?;
            // `proj_dst` is one of two ARENA buffers, picked above by
            // `sequential_qkvz`; the cuBLASLt arm writes `ceil16(M)` rows into
            // it, so it is bounds-checked against the one it actually got.
            let dst_capacity = if self.sequential_qkvz {
                ctx.buffers.ssm_deinterleaved_bytes()
            } else {
                ctx.buffers.ssm_qkvz_bytes()
            };
            self.qkvz_w8a8_gemm(
                ctx,
                a_fp8_buf,
                a_scale_buf,
                fp8w,
                proj_dst,
                dst_capacity,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(ref fp8w) = self.qkvz_fp8w
            && self.w8a16_gemm_pipelined_k.0 != 0
        {
            // Block-scaled W8A16 prefill: matches vLLM's per-128-block FP32
            // scale precision (vs the single-scale fp8_gemm_n128 below
            // which bakes ALL per-block scales into one global scale,
            // dropping per-block dynamic range). This is the SSM-side of
            // the W8A8+FP32-epilogue fix shipped for the attention layer.
            //
            // Block-scaled W8A16 QKVZ routed through the bit-identical
            // (cosine=1.0) ~4.6× faster tensor-core w8a16_gemm_pipelined kernel
            // where available (NVIDIA). gfx1151/HIP has no cp.async, so that
            // kernel is absent there â fall through to the cp.async-free
            // non-pipelined w8a16_gemm branch below.
            ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ w8a16_gemm_pipelined failed (M={k}, N={qkvz_size}): {e}"
                )
            })?;
        } else if let Some(ref fp8w) = self.qkvz_fp8w
            && self.w8a16_gemm_k.0 != 0
        {
            // cp.async-free fallback (gfx1151/HIP): non-pipelined block-scaled
            // W8A16 GEMM. Same per-128-block FP32-scale math as the pipelined
            // kernel, without the sm_80+ cp.async multistage prefetch.
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ w8a16_gemm (block-scaled) failed (M={k}, N={qkvz_size}): {e}"
                )
            })?;
        } else if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            if k > 128 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ssm prefill: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}"
                    )
                })?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else {
            // Pure dense-BF16 weights (qwen4_exp keeps the GDN projections
            // BF16 as shipped): cuBLASLt, NOT the hand-written scalar
            // `dense_gemm`. Same finding as the `force_bf16` arm above —
            // measured HERE on Qwen3.8-Flash-Next 2026-08-26: this arm
            // through `dense_gemm` was 147 ms/call x 36 layers = 5.3 s of an
            // 8.4 s TTFT (63% of prefill, ~2.7 TFLOPS on an 85-TFLOP part).
            // The scalar kernel stays as the fallback for backends without
            // cuBLASLt.
            if ops::cublas_bf16_proj_dense(
                normed,
                self.ssm.in_proj_qkvz.weight,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .is_err()
            {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    normed,
                    &self.ssm.in_proj_qkvz,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        }
        if !self.sequential_qkvz {
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                k,
                nk as u32,
                kd as u32,
                vpg as u32,
                vd as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
