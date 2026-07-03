// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq — batched-projection SSM mixer.

use super::super::*;

impl Qwen3SsmLayer {
    /// Batched-projection SSM mixer for N concurrent decode sequences.
    ///
    /// Returns `Ok(false)` (caller falls back to the per-seq loop) unless the
    /// layer is in the GB10 Holo serving config: sequential-QKVZ dense/NVFP4
    /// weights + FP32 conv/GDN recurrent kernels. When eligible, the big QKVZ
    /// and out_proj projections run as a single `[N, ...]` GEMM each (weights
    /// read ONCE, not N times — the dominant bandwidth cost on LPDDR5X), while
    /// the recurrent inner (BA/gates → conv1d → GDN → gated-norm) stays a
    /// per-seq loop using the SAME single-token kernels as `ssm_forward`, so
    /// the recurrence is byte-identical to the proven path. The per-seq states
    /// are read straight from each `SsmLayerState`, so no contiguous-slot
    /// assumption is required.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_decode_multi_seq_ssm_batched<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
        let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
        // QKVZ via dense BF16 GEMM or block-scaled FP8 GEMM (w8a16). NVFP4 and
        // interleaved-QKVZ layouts take the proven per-seq loop.
        // FP8 build → batched w8a16 GEMM; NVFP4 build → batched w4a16 GEMV
        // (batch4/16, M<=16). Either amortizes the QKVZ/out_proj weight read
        // across the n seqs; otherwise the per-seq loop re-streams it n times.
        let qkvz_ok = (self.qkvz_nvfp4.is_none() && self.w8a16_gemm_k.0 != 0)
            || (self.qkvz_nvfp4.is_some() && self.w4a16_gemv_batch4_k.0 != 0 && n <= 16);
        let out_ok = self.out_proj_fp8w.is_some()
            || self.out_proj_dense.is_some()
            || self.qkvz_nvfp4.is_some();
        if n < 2 || !self.sequential_qkvz || !use_f32_conv || !use_f32_gdn || !qkvz_ok || !out_ok {
            return Ok(false);
        }

        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let qk_channels = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size() as u32;

        let normed_base = ctx.buffers.norm_output();
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        // normed_out[0..n] (post gated-norm, [N, value_dim] BF16) parks in the
        // QKVZ scratch — free here because QKVZ projects into `deinterleaved`
        // and the FP32 conv path uses `ssm_conv_out_f32`, not `ssm_qkvz`.
        let normed_out_base = ctx.buffers.ssm_qkvz();
        let ssm_out_base = ctx.buffers.moe_output();
        let detail_profile = std::env::var("ATLAS_SSM_DETAIL_PROFILE").ok().as_deref() == Some("1")
            && !ctx.graph_capture;
        let mut detail_parts: Vec<(&'static str, u128)> = Vec::new();
        let mut detail_t0 = if detail_profile {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! detail_step {
            ($label:expr) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                    detail_t0 = Some(std::time::Instant::now());
                }
            };
            ($label:expr, final) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                }
            };
        }

        // ── 1. Batched input RMS norm: hidden[0..n] → normed[0..n], residual ──
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed_base,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        detail_step!("input_norm");

        // ── 2. Batched QKVZ projection: ONE [N,h]→[N,qkvz] GEMM (weights ×1) ──
        // FP8 (w8a16) when the decode overlay is installed, else BF16 dense.
        // Prefer the pipelined (cp.async) w8a16 kernel — bit-identical, ~4.6×
        // faster than the base w8a16_gemm, which nsys showed as 44.6% of the
        // C>1 decode step. `.0 == 0` → fall back to the base kernel.
        let w8a16_pipe = self.w8a16_gemm_pipelined_k.0 != 0;
        // Weight-streaming block-scaled GEMV for batched decode: avoids the
        // pipelined kernel's M->128 MMA pad (issue-bound). batch4 (M<=4) for the
        // common path, batch16 (M<=16) for high-concurrency C=8/16. Bit-identical
        // per row to w8a16_gemv. Disable with ATLAS_SSM_GEMV_BATCH4=0.
        let gemv_batch_k = if n <= 4 {
            self.w8a16_gemv_batch4_k
        } else {
            self.w8a16_gemv_batch16_k
        };
        let use_batch4 = gemv_batch_k.0 != 0
            && n <= 16
            && std::env::var("ATLAS_SSM_GEMV_BATCH4").ok().as_deref() != Some("0");
        // FP4 sibling: w4a16_gemv batch4 (M<=4) / batch16 (M<=16). Single NVFP4
        // weight pass for the QKVZ + out_proj GEMVs (amortizes the weight read).
        let fp4_gemv_batch_k = if n <= 4 {
            self.w4a16_gemv_batch4_k
        } else {
            self.w4a16_gemv_batch16_k
        };
        if let Some(ref fp8) = self.qkvz_fp8w {
            if use_batch4 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            // FP4 batched QKVZ: ONE NVFP4 weight pass for all n seqs
            // (sequential layout writes the deinterleaved buffer directly).
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                fp4_gemv_batch_k,
                normed_base,
                nvfp4,
                deinterleaved,
                n as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_base,
                &self.ssm.in_proj_qkvz,
                deinterleaved,
                n as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        detail_step!("qkvz");

        // ── 3. Recurrent inner ──
        // Default: per-seq, byte-identical to ssm_forward. Experimental path:
        // use existing batch dimensions for BA/gates, conv, GDN, and gated norm
        // when the SSM pool states are contiguous slots [0..n).
        self.decode_ms_ssm_recurrent(
            states,
            n,
            normed_base,
            deinterleaved,
            normed_out_base,
            qkvz_size,
            key_dim,
            value_dim,
            conv_dim,
            qk_channels,
            d_conv,
            nk,
            nv,
            kd,
            vd,
            vpg,
            ba_size,
            h,
            bf16,
            eps,
            detail_profile,
            &mut detail_parts,
            &mut detail_t0,
            ctx,
            stream,
        )?;
        detail_step!("recurrent_total_tail");

        // ── 4. Batched out_proj: ONE [N,value_dim]→[N,h] GEMM (weights ×1) ──
        // FP8 (w8a16) when the decode overlay is installed, else BF16 dense.
        if let Some(ref fp8) = self.out_proj_fp8w {
            if use_batch4 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if let Some(ref out_proj_dense) = self.out_proj_dense {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_out_base,
                out_proj_dense,
                ssm_out_base,
                n as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if self.qkvz_nvfp4.is_some() {
            // FP4 batched out_proj: ONE NVFP4 weight pass for all n seqs.
            // (qkvz_nvfp4.is_some() ⇒ the NVFP4 SSM build, where ssm.out_proj
            // is the NVFP4 weight the per-seq path also uses via w4a16_gemv.)
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                fp4_gemv_batch_k,
                normed_out_base,
                &self.ssm.out_proj,
                ssm_out_base,
                n as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        }
        detail_step!("out_proj");

        // GDN HeadParallel: reduce the row-parallel partial out_proj across TP
        // ranks (n × h BF16) before the residual add. No-op at tp=1.
        self.ssm_tp_all_reduce(ssm_out_base, n, ctx, stream)?;

        // ── 5. Batched residual add + post-attn RMS norm → norm_output[0..n] ──
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out_base,
            &self.post_attn_norm,
            normed_base,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        detail_step!("post_norm", final);
        if detail_profile {
            let summary = detail_parts
                .iter()
                .map(|(label, us)| format!("{label}={us}us"))
                .collect::<Vec<_>>()
                .join(" ");
            tracing::info!("ATLAS_SSM_DETAIL n={n}: {summary}");
        }

        Ok(true)
    }
}
