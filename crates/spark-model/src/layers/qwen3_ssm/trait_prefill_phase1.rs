// SPDX-License-Identifier: AGPL-3.0-only

//! is_ssm_layer + prefill_phase1.

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn is_ssm_layer_inner(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();

        // Diagnostic: always sync at entry to catch prior-layer errors
        tracing::info!("ssm phase1 ENTRY: k={k} h={h} qkvz={qkvz_size}");
        ctx.gpu.synchronize(stream).map_err(|e| {
            anyhow::anyhow!("ssm phase1 ENTRY: stream broken BEFORE we start (M={k}): {e}")
        })?;

        // ── 1. RMS norm + residual for N tokens ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            k,
            h as u32,
            eps,
            stream,
        )?;
        if k > 4096 {
            ctx.gpu.synchronize(stream).map_err(|e| {
                anyhow::anyhow!(
                    "ssm phase1 L{}: SYNC after rms_norm (M={k}): {e}",
                    0 /*SSM*/
                )
            })?;
        }

        // ── 2+3. QKVZ GEMM (+ deinterleave if needed) ──
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        if let Some(fp8) = self.qkvz_fp8 {
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
                anyhow::anyhow!("ssm phase1: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
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
                    anyhow::anyhow!("ssm phase1: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}")
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
                    anyhow::anyhow!("ssm phase1: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
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
                anyhow::anyhow!("ssm phase1: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else {
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
        // Diagnostic sync: isolate QKVZ GEMM crash from later kernels.
        // Only for long sequences (>4096) where the crash occurs.
        if k > 4096 {
            ctx.gpu.synchronize(stream).map_err(|e| {
                anyhow::anyhow!("ssm phase1: SYNC after QKVZ GEMM (M={k} N={qkvz_size}): {e}")
            })?;
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

        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("ssm phase1: SYNC after deinterleave (M={k}): {e}"))?;
        }
        // ── 4+5. Fused BA GEMM + GDN gates (token-parallel) ──
        let ba_size = ctx.config.ssm_ba_size();
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2;
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            normed,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates_buf,
            k,
            ba_size as u32,
            h as u32,
            h as u32,
            gate_stride as u32,
            nv as u32,
            vpg as u32,
            stream,
        )?;

        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("ssm phase1: SYNC after BA+gates (M={k}): {e}"))?;
        }
        // ── 6. Batched conv1d for all N tokens ──
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            ssm_state.conv_state,
            deinterleaved,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            conv_out_buf,
            conv_dim as u32,
            d_conv as u32,
            k,
            qkvz_size as u32,
            conv_dim as u32,
            stream,
        )?;
        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("ssm phase1: SYNC after conv1d (M={k}): {e}"))?;
        }

        // ── 7. Batched L2 norm on Q,K for all N tokens ──
        ops::l2_norm(
            ctx.gpu,
            self.l2_norm_k,
            conv_out_buf,
            (nk * 2) as u32,
            kd as u32,
            1e-6,
            k,
            conv_dim as u32,
            stream,
        )?;

        // ── 8. Copy GDN inputs to full-sequence buffers ──
        // QKV: conv_out_buf [num_tokens, conv_dim] BF16 → gdn_bufs.qkv at token_offset
        // This is a contiguous copy because both layouts are [N, conv_dim].
        let qkv_dst = gdn_bufs.qkv.offset(token_offset * conv_dim * bf16);
        ctx.gpu
            .copy_d2d_async(conv_out_buf, qkv_dst, num_tokens * conv_dim * bf16, stream)?;

        // Gate/beta: gates_buf [num_tokens, 2*nv] FP32 → gdn_bufs.gate_beta at token_offset
        // Contiguous copy: both layouts are [N, 2*nv] FP32.
        let gb_dst = gdn_bufs.gate_beta.offset(token_offset * gate_stride * fp32);
        ctx.gpu
            .copy_d2d_async(gates_buf, gb_dst, num_tokens * gate_stride * fp32, stream)?;

        // Z gate: deinterleaved [num_tokens, qkvz_size] BF16, Z at offset (key_dim*2 + value_dim).
        // Z stride in source = qkvz_size, Z stride in dest = value_dim.
        // Strided copy: one per-token D2D async call.
        let z_src_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        let z_dst_base = gdn_bufs.z.offset(token_offset * value_dim * bf16);
        let z_elem_bytes = value_dim * bf16;
        for t in 0..num_tokens {
            let z_src = z_src_base.offset(t * qkvz_size * bf16);
            let z_dst = z_dst_base.offset(t * value_dim * bf16);
            ctx.gpu.copy_d2d_async(z_src, z_dst, z_elem_bytes, stream)?;
        }

        Ok(())
    }
}
