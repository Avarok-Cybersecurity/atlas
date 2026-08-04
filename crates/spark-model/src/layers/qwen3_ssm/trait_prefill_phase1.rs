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

        // Diagnostic: sync at entry to catch prior-layer errors — ONLY for the
        // long sequences (>4096) where the historical crash occurred. For
        // normal-size requests this unconditional full-device drain serialized
        // the per-request varlen Phase-1 loop (4 reqs × 30 SSM layers = 120
        // stalls/forward), collapsing batched prefill 3-4×. Gate it like the
        // sibling syncs below.
        if k > 4096 {
            tracing::info!("ssm phase1 ENTRY: k={k} h={h} qkvz={qkvz_size}");
            ctx.gpu.synchronize(stream).map_err(|e| {
                anyhow::anyhow!("ssm phase1 ENTRY: stream broken BEFORE we start (M={k}): {e}")
            })?;
        }

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

        // ── 2+3. QKVZ GEMM + deinterleave ──
        // Route through the SHARED single-stream dispatch `prefill_qkvz_proj`,
        // which tries CUTLASS-NVFP4 (from nvfp4_t or the fp8-packed weight)
        // first and uses the tensor-core pipelined BF16 kernel for the dense
        // fallback. The batched path previously inlined a chain that LACKED the
        // CUTLASS-NVFP4 branches, so co-dispatched requests fell back to the
        // scalar `dense_gemm` — nsys showed that scalar GEMM at ~60% of batched-
        // prefill GPU time (19–31 ms/call) while single-stream requests, which
        // already used this helper, ran NVFP4 and were ~5× faster.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        self.prefill_qkvz_proj(
            normed,
            deinterleaved,
            k,
            qkvz_size,
            h,
            nk,
            kd,
            vpg,
            vd,
            ctx,
            stream,
        )?;
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
        // One pitched copy instead of a per-token D2D loop (num_tokens launches).
        ctx.gpu.copy_d2d_2d_async(
            z_src_base,
            qkvz_size * bf16, // src row pitch
            z_dst_base,
            value_dim * bf16, // dst row pitch
            z_elem_bytes,     // width per row
            num_tokens,       // rows
            stream,
        )?;

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // M1: large-M batched Phase-1. The RMS/QKVZ/BA-gates GEMMs are
    // token-parallel (no recurrent state), so the co-dispatch path runs them
    // ONCE over all stacked tokens (M = Σ req_len) instead of once per request.
    // Large M is where the GB10 tensor cores get efficient (this is how vLLM
    // reaches its prefill throughput). Only conv1d depends on per-request
    // conv_state, so the caller loops `prefill_phase1_conv1d_one` per request
    // between the proj (below) and the L2 norm (`prefill_phase1_l2_batched`).
    //
    // Buffers (norm_output/ssm_deinterleaved/ssm_gates) are the prefill scratch,
    // sized for max-prefill tokens, so they hold all `total_tokens` at once.
    // `deinterleaved` is LEFT populated in scratch for the caller's conv1d loop.
    // ─────────────────────────────────────────────────────────────────────

    /// Steps 1-5 + gate/Z staging over the full stacked batch (one large-M GEMM
    /// each). Writes gdn_bufs.gate_beta and gdn_bufs.z; leaves the QKVZ result in
    /// `ctx.buffers.ssm_deinterleaved()` for the per-request conv1d tail.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_proj_batched_inner(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = total_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let qkvz_size = ctx.config.ssm_qkvz_size();

        // 1. RMS norm + residual over ALL tokens.
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden_stacked,
            &self.input_norm,
            normed,
            residual_stacked,
            k,
            h as u32,
            eps,
            stream,
        )?;

        // 2+3. QKVZ projection (large-M GEMM) + deinterleave → ssm_deinterleaved.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        self.prefill_qkvz_proj(
            normed,
            deinterleaved,
            k,
            qkvz_size,
            h,
            nk,
            kd,
            vpg,
            vd,
            ctx,
            stream,
        )?;

        // 4+5. BA GEMM + GDN gates over ALL tokens → ssm_gates, then copy to
        // gdn_bufs.gate_beta in one contiguous D2D ([total, 2*nv] FP32 both).
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
        ctx.gpu.copy_d2d_async(
            gates_buf,
            gdn_bufs.gate_beta,
            total_tokens * gate_stride * fp32,
            stream,
        )?;

        // Z gate: strided copy from deinterleaved (stride qkvz_size) → gdn_bufs.z
        // (stride value_dim) over all tokens.
        let z_src_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        let z_elem_bytes = value_dim * bf16;
        // One pitched copy instead of a per-token D2D loop (total_tokens launches).
        ctx.gpu.copy_d2d_2d_async(
            z_src_base,
            qkvz_size * bf16, // src row pitch
            gdn_bufs.z,
            value_dim * bf16, // dst row pitch
            z_elem_bytes,     // width per row
            total_tokens,     // rows
            stream,
        )?;
        Ok(())
    }

    /// Per-request conv1d tail: reads the request's slice of the stacked
    /// `ssm_deinterleaved` scratch (filled by `prefill_phase1_proj_batched`),
    /// advances its own `conv_state`, and writes conv output directly into
    /// `gdn_bufs.qkv` at the request's global token offset.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_conv1d_one_inner(
        &self,
        state: &mut dyn LayerState,
        token_offset: usize,
        len: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let src = deinterleaved.offset(token_offset * qkvz_size * bf16);
        let dst = gdn_bufs.qkv.offset(token_offset * conv_dim * bf16);
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            ssm_state.conv_state,
            src,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            dst,
            conv_dim as u32,
            d_conv as u32,
            len as u32,
            qkvz_size as u32,
            conv_dim as u32,
            stream,
        )
    }

    /// Batched L2 norm on Q,K over the full stacked QKV buffer (gdn_bufs.qkv),
    /// after all per-request conv1d tails have written their slices.
    pub(super) fn prefill_phase1_l2_batched_inner(
        &self,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let conv_dim = nk * kd * 2 + nv * vd;
        ops::l2_norm(
            ctx.gpu,
            self.l2_norm_k,
            gdn_bufs.qkv,
            (nk * 2) as u32,
            kd as u32,
            1e-6,
            total_tokens as u32,
            conv_dim as u32,
            stream,
        )
    }
}
