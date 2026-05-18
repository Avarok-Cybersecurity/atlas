// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// Multi-sequence decode: falls back to per-sequence single decode.
    ///
    /// The batched SSM path had buffer aliasing bugs (#6) where shared scratch
    /// buffers (conv_out, gdn_out, moe_output) corrupted across sequences,
    /// producing gibberish (Chinese/multilingual tokens). Instead of debugging
    /// every buffer interaction, we delegate to the proven single-sequence
    /// decode path which has no aliasing issues.
    ///
    /// Performance impact: negligible — SSM decode is memory-bandwidth-bound
    /// and per-sequence GEMV weights stay in L2 cache across iterations.
    #[allow(unreachable_code, unused_variables)]
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;

        // CONCURRENT-DECODE BUG FIX: per-seq stride must match the ACTUAL
        // hidden/residual element size, not always FP32 (4 bytes). When
        // `use_fp32_residual()` is false (BF16 hidden — the default for
        // GB10 LPDDR5X bandwidth-limited systems via HARDWARE.toml), the
        // hardcoded `i * h * 4` skipped to position 2 of the BF16 buffer
        // for `i=1`, leaving seq-1's actual position-1 slice UNTOUCHED by
        // every SSM layer. Result: seq 1 only got modifications from the
        // attention layers (which use the correct n>=2 batched indexing
        // internally), producing the position-specific gibberish that
        // reproduced even with identical prompts. Use the same `fp32`
        // bytes-per-element computation the dispatcher uses at
        // model.rs:4250.
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        // Delegate to per-sequence single decode (proven correct, no buffer aliasing).
        let mut _stub_disk = Vec::<u32>::new();
        let mut _stub_last_offloaded = Vec::<u32>::new();
        for i in 0..num_seqs {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let residual_i = residual.offset(i * h * residual_elem);
            self.decode(
                hidden_i,
                residual_i,
                states[i],
                _kv_cache,
                _seq_lens[i],
                &mut _block_tables[i].clone(),
                &mut _stub_disk,
                &mut _stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        return Ok(());

        // ── Original batched path (disabled — buffer aliasing bug #6) ──
        let eps = ctx.config.rms_norm_eps as f32;
        let fp32 = 4usize;
        let n = num_seqs;

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
        let ba_size = ctx.config.ssm_ba_size();

        // ── 1. RMS norm + residual for N tokens ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // ── 2-9. Per-sequence SSM forward + projections ──
        // GEMV projections are sequential (weights cached in L2 after first call).
        // Conv1d, GDN are per-sequence (independent recurrent state).
        let qkvz_out = ctx.buffers.ssm_qkvz();
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let gates_buf = ctx.buffers.ssm_gates();
        let conv_out_buf = ctx.buffers.attn_output(); // reuse
        let gdn_out_buf = ctx.buffers.qkv_output(); // reuse for GDN output

        for i in 0..n {
            let normed_i = normed.offset(i * h * bf16);
            let ssm_state = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

            // QKVZ projection: GEMV (sequential writes directly to deinterleaved)
            let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
            if self.sequential_qkvz {
                if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed_i,
                        nvfp4,
                        deint_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed_i,
                        &self.ssm.in_proj_qkvz,
                        deint_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            } else {
                let qkvz_i = qkvz_out.offset(i * qkvz_size * bf16);
                if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed_i,
                        nvfp4,
                        qkvz_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed_i,
                        &self.ssm.in_proj_qkvz,
                        qkvz_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    qkvz_i,
                    deint_i,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }

            // BA projection + GDN gates
            let ba_out = ctx.buffers.ssm_ba().offset(i * ba_size * bf16);
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed_i,
                &self.ssm.in_proj_ba,
                ba_out,
                ba_size as u32,
                h as u32,
                stream,
            )?;
            let gate_beta_stride = nv * 2 * fp32;
            let gate_i = gates_buf.offset(i * gate_beta_stride);
            let beta_i = gates_buf.offset(i * gate_beta_stride + nv * fp32);
            ops::compute_gdn_gates(
                ctx.gpu,
                self.compute_gdn_gates_k,
                ba_out,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gate_i,
                beta_i,
                1,
                nv as u32,
                nk as u32,
                vpg as u32,
                ba_size as u32,
                stream,
            )?;

            // Conv1d update
            let qkv_i = deint_i;
            let conv_out_i = conv_out_buf.offset(i * conv_dim * bf16);
            ops::conv1d_update(
                ctx.gpu,
                self.conv1d_k,
                ssm_state.conv_state,
                qkv_i,
                &self.ssm.conv1d,
                conv_out_i,
                conv_dim as u32,
                d_conv as u32,
                1,
                stream,
            )?;

            // L2 norm on Q,K
            ops::l2_norm(
                ctx.gpu,
                self.l2_norm_k,
                conv_out_i,
                (nk * 2) as u32,
                kd as u32,
                1e-6,
                1,
                (nk * 2 * kd) as u32,
                stream,
            )?;

            // GDN decode
            let q_i = conv_out_i;
            let k_i = conv_out_i.offset(key_dim * bf16);
            let v_i = conv_out_i.offset(key_dim * 2 * bf16);
            let gdn_out_i = gdn_out_buf.offset(i * value_dim * bf16);
            ops::gdn_decode(
                ctx.gpu,
                self.gdn_k,
                ssm_state.h_state,
                q_i,
                k_i,
                v_i,
                gate_i,
                beta_i,
                gdn_out_i,
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                stream,
            )?;

            // Gated RMS norm
            let z_i = deint_i.offset((key_dim * 2 + value_dim) * bf16);
            let normed_ssm_i = conv_out_i; // reuse
            ops::gated_rms_norm(
                ctx.gpu,
                self.gated_rms_norm_k,
                gdn_out_i,
                z_i,
                &self.ssm.norm,
                normed_ssm_i,
                nv as u32,
                vd as u32,
                vd as u32,
                eps,
                vd as u32,
                stream,
            )?;

            // Output projection: GEMV
            let ssm_out_i = ctx.buffers.moe_output().offset(i * h * bf16);
            if let Some(ref dense_out) = self.out_proj_dense {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_ssm_i,
                    dense_out,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed_ssm_i,
                    &self.ssm.out_proj,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        }

        // ── 10. Residual + post-norm + MoE per-sequence ──
        // Bug #6 fix: copy SSM outputs to a safe buffer before running MoE.
        // `self.ffn.forward()` writes its result to `moe_output[0]`, which would
        // overwrite seq 1's SSM output at `moe_output[h]` if the MoE internally
        // uses the full moe_output region as scratch. By copying SSM outputs to
        // `ssm_deinterleaved` (no longer needed after step 9), we decouple them.
        let ssm_out_safe = ctx.buffers.ssm_deinterleaved(); // reuse, large enough for n*h
        for i in 0..n {
            let src = ctx.buffers.moe_output().offset(i * h * bf16);
            let dst = ssm_out_safe.offset(i * h * bf16);
            ctx.gpu.copy_d2d_async(src, dst, h * bf16, stream)?;
        }
        // STRIDE FIX (mirrors 2026-04-22 fix at lines 47/57): use dynamic
        // residual_elem instead of hardcoded `* 4`. On GB10 hidden states
        // are BF16 (2 bytes), not FP32. Hardcoded `i * h * 4` causes
        // position 1+ in concurrent batched decode to read/write at WRONG
        // offsets, producing either silent gibberish (small N) or CUDA-700
        // illegal memory access (large per-seq offsets exceeding allocated
        // buffer region). See project_batch_decode_corruption.md memory.
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        for i in 0..n {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let ssm_out_i = ssm_out_safe.offset(i * h * bf16);
            let residual_i = residual.offset(i * h * residual_elem);
            let normed2 = ctx.buffers.norm_output().offset(i * h * bf16);
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden_i,
                ssm_out_i,
                &self.post_attn_norm,
                normed2,
                residual_i,
                1,
                h as u32,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden_i,
                moe_out,
                h as u32,
                stream,
            )?;
        }

        Ok(())
    }
}
