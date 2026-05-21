// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// Multi-sequence decode (one token per sequence, independent SSM state).
    ///
    /// SSM decode has two kinds of work per layer:
    /// - **State-free outer ops** (input/output RMS norm + residual) —
    ///   token-independent; safe to batch across N sequences in a single
    ///   kernel launch.
    /// - **State-bearing inner ops** (`conv1d_update`, `gdn_decode`) — each
    ///   carries its own per-sequence recurrent state, so the kernels can't
    ///   trivially fan over multiple states in one grid.
    ///
    /// We batch the outer ops (`rms_norm_residual` + the
    /// `residual_add_rms_norm` after SSM) and run the inner ops in a
    /// per-sequence loop using disjoint slices of the per-pass scratch arena
    /// (sized for `max_batch_tokens` in `BufferSizes::from_config`).
    ///
    /// An earlier batched attempt (#6) wrote conv1d output to
    /// `ctx.buffers.attn_output()`, which is only sized for
    /// `m * mamba2_d_inner * bf16` — half of what `n * conv_dim * bf16`
    /// requires on Qwen3.5-A3B. The out-of-bounds writes were what produced
    /// the multilingual gibberish. The corrected layout writes conv and GDN
    /// output into the much larger `ssm_conv_out_f32`
    /// (`m * ssm_qkvz_size * 4`), mirroring the single-seq `ssm_forward`
    /// layout per slice.
    ///
    /// MoE forward writes to `moe_output[0..h]` and is therefore interleaved
    /// with the per-seq `residual_add` (same pattern as
    /// [`decode_batched_inner`] for non-fused K).
    ///
    /// If the FP32 conv1d / GDN / gated-RMS kernels aren't loaded on the
    /// active backend (e.g. Metal), fall back to per-sequence
    /// [`decode_inner`] — the BF16 path stores GDN output in `attn_output`,
    /// which the safe layout above can't accommodate.
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let use_batched = self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0;
        if !use_batched {
            return self.decode_multi_seq_per_seq_fallback(
                hidden,
                residual,
                num_seqs,
                states,
                kv_cache,
                seq_lens,
                block_tables,
                ctx,
                stream,
            );
        }

        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let n = num_seqs as u32;

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
        let ba_size = ctx.config.ssm_ba_size();

        // ── 1. Batched RMS norm + residual across all N sequences ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 2-7. Per-seq SSM forward (state-bearing) ──
        //
        // Buffer layout per seq i:
        //   ssm_deinterleaved[i * qkvz_size * bf16 ..]            — [Q|K|V|Z]
        //   ssm_gates[i * nv * 2 * fp32 ..]                       — [gate(nv) | beta(nv)]
        //   ssm_conv_out_f32[i * qkvz_size * fp32 ..]             — conv1d FP32 out
        //     + (key_dim*2 + value_dim)*fp32                      — GDN FP32 out (in-slice)
        //   ssm_qkvz[i * qkvz_size * bf16 ..]                     — normed_out (gated_rms)
        //   moe_output[i * h * bf16 ..]                           — SSM output projection
        //
        // All strides match the per-token natural stride of the respective
        // arena buffer (see `BufferSizes::from_config`), so per-seq writes
        // stay in-bounds for any `num_seqs ≤ max_batch_tokens`.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let gates_buf = ctx.buffers.ssm_gates();
        let conv_out_f32 = ctx.buffers.ssm_conv_out_f32();
        let normed_out_buf = ctx.buffers.ssm_qkvz();
        let ssm_out_buf = ctx.buffers.moe_output();
        let gdn_local_offset = (key_dim * 2 + value_dim) * fp32;

        for i in 0..num_seqs {
            let normed_i = normed.offset(i * h * bf16);
            let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
            let gate_i = gates_buf.offset(i * nv * 2 * fp32);
            let beta_i = gate_i.offset(nv * fp32);
            let conv_out_i = conv_out_f32.offset(i * qkvz_size * fp32);
            let gdn_out_i = conv_out_i.offset(gdn_local_offset);
            let normed_out_i = normed_out_buf.offset(i * qkvz_size * bf16);
            let ssm_out_i = ssm_out_buf.offset(i * h * bf16);

            let ssm_state = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

            // ── 2. QKVZ projection (+ deinterleave if needed) ──
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
            } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_qkvz(
                    ctx.gpu,
                    self.w4a16_gemv_qkvz_k,
                    normed_i,
                    nvfp4,
                    deint_i,
                    qkvz_size as u32,
                    h as u32,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            } else {
                // Dense fallback: interleaved GEMV then in-place deinterleave.
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
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    deint_i,
                    deint_i,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }

            // ── 3. Fused BA projection + GDN gates ──
            ops::dense_gemv_ba_gates(
                ctx.gpu,
                self.ba_gates_k,
                normed_i,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gate_i,
                beta_i,
                ba_size as u32,
                h as u32,
                vpg as u32,
                stream,
            )?;

            // ── 4. Conv1d update + SiLU + L2 norm (FP32 output) ──
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_f32_k,
                ssm_state.conv_state,
                deint_i,
                &self.ssm.conv1d,
                conv_out_i,
                conv_dim,
                d_conv,
                1,
                qk_channels,
                kd as u32,
                1e-6,
                stream,
            )?;

            // ── 5. GDN decode (FP32 state + output) ──
            let q_conv = conv_out_i;
            let k_conv = conv_out_i.offset(key_dim * fp32);
            let v_conv = conv_out_i.offset(key_dim * 2 * fp32);
            ops::gdn_decode(
                ctx.gpu,
                self.gdn_f32_k,
                ssm_state.h_state,
                q_conv,
                k_conv,
                v_conv,
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

            // ── 6. Gated RMS norm (FP32 GDN input → BF16 normed output) ──
            let z_ptr = deint_i.offset((key_dim * 2 + value_dim) * bf16);
            ops::gated_rms_norm(
                ctx.gpu,
                self.gated_rms_norm_f32_k,
                gdn_out_i,
                z_ptr,
                &self.ssm.norm,
                normed_out_i,
                nv as u32,
                vd as u32,
                vd as u32,
                eps,
                vd as u32,
                stream,
            )?;

            // ── 7. Output projection [value_dim → hidden_size] ──
            if let Some(ref dense_out) = self.out_proj_dense {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_out_i,
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
                    normed_out_i,
                    &self.ssm.out_proj,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        }

        // ── 8. Batched residual + post-attn RMS norm across all N seqs ──
        // Reads SSM output from `ssm_out_buf[0..n*h*bf16]` (contiguous,
        // written in step 7) and produces post-norm input for the MoE.
        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out_buf,
            &self.post_attn_norm,
            normed2,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 9. Per-seq MoE forward + residual_add (interleaved) ──
        //
        // MoE.forward writes its output into `moe_output[0..h*bf16]`
        // regardless of which sequence it's processing, so we add it back
        // into the hidden slot for seq i before invoking MoE for seq i+1.
        // Same pattern as `decode_batched_inner` for non-fused K.
        for i in 0..num_seqs {
            let normed2_i = normed2.offset(i * h * bf16);
            let moe_out = self.ffn.forward(normed2_i, ctx, stream)?;
            let hidden_i = hidden.offset(i * h * residual_elem);
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

    /// Per-sequence single-decode fallback. Used when FP32 conv1d / GDN
    /// kernels aren't loaded (e.g. Metal backend) — the BF16 path uses
    /// `attn_output` for GDN output, which the batched layout above can't
    /// safely accommodate for N>1 without overflow.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_per_seq_fallback<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        let mut stub_disk = Vec::<u32>::new();
        let mut stub_last_offloaded = Vec::<u32>::new();
        for i in 0..num_seqs {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let residual_i = residual.offset(i * h * residual_elem);
            self.decode(
                hidden_i,
                residual_i,
                states[i],
                kv_cache,
                seq_lens[i],
                &mut block_tables[i].clone(),
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }
}
