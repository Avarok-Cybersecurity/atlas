// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::prefill.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner(
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
        _kv_write_start: usize, // SSM layers ignore — recurrent state requires all tokens
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        // Per-SSM-layer-prefill counter — used by ATLAS_GDN_DUMP hooks
        // to attribute a captured intermediate to a specific SSM layer
        // index. The N SSM layers in the model are called in order
        // during one prefill, so layer N-1 sees counter == N-1.
        let ssm_layer_idx =
            super::debug::SSM_LAYER_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd; // 2048
        let value_dim = nv * vd; // 4096
        let conv_dim = key_dim * 2 + value_dim; // 8192
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size(); // 12288

        // Profiling helper: sync + timestamp when ATLAS_PROFILE=1
        macro_rules! prof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  SSM prefill [{}] N={}: {}µs", $label, k, elapsed);
                    }
                }
            };
        }
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Diagnostic: sync at entry to catch prior-layer errors
        if k > 4096 {
            tracing::info!("SSM prefill ENTRY: k={k} h={h}");
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("SSM prefill ENTRY: stream broken (k={k}): {e}"))?;
        }

        // ATLAS_GDN_DUMP hook #0a: pre-input-norm hidden state for THIS
        // layer (= last layer's output + residual). If this matches HF
        // byte-perfectly while gnorm doesn't, drift originates INSIDE
        // the current layer's compute (norm/qkv/conv/recur/gnorm).
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            hidden,
            (num_tokens - 1) * h * fp32,
            h,
            ssm_layer_idx,
            "pre_norm",
            &super::debug::DUMP_CONV,
            stream,
        )?;

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
        // ATLAS_GDN_DUMP hook #0b: post-input-norm (input to in_proj_qkv).
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            normed,
            (num_tokens - 1) * h * 2,
            h,
            ssm_layer_idx,
            "post_norm",
            &super::debug::DUMP_L2,
            stream,
        )?;
        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("SSM prefill: SYNC after rms_norm (k={k}): {e}"))?;
        }

        prof!("rms_norm_residual", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 2+3. QKVZ GEMM (+ deinterleave if needed) ──
        // Dispatch hoisted to trait_prefill_proj.rs to keep this file under
        // the 500 LoC cap; behavior identical.
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
        // ATLAS_GDN_DUMP hook #0c: post-qkvz GEMM (deinterleaved input
        // to conv1d). qkvz_size = key_dim*2 + value_dim*2 = 12288 for A3B
        // (Q+K+V+Z, head-major within each segment). Compare against HF's
        // in_proj_qkv output (only 8192 — Q+K+V; HF has separate in_proj_z).
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            deinterleaved,
            (num_tokens - 1) * qkvz_size * bf16,
            qkvz_size,
            ssm_layer_idx,
            "post_qkvz",
            &super::debug::DUMP_GDN,
            stream,
        )?;

        prof!("qkvz_gemm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 4+5. Fused BA GEMM + GDN gates (token-parallel) ──
        // Replaces dense_gemm([M,K]×[N,K]) + compute_gdn_gates.
        // Vectorized uint4 loads, warp shuffle reduction, inline sigmoid/exp.
        // gate_out layout: [gate(nv), beta(nv)] per token, gate_stride = 2*nv FP32.
        let ba_size = ctx.config.ssm_ba_size(); // 64
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2; // FP32 elements per token
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
        prof!("ba+gates", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 6. Batched conv1d for all N tokens (sequential per-channel in registers) ──
        // Reuse ssm_qkvz buffer for conv output (safe: deinterleave is done)
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();

        // Input: deinterleaved [N, qkvz_size], output: conv_out [N, conv_dim]
        // Conv1d processes QKV channels (first conv_dim of each token's qkvz_size)
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
        // ATLAS_GDN_DUMP hook #1: post-conv1d (post-silu, applied inside
        // the kernel). Last-token slice, flat [conv_dim] bf16. Layer
        // index from SSM_LAYER_CALL_COUNTER; latched by per-layer
        // AtomicBool so each (layer_idx, stage) dumps at most once.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            conv_out_buf,
            (num_tokens - 1) * conv_dim * bf16,
            conv_dim,
            ssm_layer_idx,
            "conv",
            &super::debug::DUMP_CONV,
            stream,
        )?;
        prof!("conv1d", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 7. Batched L2 norm on Q,K for all N tokens ──
        // Q,K are the first 2*key_dim elements of each token's conv_out.
        // Stride between tokens in conv_out = conv_dim.
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
        // ATLAS_GDN_DUMP hook #2: post-L2 norm on q,k (v unchanged).
        // Same buffer/shape as the conv dump — l2_norm operates in
        // place on the q,k segments of conv_out_buf.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            conv_out_buf,
            (num_tokens - 1) * conv_dim * bf16,
            conv_dim,
            ssm_layer_idx,
            "l2",
            &super::debug::DUMP_L2,
            stream,
        )?;
        prof!("l2_norm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 8. GDN prefill via WY4-persistent kernel ──
        // Processes 4 tokens per iteration with WY algebraic correction, keeping
        // H state in shared memory for the entire sequence. 4× fewer sequential
        // state multiplications vs single-token kernel, preventing precision
        // drift at long context (28K+). Falls back to single-token persistent,
        // then split4 for unsupported configurations.
        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);

        // Recurrence kernel dispatch hoisted to trait_prefill_recur.rs to
        // keep this file under the 500 LoC cap; behavior identical.
        self.prefill_gdn_recurrence(
            ssm_state.h_state,
            q_ptr,
            k_ptr,
            v_ptr,
            gates_buf,
            gdn_out_buf,
            k,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            ctx,
            stream,
        )?;

        // ATLAS_GDN_DUMP hook #3: post-GDN recurrence (pre-gnorm,
        // value-space). gdn_out_buf is [num_tokens, value_dim] bf16
        // row-major; dump the last token's value_dim slice.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            gdn_out_buf,
            (num_tokens - 1) * value_dim * bf16,
            value_dim,
            ssm_layer_idx,
            "gdn",
            &super::debug::DUMP_GDN,
            stream,
        )?;
        prof!("gdn_prefill", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 9. Gated RMS norm (batched: all tokens × heads in one launch) ──
        let normed_out_buf = conv_out_buf;
        let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        ops::gated_rms_norm_prefill(
            ctx.gpu,
            self.gated_rms_norm_prefill_k,
            gdn_out_buf,
            z_base,
            &self.ssm.norm,
            normed_out_buf,
            nv as u32,
            vd as u32,
            eps,
            k,
            value_dim as u32,
            qkvz_size as u32,
            stream,
        )?;
        // ATLAS_GDN_DUMP hook #4: post-gated-RMSNorm. Downstream
        // `prefill_out_proj_dispatch` (line ~411) consumes this buffer
        // as `[num_tokens, value_dim]`, so the row stride is value_dim
        // (= nv*vd = 4096 for A3B). normed_out_buf aliases conv_out_buf
        // (in-place reuse — conv_out is dead by this point).
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            normed_out_buf,
            (num_tokens - 1) * value_dim * bf16,
            value_dim,
            ssm_layer_idx,
            "gnorm",
            &super::debug::DUMP_GNORM,
            stream,
        )?;
        prof!("gated_rms_norm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 10. Output projection GEMM: [N, 4096] × [4096, 2048] → [N, 2048] ──
        let out_proj_buf = ctx.buffers.moe_output();
        self.prefill_out_proj_dispatch(ctx, normed_out_buf, out_proj_buf, k, h, value_dim, stream)?;
        // GDN HeadParallel: reduce the row-parallel partial out_proj across TP
        // ranks (num_tokens × h BF16) before the residual add. No-op at tp=1.
        self.ssm_tp_all_reduce(out_proj_buf, num_tokens, ctx, stream)?;
        // ATLAS_GDN_DUMP hook: SSM out_proj output — drift attribution.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            out_proj_buf,
            (num_tokens - 1) * h * bf16,
            h,
            ssm_layer_idx,
            "out_proj",
            &super::debug::DUMP_GDN,
            stream,
        )?;

        prof!("out_proj", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ATLAS_DUMP_EXPERT_IDS=1 also dumps the residual_add_rms_norm
        // INPUTS (hidden + out_proj_buf separately) for last token.
        // This isolates whether the gate-input direction-divergence vs HF
        // comes from (a) hidden being corrupted, (b) out_proj_buf differing,
        // or (c) the residual_add_rms_norm kernel itself computing differently.
        if std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() == Some("1") {
            ctx.gpu.synchronize(stream)?;
            let offset = (num_tokens - 1) * h * 2;
            // Read hidden
            let mut buf_h = vec![0u8; h * 2];
            let _ = ctx.gpu.copy_d2h(hidden.offset(offset), &mut buf_h);
            let v_h: Vec<f32> = buf_h
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let n_h = v_h.iter().map(|x| x * x).sum::<f32>().sqrt();
            // Read out_proj_buf
            let mut buf_o = vec![0u8; h * 2];
            let _ = ctx.gpu.copy_d2h(out_proj_buf.offset(offset), &mut buf_o);
            let v_o: Vec<f32> = buf_o
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let n_o = v_o.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!(
                "ATLAS_PRENORM_HIDDEN last_tok: |x|={:.4} first5={:?}",
                n_h,
                &v_h[..5]
            );
            tracing::info!(
                "ATLAS_PRENORM_OUTPROJ last_tok: |x|={:.4} first5={:?}",
                n_o,
                &v_o[..5]
            );
            // Also log the SUM manually
            let v_sum: Vec<f32> = v_h.iter().zip(v_o.iter()).map(|(a, b)| a + b).collect();
            let n_sum = v_sum.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!(
                "ATLAS_PRENORM_SUM (hidden+out_proj): |x|={:.4} first5={:?}",
                n_sum,
                &v_sum[..5]
            );
        }

        // ── 11. Batched residual + post-norm + MoE ──
        // residual_add_rms_norm already supports num_tokens via grid.x
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj_buf,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            num_tokens as u32,
            h as u32,
            eps,
            stream,
        )?;
        // Batched MoE: 5 kernel launches for all N tokens
        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
        // ATLAS_GDN_DUMP hook: MoE output — KEY drift attribution test.
        // If this matches HF byte-perfectly, MoE quant is not the source.
        // If it drifts, MoE expert quantization is the confirmed cause.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            ctx.buffers.moe_output(),
            (num_tokens - 1) * h * bf16,
            h,
            ssm_layer_idx,
            "moe_out",
            &super::debug::DUMP_GNORM,
            stream,
        )?;
        // Batch residual_add: moe_output[N*H] → hidden[N*H]
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            ctx.buffers.moe_output(),
            (num_tokens * h) as u32,
            stream,
        )?;

        prof!("moe_ffn", t0);

        Ok(())
    }
}
