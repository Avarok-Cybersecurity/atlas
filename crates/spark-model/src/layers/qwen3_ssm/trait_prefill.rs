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

        // The SSM geometry (`nk`/`kd`/`nv`/`vd`/`conv_dim`/`qkvz_size`) and
        // the `SsmLayerState` downcast moved into `prefill_block` with the
        // body that reads them; only the residual bookkeeping is left here.

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

        let out_proj_buf =
            self.prefill_block(normed, num_tokens, state, ssm_layer_idx, ctx, stream)?;

        // ATLAS_DUMP_EXPERT_IDS=1: also dumps residual_add_rms_norm INPUTS (hidden + out_proj_buf) for drift attribution.
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
