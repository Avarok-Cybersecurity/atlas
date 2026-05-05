// SPDX-License-Identifier: AGPL-3.0-only

//! N-token prefill body for [`super::super::Qwen3AttentionLayer`],
//! split out of the trait impl for file-size budget. Trait impl delegates
//! to [`Qwen3AttentionLayer::prefill_inner`].

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::diag_norm;
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;

        // ── 1. RMS norm + residual for N tokens ──
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
        )
        .map_err(|e| anyhow::anyhow!("rms_norm_residual failed: {e}"))?;

        // DIAGNOSTIC: dump norms for L0 and L35 of Mistral
        let is_mistral_diag = ctx.profile
            && ctx.config.model_type == "mistral"
            && (self.attn_layer_idx == 0 || self.attn_layer_idx == 35);
        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{} hidden_in", self.attn_layer_idx),
            );
            diag_norm(
                ctx.gpu,
                normed,
                h,
                stream,
                &format!("L{} normed", self.attn_layer_idx),
            );
        }

        // ── 2. Attention ──
        let attn_out = if seq_len_start == 0 {
            // Chunk 0 (or non-chunked): Flash Attention on contiguous Q/K/V.
            self.prefill_attention_with_cache_skip(
                normed,
                num_tokens,
                kv_write_start,
                kv_cache,
                ctx,
                stream,
            )?
        } else {
            // Chunk 1+: GEMM-batched Q/K/V + per-token paged decode attention.
            // New Q tokens must attend to ALL prior K/V stored in paged cache.
            // Phase 6.2.b — block_table/disk_block_ids/disk_last_offloaded_per_layer
            // are threaded so the function can route through HSS when --high-speed-swap
            // is engaged on a BF16 layer.
            self.prefill_attention_paged(
                normed,
                num_tokens,
                seq_len_start,
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?
        };

        // (Two earlier attempts to overlap the lazy down transpose on
        // `prefill_stream` regressed cold TTFT by 30 % — both when
        // overlapping with compute-bound MoE GEMMs AND when overlapping
        // with the TP allreduce window. GB10 either has SM-contention
        // costs from the second stream or stream-event sync overhead
        // that exceeds the transpose savings. Keeping the synchronous
        // path in `forward_prefill` for now.)
        // TP all-reduce on attn_out after o_proj (Megatron row-parallel
        // pattern). When tp_world_size==1 this is a no-op. The o_proj GEMM
        // produced this rank's partial output on the full hidden dim; the
        // reduction across TP ranks gives the full attention output ready
        // for the residual add.
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = num_tokens * h * 2; // BF16
            let _t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
            if let Some(t0) = _t0 {
                ctx.gpu.synchronize(stream)?;
                tracing::info!(
                    "  TP allreduce (attn out) N={} L{:02}: {}µs",
                    num_tokens,
                    self.attn_layer_idx,
                    t0.elapsed().as_micros(),
                );
            }
        }

        // Phase 6.2.a: after prefill writes K/V to the cache, mirror every
        // new block to disk so future decode steps can read them via the
        // orchestrator. Sliding-window eviction is NOT triggered here —
        // prefill grows HBM monotonically; the cap kicks in during decode.
        // For prefill writes that exceed cache_blocks_per_seq in one shot
        // (long single-chunk prompts), the user must size cache_blocks_per_seq
        // to fit the prefill. Phase 6.2.b will route chunked-prefill reads
        // through the orchestrator and remove this constraint.
        if self.high_speed_swap_engaged(kv_cache) {
            let nq = self
                .num_q_heads_override
                .unwrap_or(ctx.config.num_attention_heads) as u32;
            let nkv = self
                .num_kv_heads_override
                .unwrap_or(ctx.config.num_key_value_heads) as u32;
            let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
            let bs = kv_cache.block_size();
            let _ = nq; // silence unused
            self.high_speed_swap_offload_new_blocks(
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
                nkv,
                hd,
                bs,
            )?;
            // Touch nq once to keep the existing variable binding's compile error away.
            let _ = nq;
        }

        // DIAGNOSTIC: attention output for L0 and L35
        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                attn_out,
                h,
                stream,
                &format!("L{} attn_out", self.attn_layer_idx),
            );
        }

        // ── 3. Post-attention norm (Gemma-4: normalize attn output before residual add) ──
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                attn_out,
                post_norm,
                attn_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }

        // ── 4. Batched residual + pre-FFN norm + FFN ──
        if self.ffn.is_none() {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                attn_out,
                (num_tokens * h) as u32,
                stream,
            )?;
            return Ok(());
        }

        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            attn_out,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            n,
            h as u32,
            eps,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("residual_add_rms_norm failed: n={n} h={h}: {e}"))?;

        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)
            .map_err(|e| anyhow::anyhow!("ffn.forward_prefill failed: {e}"))?;

        let dense_out = ctx.buffers.moe_output();

        // DIAGNOSTIC: MoE output for L0 and L35
        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                dense_out,
                h,
                stream,
                &format!("L{} moe_out", self.attn_layer_idx),
            );
        }

        // Gemma-4 26B MoE dual FFN (prefill): match HF Gemma4TextDecoderLayer.forward
        if let (Some(moe_ffn), Some(_pre_norm), Some(post_norm), Some(dense_norm)) = (
            &self.moe_ffn,
            &self.pre_moe_norm,
            &self.post_moe_out_norm,
            &self.post_dense_ffn_norm,
        ) {
            // 1. Norm dense MLP output with post_feedforward_layernorm_1
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                dense_out,
                dense_norm,
                dense_out,
                n,
                h as u32,
                eps,
                stream,
            )?;

            // 2. Save normed dense_out to scratch
            let scratch = ctx.buffers.attn_output();
            let nbytes = num_tokens * h * 2;
            ctx.gpu.copy_d2d_async(dense_out, scratch, nbytes, stream)?;

            // 3. MoE path: pass raw residual (router has internal norm+scale)
            moe_ffn
                .forward_prefill(hidden, num_tokens, ctx, stream)
                .map_err(|e| anyhow::anyhow!("moe_ffn.forward_prefill failed: {e}"))?;
            let moe_out = ctx.buffers.moe_output();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                moe_out,
                post_norm,
                moe_out,
                n,
                h as u32,
                eps,
                stream,
            )?;

            // 4. Combine: dense_normed + moe_normed → moe_out
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                moe_out,
                scratch,
                (num_tokens * h) as u32,
                stream,
            )?;

            // 5. post_feedforward_layernorm on combined
            if let Some(ref combined_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    moe_out,
                    combined_norm,
                    moe_out,
                    n,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            // 6. Residual add
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (num_tokens * h) as u32,
                stream,
            )?;
        } else {
            // Non-MoE: post_ffn_out_norm on dense output, then residual add
            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    dense_out,
                    post_norm,
                    dense_out,
                    n,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                (num_tokens * h) as u32,
                stream,
            )
            .map_err(|e| anyhow::anyhow!("residual_add failed: n={num_tokens} h={h}: {e}"))?;
        }

        // Gemma-4: hidden *= layer_scalar at end of layer (applied to ALL tokens)
        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(
                ctx.gpu,
                hidden,
                num_tokens * h,
                scalar,
                stream,
                ctx.config.use_fp32_residual(),
            )?;
        }

        // DIAGNOSTIC: residual after L0 and L35
        if is_mistral_diag {
            diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{} residual", self.attn_layer_idx),
            );
        }

        Ok(())
    }
}
