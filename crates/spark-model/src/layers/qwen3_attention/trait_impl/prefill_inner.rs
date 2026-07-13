// SPDX-License-Identifier: AGPL-3.0-only

//! N-token prefill body for [`super::super::Qwen3AttentionLayer`],
//! split out of the trait impl for file-size budget. Trait impl delegates
//! to [`Qwen3AttentionLayer::prefill_inner`].

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::diag_norm;
use crate::layer::{BatchedAttnMetadata, ForwardContext, LayerState};
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
        // Q12 Path B: when Some, the attention compute step uses the
        // batched paged-prefill kernel. Stacked-input semantics apply —
        // `num_tokens` must equal `batched_meta.total_tokens` and the
        // hidden/residual buffers contain N streams' data concatenated
        // at offsets `b * chunk_len * H * dtype`.
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // DeepSeek-V4: Manifold-Constrained Hyper-Connections (mHC).
        if self.hc.is_some() {
            return self.prefill_inner_hc(
                hidden,
                residual,
                num_tokens,
                _state,
                kv_cache,
                seq_len_start,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                kv_write_start,
                batched_meta,
                ctx,
                stream,
            );
        }

        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;
        let bf16 = 2usize;

        // ATLAS_OP_DUMP hook: pre-input-norm hidden state (input to layer).
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                hidden,
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "input_norm_in",
                stream,
            )?;
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
            n,
            h as u32,
            eps,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("rms_norm_residual failed: {e}"))?;
        // ATLAS_OP_DUMP hook: post-input-norm (input to Q/K/V GEMM).
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                normed,
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "input_norm_out",
                stream,
            )?;
        }

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
        // Q12 Path B: batched mode requires seq_len_start > 0 (paged path).
        // NOTE: batched FLASH (reusing the kernel's blockIdx.z batch dim) was
        // tried and FAILED for large prefill chunks — numerically off + ~7x
        // slower (the kernel's batch dim is not compatible with the co-dispatch
        // stacking at large seq_len). So batched chunk-0 still uses paged.
        let allow_batched_first_chunk = batched_meta.is_some()
            && std::env::var("ATLAS_Q12_BATCHED_FIRST_CHUNK")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        if batched_meta.is_some() && seq_len_start == 0 && !allow_batched_first_chunk {
            anyhow::bail!(
                "prefill_inner: batched mode requires seq_len_start > 0 (paged path); \
                 got seq_len_start=0. Caller must fall back to per-stream for this chunk."
            );
        }
        let attn_out = if seq_len_start == 0 && !allow_batched_first_chunk {
            // Chunk 0 (or non-chunked): Flash Attention on contiguous Q/K/V.
            self.prefill_attention_with_cache_skip(
                normed,
                num_tokens,
                kv_write_start,
                kv_cache,
                None,
                ctx,
                stream,
            )?
        } else {
            // Chunk 1+: GEMM-batched Q/K/V + per-token paged decode attention.
            // batched_meta is threaded so prefill_attention_paged uses the
            // batched kernel + block_table_ptrs when set.
            self.prefill_attention_paged(
                normed,
                num_tokens,
                seq_len_start,
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                batched_meta,
                kv_write_start,
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
        if batched_meta.is_some() && self.high_speed_swap_engaged(kv_cache) {
            anyhow::bail!(
                "prefill_inner: batched mode does not support HSS-engaged layers \
                 (layer {}). Caller should fall back to per-stream for this chunk.",
                self.attn_layer_idx
            );
        }
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
                self.rms_norm_w_k,
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
        // ATLAS_OP_DUMP hook: post-attn-norm output = input to MoE FFN.
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                ctx.buffers.norm_output(),
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "post_attn_norm_out",
                stream,
            )?;
        }

        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)
            .map_err(|e| anyhow::anyhow!("ffn.forward_prefill failed: {e}"))?;

        let dense_out = ctx.buffers.moe_output();
        // ATLAS_OP_DUMP hook: MoE output (sum of all weighted expert outputs).
        // For Qwen3.6-A3B at full-attention layers, dense_out holds the
        // post-FFN delta to add to the residual. This is the "MoE block
        // output" comparable against HF `mlp.forward` last-token output.
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                dense_out,
                (num_tokens - 1) * h * bf16,
                h,
                self.attn_layer_idx,
                "moe_out",
                stream,
            )?;
        }

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
                self.rms_norm_w_k,
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
                self.rms_norm_w_k,
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
                    self.rms_norm_w_k,
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
                    self.rms_norm_w_k,
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
            self.apply_layer_scalar(ctx.gpu, hidden, num_tokens * h, scalar, stream)?;
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

    /// HC-enabled prefill inner.  `hc_streams` holds the persistent
    /// multi-stream state; `hidden` is single-stream scratch.
    #[allow(clippy::too_many_arguments)]
    fn prefill_inner_hc(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        let is_first_layer = self.attn_layer_idx == 0;
        let is_last_layer = self.attn_layer_idx + 1 == ctx.config.num_hidden_layers;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let diag_all =
            std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = self.attn_layer_idx == 0 || diag_all;

        if is_first_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                hc_streams,
                n,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── Attention sublayer ──
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            hc.attn.hc_fn,
            hc.attn.hc_scale,
            hc.attn.hc_base,
            hidden,
            post,
            comb,
            n,
            h as u32,
            hc_mult,
            hc.sinkhorn_iters as u32,
            eps,
            hc.hc_eps,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("V4-prefill L{} hc_pre-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                (n as usize) * (hc_mult as usize),
                stream,
                &format!("V4-prefill L{} post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                (n as usize) * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-prefill L{} comb-attn", self.attn_layer_idx),
            );
        }

        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            hidden,
            &self.input_norm,
            normed,
            n,
            h as u32,
            eps,
            stream,
        )?;

        if batched_meta.is_some() && seq_len_start == 0 {
            anyhow::bail!(
                "prefill_inner_hc: batched mode requires seq_len_start > 0; \
                 got seq_len_start=0."
            );
        }
        let attn_out = if seq_len_start == 0 {
            self.prefill_attention_with_cache_skip(
                normed,
                num_tokens,
                kv_write_start,
                kv_cache,
                None, // batched_meta: single-stream (seq_len_start == 0)
                ctx,
                stream,
            )?
        } else {
            self.prefill_attention_paged(
                normed,
                num_tokens,
                seq_len_start,
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                batched_meta,
                kv_write_start,
                ctx,
                stream,
            )?
        };

        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = num_tokens * h * 2;
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
        }

        if batched_meta.is_some() && self.high_speed_swap_engaged(kv_cache) {
            anyhow::bail!(
                "prefill_inner_hc: batched mode does not support HSS-engaged layers \
                 (layer {})",
                self.attn_layer_idx
            );
        }
        if self.high_speed_swap_engaged(kv_cache) {
            let nkv = self
                .num_kv_heads_override
                .unwrap_or(ctx.config.num_key_value_heads) as u32;
            let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
            let bs = kv_cache.block_size();
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
        }

        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                attn_out,
                post_norm,
                attn_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }

        if self.ffn.is_none() {
            ops::hc_post(
                ctx.gpu,
                self.hc_post_k,
                attn_out,
                hc_streams,
                post,
                comb,
                hc_streams,
                n,
                h as u32,
                hc_mult,
                stream,
            )?;
            if is_last_layer && let Some(ref head) = hc.head {
                ops::hc_head(
                    ctx.gpu,
                    self.hc_head_k,
                    hc_streams,
                    head.hc_fn,
                    head.hc_scale,
                    head.hc_base,
                    hidden,
                    n,
                    h as u32,
                    hc_mult,
                    eps,
                    hc.hc_eps,
                    stream,
                )?;
            }
            return Ok(());
        }

        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            attn_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h as u32,
            hc_mult,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-prefill L{} hc_post-attn", self.attn_layer_idx),
            );
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                (n as usize) * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-prefill L{} hc_post-attn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        // ── FFN sublayer ──
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            hc.ffn.hc_fn,
            hc.ffn.hc_scale,
            hc.ffn.hc_base,
            hidden,
            post,
            comb,
            n,
            h as u32,
            hc_mult,
            hc.sinkhorn_iters as u32,
            eps,
            hc.hc_eps,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("V4-prefill L{} hc_pre-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                (n as usize) * (hc_mult as usize),
                stream,
                &format!("V4-prefill L{} post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                (n as usize) * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-prefill L{} comb-ffn", self.attn_layer_idx),
            );
        }

        let normed2 = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            hidden,
            &self.post_attn_norm,
            normed2,
            n,
            h as u32,
            eps,
            stream,
        )?;

        self.ffn
            .forward_prefill(normed2, num_tokens, ctx, stream)
            .map_err(|e| anyhow::anyhow!("ffn.forward_prefill (HC) failed: {e}"))?;

        let dense_out = ctx.buffers.moe_output();

        if let Some(ref post_norm) = self.post_ffn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                dense_out,
                post_norm,
                dense_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }

        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            dense_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h as u32,
            hc_mult,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-prefill L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                (n as usize) * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-prefill L{} hc_post-ffn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        if is_last_layer && let Some(ref head) = hc.head {
            ops::hc_head(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head.hc_fn,
                head.hc_scale,
                head.hc_base,
                hidden,
                n,
                h as u32,
                hc_mult,
                eps,
                hc.hc_eps,
                stream,
            )?;
            if diag_this {
                super::diag_norm(
                    ctx.gpu,
                    hidden,
                    (n as usize) * h,
                    stream,
                    &format!("V4-prefill L{} hc_head", self.attn_layer_idx),
                );
            }
        } else if is_last_layer {
            tracing::warn!(
                "V4-prefill L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        Ok(())
    }
}
