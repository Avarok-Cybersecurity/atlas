// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_attention_paged` — full N-token prefill (MLA + standard).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{BatchedAttnMetadata, ForwardContext};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn prefill_attention_paged(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        kv_cache: &mut PagedKvCache,
        block_table: &Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        // Q12 Path B: when Some, attention compute uses the batched kernel
        // with `block_table_ptrs` from the supplied BatchedAttnMetadata;
        // positions / slot reads switch to the stacked variants from the
        // same struct. When None, the function behaves single-stream
        // (reading from ctx.attn_metadata as before).
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bs = kv_cache.block_size();
        let n = num_tokens as u32;
        let bf16 = 2usize;

        let q_dim = (nq * hd) as usize;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = (nkv * hd) as usize;

        // Q12 Path B: batched mode does not support MLA layers (separate
        // KV layout). Caller must gate this out at the outer dispatch.
        if batched_meta.is_some() && self.mla.is_some() {
            anyhow::bail!(
                "prefill_attention_paged: batched_meta with MLA layer is not supported \
                 (layer {}). Caller must route MLA layers to per-stream.",
                self.attn_layer_idx
            );
        }

        // ── MLA 2-step prefill (reference: HuggingFace modeling_mistral4.py) ──
        if self.mla.is_some() {
            let args = super::paged_mla::MlaPrefillArgs {
                normed,
                num_tokens,
                n,
                h,
                nq,
                nkv,
                hd,
                kv_dim,
                eps,
                bf16,
                bs: bs as u32,
                stream,
            };
            return self.prefill_attention_paged_mla(kv_cache, ctx, &args);
        }

        // ── Standard Q/K/V projection (non-MLA models) ──
        if self.mla.is_none() {
            self.prefill_attention_paged_qkv(
                normed, n, h, nkv, hd, q_proj_dim, kv_dim, num_tokens, bf16, ctx, stream,
            )?;
        } // end if self.mla.is_none() (standard projection path)

        // ── 4+5. Deinterleave Q/Gate + per-head Q/K RMS norms ──
        // v_contiguous must point at where the V GEMM actually wrote
        // (k_contiguous + kv_dim*n). The previous binding to attn_output()
        // was a stale-buffer bug that corrupted V on chunk-1+ prefill for
        // every model that took this path (root cause of long-context
        // gibberish at 8 k+ contexts).
        let qg_out = ctx.buffers.qkv_output();
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        let q_contiguous = ctx.buffers.ssm_deinterleaved();
        if self.gated && !self.attn.q_norm.weight.is_null() {
            // Fused deinterleave + Q norm: eliminates Q global memory round-trip
            ops::deinterleave_qg_split_qnorm(
                ctx.gpu,
                self.deinterleave_qg_split_qnorm_k,
                qg_out,
                q_contiguous,
                self.attn.q_norm.weight,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                eps,
                stream,
            )?;
        } else if self.gated {
            ops::deinterleave_qg_split(
                ctx.gpu,
                self.deinterleave_qg_split_k,
                qg_out,
                q_contiguous,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                stream,
            )?;
        } else if let Some(mla_ref) = self.mla.as_ref() {
            // MLA: swap Q from [nope|rope] to [rope|nope] per head so RoPE rotates correct dims
            let mla_nope_sz = mla_ref.nope;
            let mla_rope_sz = mla_ref.rope;
            for t in 0..num_tokens {
                for head_idx in 0..nq as usize {
                    let src = qg_out.offset((t * q_dim + head_idx * hd as usize) * bf16);
                    let dst = q_contiguous.offset((t * q_dim + head_idx * hd as usize) * bf16);
                    ctx.gpu.copy_d2d_async(
                        src.offset(mla_nope_sz * bf16),
                        dst,
                        mla_rope_sz * bf16,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        src,
                        dst.offset(mla_rope_sz * bf16),
                        mla_nope_sz * bf16,
                        stream,
                    )?;
                }
            }
        } else {
            ctx.gpu
                .copy_d2d_async(qg_out, q_contiguous, num_tokens * q_dim * bf16, stream)?;
            if let Some(ref q_norm_full) = self.attn.q_norm_full {
                // MiniMax: single RMS over full `[nq*hd]` per token
                // (rows=n, cols=nq*hd). Mistral/DeepSeek MLA models
                // never reach this branch — they early-return above.
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    q_contiguous,
                    q_norm_full,
                    q_contiguous,
                    n,
                    nq * hd,
                    eps,
                    stream,
                )?;
            } else if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    q_contiguous,
                    &self.attn.q_norm,
                    q_contiguous,
                    nq * n,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        if let Some(ref k_norm_full) = self.attn.k_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                k_contiguous,
                k_norm_full,
                k_contiguous,
                n,
                nkv * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.k_norm.weight.is_null() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                k_contiguous,
                &self.attn.k_norm,
                k_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }

        // Gemma-4 v_norm — applied at EVERY layer in HF reference
        // (modeling_gemma4.py:1220 `value_states = self.v_norm(value_states)`
        // with `Gemma4RMSNorm(with_scale=False)`). For full-attention K=V
        // layers, v_contiguous holds raw K (aliased v_proj). For sliding
        // layers, v_contiguous holds V_proj output. Either way normalize
        // with pure RMSNorm via the ones-buffer (Gemma-4's absolute-
        // formula rms_norm kernel: `x * rms * 1.0 = x * rms`). V does NOT
        // receive RoPE.
        if let Some(v_norm_w) = self.v_norm_weight.as_ref() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                v_contiguous,
                v_norm_w,
                v_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }

        // ── 6. RoPE for chunk tokens ──
        // Q12 Path B: in batched mode, ctx.attn_metadata is None — the
        // model dispatcher (prefill_batch_chunk_kernel_batched) sets
        // attn_metadata=None when calling the layer because per-stream
        // metadata is split across the batched_meta device arrays. So
        // we only require ctx.attn_metadata to be Some in single-stream
        // mode (batched_meta = None).
        let meta_for_single = match (batched_meta, ctx.attn_metadata) {
            (Some(_), _) => None,
            (None, Some(m)) => Some(m),
            (None, None) => anyhow::bail!(
                "prefill_attention_paged: single-stream mode requires ctx.attn_metadata"
            ),
        };

        // Resolve positions / slot pointers. When `batched_meta` is set,
        // RoPE / KV-write read from the stacked arrays. When in
        // single-stream mode, fall back to meta.{positions,slot}.
        let bmeta_positions = batched_meta
            .map(|m| m.positions_stacked)
            .or(meta_for_single.map(|m| m.positions))
            .unwrap();
        let bmeta_positions_h = batched_meta
            .map(|m| m.positions_h_stacked)
            .or(meta_for_single.map(|m| m.positions_h))
            .unwrap();
        let bmeta_positions_w = batched_meta
            .map(|m| m.positions_w_stacked)
            .or(meta_for_single.map(|m| m.positions_w))
            .unwrap();
        let bmeta_slot = batched_meta
            .map(|m| m.slot_stacked)
            .or(meta_for_single.map(|m| m.slot))
            .unwrap();
        if self.mla.is_some() {
            // MLA: RoPE already applied inside the MLA block to rope portions only.
        } else if let Some(ref mla) = self.mla {
            // unreachable but keeps the else chain valid
            if !mla.yarn_inv_freq.is_null() {
                ops::rope_yarn(
                    ctx.gpu,
                    self.rope_yarn_k,
                    q_contiguous,
                    k_contiguous,
                    bmeta_positions,
                    n,
                    nq,
                    nkv,
                    hd,
                    ctx.config.rotary_dim() as u32,
                    mla.yarn_inv_freq,
                    ctx.config.rope_theta as f32,
                    stream,
                )?;
            } else {
                ops::rope(
                    ctx.gpu,
                    self.rope_k,
                    q_contiguous,
                    k_contiguous,
                    bmeta_positions,
                    n,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(ctx.config.rotary_dim() as u32),
                    self.rope_theta_override
                        .unwrap_or(ctx.config.rope_theta as f32),
                    stream,
                )?;
            }
        } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
            let rope_angles = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            ops::rope_proportional(
                ctx.gpu,
                self.rope_proportional_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                n,
                nq,
                nkv,
                hd,
                rope_angles,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else if self.mrope_interleaved && self.rope_mrope_interleaved_k.0 != 0 {
            ops::rope_mrope_interleaved(
                ctx.gpu,
                self.rope_mrope_interleaved_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                bmeta_positions_h,
                bmeta_positions_w,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else {
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        }

        // ── 7. Write all K/V to paged cache ──
        // MLA models write compressed cache inside the MLA block above (1 head × 320 dims).
        // Standard models write expanded cache here (nkv heads × hd dims).
        if self.mla.is_none() {
            self.write_kv_cache(
                ctx.gpu,
                k_contiguous,
                v_contiguous,
                kv_cache,
                bmeta_slot,
                n,
                nkv,
                hd,
                bs as u32,
                nkv * hd,
                nkv * hd,
                stream,
                ctx.graph_capture,
            )?;
        }

        // ── 8. Paged Flash Attention for chunk 1+ ── (extracted to paged_attn.rs)
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let kv_len = (seq_len_start + num_tokens) as u32;
        if let Some(bmeta) = batched_meta {
            // Q12 Path B: batched paged-prefill attention. The kernel reads
            // Q/O at per-batch offsets internally via blockIdx.z and uses
            // block_table_ptrs[b] for each stream's paged KV pages.
            let args = super::paged_attn_batched::PagedAttnBatchedArgs {
                q_contiguous,
                attn_out,
                seq_len_start,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                kv_len,
                batched_meta: bmeta,
                stream,
            };
            self.prefill_attention_paged_attn_batched(kv_cache, ctx, &args)?;
        } else {
            // Single-stream path requires meta_for_single (validated above).
            let meta = meta_for_single
                .expect("single-stream mode: meta_for_single guaranteed by validation above");
            let mut args = super::paged_attn::PagedAttnArgs {
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                n,
                seq_len_start,
                num_tokens,
                nq,
                nkv,
                hd,
                bs,
                bf16,
                inv_sqrt_d,
                kv_len,
                meta: &meta,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                stream,
            };
            match self.prefill_attention_paged_attn(kv_cache, ctx, &mut args)? {
                super::paged_attn::PagedAttnOutcome::EarlyReturn(out) => return Ok(out),
                super::paged_attn::PagedAttnOutcome::Continue => {}
            }
        }

        // ── 9. Sigmoid gate × attn_out (gated only) — single batched kernel ──
        if self.gated {
            // Gate data is in qg_out at offset q_dim (after deinterleave_qg_split),
            // with stride q_proj_dim between tokens.
            let gate_base = qg_out.offset(q_dim * bf16);
            ops::sigmoid_gate_mul_batched(
                ctx.gpu,
                self.sigmoid_gate_mul_batched_k,
                attn_out,
                gate_base,
                attn_out,
                nq * hd,
                q_proj_dim as u32,
                n,
                stream,
            )?;
        }

        // ── 10. O projection GEMM ── (extracted to paged_oproj.rs)
        let o_out = self.prefill_attention_paged_oproj(attn_out, n, h, nq, hd, ctx, stream)?;

        Ok(o_out)
    }
}
