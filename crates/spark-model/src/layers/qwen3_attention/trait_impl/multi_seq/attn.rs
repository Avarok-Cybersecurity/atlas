// SPDX-License-Identifier: AGPL-3.0-only

//! Phases 3-6: per-sequence RoPE, KV-cache write, batched paged
//! attention, gate multiply + O projection.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// Phase 3: per-token RoPE (each sequence has its own position).
    pub(super) fn ms_phase_rope(&self, c: &MultiSeqCtx<'_>, meta: AttnMetadataDev) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let pos_i = meta.positions.offset(i * 4); // u32 per position
            ops::rope(
                fwd.gpu,
                self.rope_k,
                q_out_i,
                k_out_i,
                pos_i,
                1,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(fwd.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(fwd.config.rope_theta as f32),
                stream,
            )?;
        }
        Ok(())
    }

    /// Phase 4: per-token KV cache write.
    pub(super) fn ms_phase_cache_write(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nkv,
            hd,
            bs,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let kv_stride = nkv * hd;
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);
            let slot_i = meta.slot.offset(i * 8); // i64 per slot
            self.write_kv_cache(
                fwd.gpu,
                k_out_i,
                v_out_i,
                kv_cache,
                slot_i,
                1,
                nkv,
                hd,
                bs,
                kv_stride,
                kv_stride,
                stream,
                fwd.graph_capture,
            )?;
        }
        Ok(())
    }

    /// Phase 5: build contiguous Q buffer + run BATCHED paged decode.
    /// Returns the attn_out buffer pointer for downstream phases.
    pub(super) fn ms_phase_paged_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // Build contiguous Q buffer [N, nq*hd] for batched attention.
        let q_contiguous = fwd.buffers.ssm_qkvz();
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            fwd.gpu.copy_d2d_async(
                q_out_i,
                q_contiguous.offset(i * q_dim as usize * bf16),
                q_dim as usize * bf16,
                stream,
            )?;
        }
        let attn_out = fwd.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        self.run_paged_decode(
            fwd.gpu,
            q_contiguous,
            kv_cache,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n as u32,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            nq * hd,
            fwd.buffers.splitk_workspace(),
            stream,
        )?;
        Ok(attn_out)
    }

    /// Phase 6: gate multiply (when gated) + O projection. Writes to
    /// `o_out`. Returns the o_out buffer pointer.
    pub(super) fn ms_phase_o_proj(
        &self,
        c: &MultiSeqCtx<'_>,
        attn_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            hd,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        if self.gated {
            for i in 0..n {
                let gate_i = qkv_buf.offset(i * per_seq_qkv + q_dim as usize * bf16);
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                ops::sigmoid_gate_mul(
                    fwd.gpu,
                    self.sigmoid_gate_mul_k,
                    attn_out_i,
                    gate_i,
                    attn_out_i,
                    q_dim,
                    stream,
                )?;
            }
        }

        let o_out = fwd.buffers.moe_output();
        if let Some(o_fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            // FP8 native: per-token w8a16_gemv for O projection.
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                ops::w8a16_gemv(
                    fwd.gpu,
                    self.w8a16_gemv_k,
                    attn_out_i,
                    o_fp8.weight,
                    o_fp8.row_scale,
                    o_out_i,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        } else if n == 3 && !self.attn.o_proj.is_null() {
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if n == 2 && !self.attn.o_proj.is_null() {
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else {
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    attn_out_i,
                    &self.attn.o_proj,
                    o_out_i,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        }
        Ok(o_out)
    }
}
