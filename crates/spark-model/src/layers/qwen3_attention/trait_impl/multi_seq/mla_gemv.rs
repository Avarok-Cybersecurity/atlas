// SPDX-License-Identifier: AGPL-3.0-only

//! Batched-GEMV helpers for the multi-sequence MLA decode path
//! (`super::mla`). Split out to keep `mla.rs` under the 500-LoC cap.
//!
//! Both helpers wrap `mla_batched_gemv` (one launch, batched over the
//! `nq` heads) with a per-head `dense_gemv` fallback for the case where
//! the batched kernel is unavailable. They mirror the absorbed-Q and
//! V-extraction steps of `decode::attention_forward_mla` 1:1.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// Per-call MLA scalar dimensions, bundled to keep the per-sequence
/// helper signature under the argument-count lint.
#[derive(Clone, Copy)]
pub(super) struct MlaDims {
    pub h: u32,
    pub nq: u32,
    pub hd: u32,
    pub q_dim: u32,
    pub q_lora: u32,
    pub kv_lora: u32,
    pub mla_nope: u32,
    pub mla_v_dim: u32,
    pub mla_rope: u32,
    pub mla_cache_dim: u32,
    pub eps: f32,
    pub bs: usize,
    pub inv_sqrt_d: f32,
}

impl Qwen3AttentionLayer {
    /// Q absorption: `Q_absorbed[head] = Q_nope[head] @ W_UK_T[head]`.
    /// Batched-GEMV kernel when available, else per-head dense GEMV.
    pub(super) fn ms_mla_q_absorb(
        &self,
        c: &MultiSeqCtx<'_>,
        mla: &crate::layers::qwen3_attention::types::MlaWeights,
        d: &MlaDims,
        q_full: DevicePtr,
        q_absorbed_buf: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        if self.mla_batched_gemv_k.0 != 0 {
            ops::mla_batched_gemv(
                gpu,
                self.mla_batched_gemv_k,
                q_full,
                mla.w_uk_t.weight,
                q_absorbed_buf,
                d.kv_lora,
                d.mla_nope,
                d.nq,
                d.hd,
                d.mla_cache_dim,
                stream,
            )
        } else {
            for head_idx in 0..d.nq as usize {
                let q_nope_ptr = q_full.offset(head_idx * d.hd as usize * 2);
                let q_abs_dst = q_absorbed_buf.offset(head_idx * d.mla_cache_dim as usize * 2);
                let w_uk_head = mla
                    .w_uk_t
                    .weight
                    .offset(head_idx * mla.nope * mla.kv_lora_rank * 2);
                let w_uk_dense = crate::weight_map::DenseWeight { weight: w_uk_head };
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    q_nope_ptr,
                    &w_uk_dense,
                    q_abs_dst,
                    d.kv_lora,
                    d.mla_nope,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    /// V extraction: `v_out[head] = attn_latent[head] @ W_UV[head]`.
    pub(super) fn ms_mla_v_extract(
        &self,
        c: &MultiSeqCtx<'_>,
        mla: &crate::layers::qwen3_attention::types::MlaWeights,
        d: &MlaDims,
        attn_out: DevicePtr,
        v_extracted: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        if self.mla_batched_gemv_k.0 != 0 {
            ops::mla_batched_gemv(
                gpu,
                self.mla_batched_gemv_k,
                attn_out,
                mla.w_uv.weight,
                v_extracted,
                d.mla_v_dim,
                d.kv_lora,
                d.nq,
                d.mla_cache_dim,
                d.mla_v_dim,
                stream,
            )
        } else {
            for head_idx in 0..d.nq as usize {
                let attn_head = attn_out.offset(head_idx * d.mla_cache_dim as usize * 2);
                let w_uv_head = mla
                    .w_uv
                    .weight
                    .offset(head_idx * mla.v_dim * mla.kv_lora_rank * 2);
                let v_dst = v_extracted.offset(head_idx * mla.v_dim * 2);
                let w_uv_dense = crate::weight_map::DenseWeight { weight: w_uv_head };
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    attn_head,
                    &w_uv_dense,
                    v_dst,
                    d.mla_v_dim,
                    d.kv_lora,
                    stream,
                )?;
            }
            Ok(())
        }
    }
}
