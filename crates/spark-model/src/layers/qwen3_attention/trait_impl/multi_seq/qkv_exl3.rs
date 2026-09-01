// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 arm of the multi-seq decode Q/K/V phase
//! (`ATLAS_EXL3_NATIVE_DENSE=1`). Split from `qkv.rs` (500-LoC cap).
//!
//! `normed` is contiguous `[n, h]`; `qkv_buf` is `[n, Q|K|V]` with rows
//! `per_seq_qkv` bytes apart, so the three projections write pitched
//! destinations (row stride `per_seq_qkv / 2` elements, each offset to its
//! column block) in ONE launch section: one f16 ingress of `normed`, three
//! matmuls (GEMV tier for n <= 8, row-batched GEMM above), three strided
//! egresses. Bit-for-bit the same layout the NVFP4 batch2/batch3/batchn and
//! per-seq arms produce, so `ms_qkv_apply_lora` / `ms_qkv_norms` are
//! unchanged.
//!
//! Gated Q lands RAW-interleaved `[Q|gate]` (checkpoint column order); the
//! deinterleave runs here for all `n` rows in one launch — unless a q
//! adapter is resident, in which case it is deferred past the q LoRA fold to
//! `ms_qkv_deinterleave_q`, exactly like the other gated arms.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::ops::Exl3DenseOut;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// `Ok(true)` when this layer serves q/k/v from packed EXL3 trellis and
    /// the projections were written; `Ok(false)` for the materialized arms.
    pub(super) fn ms_qkv_exl3(&self, c: &MultiSeqCtx<'_>) -> Result<bool> {
        let Some(x) = self.exl3_attn_arm(c.fwd, "decode_multi_seq q/k/v_proj")? else {
            return Ok(false);
        };
        debug_assert_eq!(c.per_seq_qkv % c.bf16, 0);
        let ld = c.per_seq_qkv / c.bf16;
        let kv_bytes = (c.nkv * c.hd) as usize * c.bf16;
        x.qkv_linear(
            c.fwd.gpu,
            c.normed,
            Exl3DenseOut::strided(c.qkv_buf, ld),
            Exl3DenseOut::strided(c.qkv_buf.offset(c.q_proj_bytes), ld),
            Exl3DenseOut::strided(c.qkv_buf.offset(c.q_proj_bytes + kv_bytes), ld),
            c.n,
            c.stream,
        )?;
        if self.gated && !self.q_lora_active() {
            // One launch over all n rows: block t deinterleaves the Q segment
            // at qkv_buf + t * stride in place.
            ops::deinterleave_qg(
                c.fwd.gpu,
                self.deinterleave_qg_k,
                c.qkv_buf,
                c.n as u32,
                c.nq,
                c.hd,
                ld as u32,
                c.stream,
            )?;
        }
        Ok(true)
    }
}
