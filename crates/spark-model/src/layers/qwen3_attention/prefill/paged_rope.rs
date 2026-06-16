// SPDX-License-Identifier: AGPL-3.0-only

//! RoPE dispatch for the standard (non-MLA) `prefill_attention_paged` path.
//!
//! Extracted verbatim from `paged.rs` to keep that file under the 500-LoC
//! cap. The branch chain and kernel-launch order are unchanged.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Resolved RoPE inputs for one prefill chunk: contiguous Q/K buffers plus the
/// (possibly stacked) position pointers chosen by the caller for single-stream
/// vs batched mode.
pub(super) struct PagedRopeArgs {
    pub q_contiguous: DevicePtr,
    pub k_contiguous: DevicePtr,
    pub positions: DevicePtr,
    pub positions_h: DevicePtr,
    pub positions_w: DevicePtr,
    pub n: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// Apply RoPE (or MRoPE / proportional / YaRN variants) to the chunk's
    /// Q and K in place. MLA layers skip this — their RoPE is applied inside
    /// the MLA block to the rope portions only.
    pub(super) fn prefill_attention_paged_rope(
        &self,
        ctx: &ForwardContext,
        args: &PagedRopeArgs,
    ) -> Result<()> {
        let PagedRopeArgs {
            q_contiguous,
            k_contiguous,
            positions: bmeta_positions,
            positions_h: bmeta_positions_h,
            positions_w: bmeta_positions_w,
            n,
            nq,
            nkv,
            hd,
            stream,
        } = *args;

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
        Ok(())
    }
}
