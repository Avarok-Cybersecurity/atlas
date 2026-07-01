// SPDX-License-Identifier: AGPL-3.0-only

//! Sequential per-token fallback (K != 2,3,4,16,17) of
//! `Qwen3SsmLayer::decode_batched_conv_gdn`. Extracted from
//! `trait_decode_batched_conv_gdn.rs` to keep the parent file under 500 LoC.

use anyhow::Result;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3SsmLayer {
    /// Models that ship a FP32 conv kernel (e.g. qwen3.6-27b/nvfp4) also ship
    /// a gdn_decode kernel that takes const float* query/key/value — passing
    /// BF16 data to it misinterprets every two BF16 elements as one FP32,
    /// causing h_state corruption and NaN after ~7 recurrent steps.
    /// Use the FP32 conv kernel and ssm_conv_out_f32 buffer when available.
    pub(super) fn decode_batched_conv_gdn_sequential(
        &self,
        ssm_state: &mut SsmLayerState,
        ctx: &ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<()> {
        let ConvGdnArgs {
            num_tokens,
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            h_bytes,
            conv_bytes,
            qkvz_size,
            conv_dim,
            key_dim,
            value_dim,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
        } = *args;

        let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
        if num_tokens > 1 {
            tracing::warn!("sequential K={} use_f32_conv={}", num_tokens, use_f32_conv);
        }
        let seq_conv_buf = if use_f32_conv {
            ctx.buffers.ssm_conv_out_f32()
        } else {
            conv_out_buf
        };
        let conv1d_k = if use_f32_conv {
            self.conv1d_l2norm_f32_k
        } else {
            self.conv1d_l2norm_k
        };
        // Element size in bytes for conv output (FP32=4, BF16=2).
        let coes = if use_f32_conv { fp32 } else { bf16 };

        for t in 0..(num_tokens as u32) {
            let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
            let conv_out_t = seq_conv_buf.offset(t as usize * conv_dim * coes);
            ops::conv1d_update_l2norm(
                ctx.gpu,
                conv1d_k,
                ssm_state.conv_state,
                qkv_t,
                &self.ssm.conv1d,
                conv_out_t,
                conv_dim as u32,
                d_conv as u32,
                1,
                qk_ch,
                kd as u32,
                1e-6,
                stream,
            )?;

            let q_t = conv_out_t;
            let k_t = seq_conv_buf.offset(t as usize * conv_dim * coes + key_dim * coes);
            let v_t = seq_conv_buf.offset(t as usize * conv_dim * coes + key_dim * 2 * coes);
            let gate_beta_stride = nv * 2 * fp32;
            let gate_t = gates_buf.offset(t as usize * gate_beta_stride);
            let beta_t = gates_buf.offset(t as usize * gate_beta_stride + nv * fp32);
            let gdn_out_t = gdn_out_buf.offset(t as usize * value_dim * bf16);
            let do_norm_t = ssm_state.norm_token_count.is_multiple_of(16) as u32;
            ssm_state.norm_token_count = ssm_state.norm_token_count.wrapping_add(1);

            ops::gdn_decode(
                ctx.gpu,
                self.gdn_k,
                ssm_state.h_state,
                q_t,
                k_t,
                v_t,
                gate_t,
                beta_t,
                gdn_out_t,
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                do_norm_t,
                stream,
            )?;

            ctx.gpu.copy_d2d_async(
                ssm_state.h_state,
                ssm_state.h_state_intermediates[t as usize],
                h_bytes,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                ssm_state.conv_state,
                ssm_state.conv_state_intermediates[t as usize],
                conv_bytes,
                stream,
            )?;
        }

        Ok(())
    }
}
