// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 5-7 of `Qwen3SsmLayer::decode_batched_inner`: Conv1d + L2 norm +
//! GDN per-token (with intermediate checkpoints). Extracted from
//! `trait_decode_batched.rs` to keep the parent file under 500 LoC.
//! Dispatches one of the fused K=2/3/4/17 paths or the sequential
//! per-token fallback. All buffers + state are owned by the caller; this
//! function only mutates `ssm_state.h_state`, `ssm_state.conv_state`,
//! their intermediates, `conv_out_buf`, and `gdn_out_buf`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
pub(super) struct ConvGdnArgs {
    pub num_tokens: usize,
    pub deinterleaved: DevicePtr,
    pub gates_buf: DevicePtr,
    pub conv_out_buf: DevicePtr,
    pub gdn_out_buf: DevicePtr,
    pub h_bytes: usize,
    pub conv_bytes: usize,
    pub qkvz_size: usize,
    pub conv_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub d_conv: usize,
    pub qk_ch: u32,
    pub nk: usize,
    pub nv: usize,
    pub kd: usize,
    pub vd: usize,
    pub bf16: usize,
    pub fp32: usize,
    pub stream: u64,
}

impl Qwen3SsmLayer {
    /// Run conv1d_update_l2norm + GDN over `num_tokens` (multi-token decode
    /// / MTP verify). Picks the K=2/3/4/17 fused WY path if available,
    /// otherwise falls back to the sequential per-token gdn_decode loop.
    pub(super) fn decode_batched_conv_gdn(
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
            value_dim: _,
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

        if num_tokens == 4 {
            // ── K=4 fused path: conv1d+L2norm sequential, GDN WY4 ──
            for t in 0..4u32 {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
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
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t as usize],
                    conv_bytes,
                    stream,
                )?;
            }

            // WY-chunkwise GDN: 2-pass algorithm for 4-token verification.
            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy4(
                ctx.gpu,
                self.gdn_wy4_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                ssm_state.h_state_intermediates[1],
                ssm_state.h_state_intermediates[2],
                1, // batch_size
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32, // qk_stride
                conv_dim as u32, // v_stride
                (nv * 2) as u32, // gb_stride
                stream,
            )?;
        } else if num_tokens == 3 {
            // ── K=3 fused path: conv1d+L2norm per token, GDN WY3 ──
            for t in 0..3u32 {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
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
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t as usize],
                    conv_bytes,
                    stream,
                )?;
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy3(
                ctx.gpu,
                self.gdn_wy3_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                ssm_state.h_state_intermediates[1],
                1, // batch_size
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32, // qk_stride
                conv_dim as u32, // v_stride
                (nv * 2) as u32, // gb_stride
                stream,
            )?;
        } else if num_tokens == 2 {
            // ── K=2 fused path: conv1d sequential, L2 norm sequential, GDN chunk2 ──
            let qkv_0 = deinterleaved;
            let conv_out_0 = conv_out_buf;
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_k,
                ssm_state.conv_state,
                qkv_0,
                &self.ssm.conv1d,
                conv_out_0,
                conv_dim as u32,
                d_conv as u32,
                1,
                qk_ch,
                kd as u32,
                1e-6,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                ssm_state.conv_state,
                ssm_state.conv_state_intermediates[0],
                conv_bytes,
                stream,
            )?;

            let qkv_1 = deinterleaved.offset(qkvz_size * bf16);
            let conv_out_1 = conv_out_buf.offset(conv_dim * bf16);
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_k,
                ssm_state.conv_state,
                qkv_1,
                &self.ssm.conv1d,
                conv_out_1,
                conv_dim as u32,
                d_conv as u32,
                1,
                qk_ch,
                kd as u32,
                1e-6,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                ssm_state.conv_state,
                ssm_state.conv_state_intermediates[1],
                conv_bytes,
                stream,
            )?;

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy2(
                ctx.gpu,
                self.gdn_wy2_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                1, // batch_size
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32, // qk_stride
                conv_dim as u32, // v_stride
                (nv * 2) as u32, // gb_stride
                stream,
            )?;
        } else if num_tokens == 17 && self.gdn_wy17_k.0 != 0 {
            // ── K=17 (DFlash γ+1): fused WY-Chunkwise path ──
            for t in 0..(num_tokens as u32) {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
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
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t as usize],
                    conv_bytes,
                    stream,
                )?;
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            let inter_stride_floats = (h_bytes / 4) as u32;
            ops::gdn_decode_wy17(
                ctx.gpu,
                self.gdn_wy17_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                inter_stride_floats,
                1, // batch_size
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32, // qk_stride
                conv_dim as u32, // v_stride
                (nv * 2) as u32, // gb_stride
                stream,
            )?;
        } else {
            // ── K!=2,17: sequential per-token path ──
            for t in 0..(num_tokens as u32) {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
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
                let k_t = conv_out_buf.offset(t as usize * conv_dim * bf16 + key_dim * bf16);
                let v_t = conv_out_buf.offset(t as usize * conv_dim * bf16 + key_dim * 2 * bf16);
                let gate_beta_stride = nv * 2 * fp32;
                let gate_t = gates_buf.offset(t as usize * gate_beta_stride);
                let beta_t = gates_buf.offset(t as usize * gate_beta_stride + nv * fp32);
                let gdn_out_t = gdn_out_buf.offset(t as usize * args.value_dim * bf16);
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
        }

        Ok(())
    }
}
