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

/// Diagnostic kill-switch: `ATLAS_GDN_WY17=0` forces the K=17 verify off the
/// fused wy17 arm (BF16 conv + WY-chunkwise GDN) onto the sequential
/// per-token fallback (FP32 conv + gdn_decode — the numerics closest to
/// single-token decode). MUCH slower; for greedy-losslessness bisection only.
fn wy17_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GDN_WY17").ok().as_deref() != Some("0"))
}

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
    /// STAGE 1: whether the fused K=2 MTP-verify epilogue (single-launch
    /// conv1d+L2norm and gated-RMS-norm for both draft positions) should run.
    ///
    /// Opt-in via `ATLAS_GDN_FUSED_VERIFY=1` (default OFF — the per-token path
    /// runs unchanged) AND only when the fused kernels are present in this
    /// target's PTX module set (NULL handle on non-gb10 targets). Bit-identical
    /// to the per-token path (gdn_verify_fused_microtest, cos == 1.0).
    pub(super) fn fused_verify_k2_enabled(&self) -> bool {
        self.gdn_verify_fused_conv_k2_k.0 != 0
            && self.gdn_verify_fused_norm_k2_k.0 != 0
            && matches!(
                std::env::var("ATLAS_GDN_FUSED_VERIFY").ok().as_deref(),
                Some("1")
            )
    }

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
            if self.fused_verify_k2_enabled() {
                // STAGE 1: single-launch conv1d+L2norm for BOTH positions.
                // Writes conv_out[0..1] and the position-0 rollback snapshot
                // (intermediates[0]) inline — saving one conv launch + one
                // copy_d2d vs the per-token path. The committed (post-t1)
                // window is left in conv_state; copy it to intermediates[1]
                // for the full-accept rollback restore.
                ops::gdn_verify_fused_conv_k2(
                    ctx.gpu,
                    self.gdn_verify_fused_conv_k2_k,
                    ssm_state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_out_buf,
                    ssm_state.conv_state_intermediates[0],
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    qkvz_size as u32, // input stride (BF16 elems between positions)
                    conv_dim as u32,  // output stride (BF16 elems between positions)
                    1e-6,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[1],
                    conv_bytes,
                    stream,
                )?;
            } else {
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
            }

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
        } else if num_tokens == 17 && self.gdn_wy17_k.0 != 0 && wy17_enabled() {
            // ── K=17 (DFlash γ+1): fused WY-Chunkwise path ──
            //
            // Conv epilogue: single fused launch over all 17 positions when
            // the kernel is present (writes every rollback snapshot inline —
            // replaces 17 conv launches + 17 copy_d2d, each serialized on the
            // in-place conv_state update at 1-token occupancy). Requires the
            // pool-contiguous intermediates layout (slot-major, stride
            // conv_bytes — same contiguity the WY17 GDN call below already
            // assumes for h_state_intermediates). Kill-switch:
            // ATLAS_GDN_FUSED_CONV17=0 restores the per-token loop for A/B.
            let conv_inter_base = ssm_state.conv_state_intermediates[0];
            let inter_contiguous = ssm_state
                .conv_state_intermediates
                .iter()
                .take(num_tokens)
                .enumerate()
                .all(|(t, p)| p.0 == conv_inter_base.0 + (t * conv_bytes) as u64);
            let fused_conv = self.gdn_verify_fused_conv_kn_k.0 != 0
                && inter_contiguous
                && !matches!(
                    std::env::var("ATLAS_GDN_FUSED_CONV17").ok().as_deref(),
                    Some("0")
                );
            if fused_conv {
                ops::gdn_verify_fused_conv_kn(
                    ctx.gpu,
                    self.gdn_verify_fused_conv_kn_k,
                    ssm_state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_out_buf,
                    conv_inter_base,
                    num_tokens as u32,
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    qkvz_size as u32, // input stride (BF16 elems between positions)
                    conv_dim as u32,  // output stride (BF16 elems between positions)
                    (conv_bytes / 4) as u32, // snapshot stride (FP32 elems)
                    1e-6,
                    stream,
                )?;
            } else {
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
            //
            // gated_delta_rule_decode expects FP32 Q/K/V (see kernel signature),
            // but causal_conv1d_update_l2norm outputs BF16 by default. Reading
            // BF16 conv output as FP32 produces garbage → every argmax disagrees
            // with the draft → 0% accept on wide-γ DFlash verify.
            //
            // Fix: use the FP32 conv variant (conv1d_l2norm_f32_k) into
            // ssm_conv_out_f32, then stride Q/K/V with FP32 element size.
            // Mirrors ssm_forward.rs single-token decode. When the FP32 kernel
            // is absent (non-GB10 backends) fall through to BF16 conv as before.
            let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
            let conv_elem = if use_f32_conv { fp32 } else { bf16 };
            let conv_kernel = if use_f32_conv {
                self.conv1d_l2norm_f32_k
            } else {
                self.conv1d_l2norm_k
            };
            // ssm_conv_out_f32 is sized m * qkvz_size * 4 bytes (m ≥ 32 for
            // DFlash), so K ≤ 32 tokens always fit without aliasing ssm_qkvz.
            let f32_conv_base = ctx.buffers.ssm_conv_out_f32();

            for t in 0..(num_tokens as u32) {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                // Write conv output for token t to the correct typed buffer.
                let conv_out_t = if use_f32_conv {
                    f32_conv_base.offset(t as usize * conv_dim * fp32)
                } else {
                    conv_out_buf.offset(t as usize * conv_dim * bf16)
                };
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    conv_kernel,
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

                // Q/K/V pointers into conv output; element size matches the
                // kernel's type expectation (FP32 for gated_delta_rule_decode).
                let q_t = conv_out_t;
                let k_t = conv_out_t.offset(key_dim * conv_elem);
                let v_t = conv_out_t.offset(key_dim * 2 * conv_elem);
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
