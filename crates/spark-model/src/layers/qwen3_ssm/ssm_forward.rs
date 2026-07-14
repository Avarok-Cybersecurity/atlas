// SPDX-License-Identifier: AGPL-3.0-only

//! ssm_forward inherent helper.

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn ssm_forward(
        &self,
        normed: DevicePtr,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
        trace: bool,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk; // vheads_per_group = 2
        // Suppress debug synchronize+dump while capturing a CUDA graph:
        // cuStreamSynchronize during capture is illegal (CUDA 900) and
        // bricks the context (issue #106).
        let debug = tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture;
        let profile = ctx.profile;

        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    SSM {}: {:.0}μs", $label, t.elapsed().as_micros());
                    r
                } else {
                    $body
                }
            }};
        }

        // ── 1+2. QKVZ projection (+ deinterleave if needed) ──
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let qkvz_size = ctx.config.ssm_qkvz_size() as u32;
        prof!("qkvz", {
            if let Some(ref fp8) = self.qkvz_fp8w {
                // FP8 native: w8a16_gemv + deinterleave (no fused QKVZ variant yet).
                // `w8a16_gemv` consumes `[N/BS,K/BS] BF16` block scales
                // (see `kernels/gb10/common/w8a16_gemv.cu` line 5).
                // Asserting catches any silent leak of per-row F32 or
                // single-scale FP8 into this code path.
                fp8.scale_format.expect(
                    crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                    "ssm_forward::qkvz_fp8w → w8a16_gemv",
                );
                if self.sequential_qkvz {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed,
                        fp8.weight,
                        fp8.row_scale,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                } else {
                    let qkvz_out = ctx.buffers.ssm_qkvz();
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed,
                        fp8.weight,
                        fp8.row_scale,
                        qkvz_out,
                        qkvz_size,
                        h,
                        stream,
                    )?;
                    ops::deinterleave_qkvz(
                        ctx.gpu,
                        self.deinterleave_k,
                        qkvz_out,
                        deinterleaved,
                        1,
                        nk as u32,
                        kd as u32,
                        vpg as u32,
                        vd as u32,
                        stream,
                    )
                }
            } else if self.sequential_qkvz {
                // Qwen3.5: QKVZ weight is pre-concatenated [Q|K|V|Z] sequential.
                // Plain GEMV writes directly to deinterleaved buffer.
                if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed,
                        nvfp4,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &self.ssm.in_proj_qkvz,
                        deinterleaved,
                        qkvz_size,
                        h,
                        stream,
                    )
                }
            } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                // 80B: Fused QKVZ GEMV writes deinterleaved output directly
                ops::w4a16_gemv_qkvz(
                    ctx.gpu,
                    self.w4a16_gemv_qkvz_k,
                    normed,
                    nvfp4,
                    deinterleaved,
                    qkvz_size,
                    h,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )
            } else {
                // 80B fallback: interleaved GEMV + separate deinterleave
                let qkvz_out = ctx.buffers.ssm_qkvz();
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.ssm.in_proj_qkvz,
                    qkvz_out,
                    qkvz_size,
                    h,
                    stream,
                )?;
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    qkvz_out,
                    deinterleaved,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )
            }
        })?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at qkvz_proj");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "deinterleaved-Q", deinterleaved, 4);
        }

        // Sequential offsets into deinterleaved buffer: [Q_2048 | K_2048 | V_4096 | Z_4096]
        let key_dim = nk * kd; // 2048
        let value_dim = nv * vd; // 4096
        // Conv1d input starts at Q (first element of deinterleaved)
        let qkv_ptr = deinterleaved;
        // Z gate for gated norm (after Q+K+V)
        let z_ptr = deinterleaved.offset((key_dim * 2 + value_dim) * 2);

        // ── 3. Fused BA projection + GDN gates (single GEMV + inline transforms) ──
        let ba_size = ctx.config.ssm_ba_size() as u32;
        let gates = ctx.buffers.ssm_gates();
        let beta_fp32 = gates.offset(nv * 4); // FP32, after gate[nv]
        prof!("ba_gates", {
            ops::dense_gemv_ba_gates(
                ctx.gpu,
                self.ba_gates_k,
                normed,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                beta_fp32,
                ba_size,
                h,
                vpg as u32,
                stream,
            )
        })?;
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at ba_gates");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_f32(ctx.gpu, "gate", gates, 4);
            Self::debug_f32(ctx.gpu, "beta", beta_fp32, 4);
        }

        // ── 4+5b. Fused conv1d update + SiLU + L2 norm on Q/K ──
        // Combines causal_conv1d_update + l2_norm_bf16 into one kernel launch.
        // Q+K channels (0..4096) get L2-normalized per head after SiLU.
        // V channels (4096..8192) get SiLU only.
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qk_channels = (key_dim * 2) as u32;
        // Use FP32 conv output to prevent recurrent precision drift at 8k+ tokens.
        // The conv→GDN path is the SSM recurrent path; BF16 truncation at each token
        // compounds to ~noise after 8000 iterations.
        let (conv_out, use_f32_conv) = if self.conv1d_l2norm_f32_k.0 != 0 {
            (ctx.buffers.ssm_conv_out_f32(), true)
        } else {
            (ctx.buffers.ssm_qkvz(), false)
        };
        if use_f32_conv {
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_f32_k,
                state.conv_state,
                qkv_ptr,
                &self.ssm.conv1d,
                conv_out,
                conv_dim,
                d_conv,
                1,
                qk_channels,
                kd as u32,
                1e-6,
                stream,
            )?;
        } else {
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_k,
                state.conv_state,
                qkv_ptr,
                &self.ssm.conv1d,
                conv_out,
                conv_dim,
                d_conv,
                1,
                qk_channels,
                kd as u32,
                1e-6,
                stream,
            )?;
        }
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at conv1d_l2norm");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "conv1d-l2norm-out", conv_out, 4);
        }

        // ── 5. Split conv output → Q', K', V' ──
        // Bytes per element: 4 if FP32 conv output, 2 if BF16
        let elem_size = if use_f32_conv { 4 } else { 2 };
        let q_conv = conv_out;
        let k_conv = conv_out.offset(key_dim * elem_size);
        let v_conv = conv_out.offset(key_dim * 2 * elem_size);

        // ── 6. GDN decode ──
        // Kernel handles GQA: maps value_head → key_head via kh = vh / head_repeat
        // Q/K: [num_k_heads, k_dim], V: [num_v_heads, v_dim]
        // Use FP32 output variant when available to prevent BF16 truncation that
        // compounds to precision drift at 15K+ tokens in multi-turn conversations.
        let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
        let gdn_out = if use_f32_gdn {
            // Write FP32 GDN output to the unused tail of ssm_conv_out_f32.
            // Buffer layout: [Q|K|V](conv data) [Z-region](unused during decode).
            // Conv uses (2*key_dim + value_dim)*4 bytes; GDN output needs value_dim*4
            // bytes which fits in the remaining Z-region.
            ctx.buffers
                .ssm_conv_out_f32()
                .offset((key_dim * 2 + value_dim) * 4)
        } else {
            ctx.buffers.attn_output() // BF16 fallback
        };
        let normed_out = ctx.buffers.ssm_qkvz();
        let gdn_kernel = if use_f32_gdn {
            self.gdn_f32_k
        } else {
            self.gdn_k
        };
        let fused_gdn_norm = use_f32_gdn
            && self.gdn_f32_norm_k.0 != 0
            && std::env::var("ATLAS_GDN_FUSED_NORM").ok().as_deref() == Some("1");
        if fused_gdn_norm {
            ops::gdn_decode_f32_norm(
                ctx.gpu,
                self.gdn_f32_norm_k,
                state.h_state,
                q_conv,
                k_conv,
                v_conv,
                gates,
                beta_fp32,
                z_ptr,
                self.ssm.norm.weight,
                normed_out,
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
            if trace {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at gdn_decode_f32_norm");
                })?;
            }
        } else {
            ops::gdn_decode(
                ctx.gpu,
                gdn_kernel,
                state.h_state,
                q_conv,
                k_conv,
                v_conv,
                gates,
                beta_fp32,
                gdn_out,
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                stream,
            )?;
            if trace {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at gdn_decode");
                })?;
            }

            // ── 7. Gated RMS norm with Z gate — per-head normalization ──
            // GDN output is [nv * vd], norm weight is [vd], applied as [nv, vd]
            let norm_kernel = if use_f32_gdn {
                self.gated_rms_norm_f32_k
            } else {
                self.gated_rms_norm_k
            };
            ops::gated_rms_norm(
                ctx.gpu,
                norm_kernel,
                gdn_out,
                z_ptr,
                &self.ssm.norm,
                normed_out,
                nv as u32,
                vd as u32,
                vd as u32,
                ctx.config.rms_norm_eps as f32,
                vd as u32,
                stream,
            )?;
            if trace {
                ctx.gpu.synchronize(stream).inspect_err(|_e| {
                    tracing::error!("CRASH at gated_rms_norm");
                })?;
            }
        }
        if trace && fused_gdn_norm {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH after fused gdn norm");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "gated-norm-out", normed_out, 4);
        }

        // ── 8. Output projection: [value_dim → hidden_size] ──
        let out = ctx.buffers.moe_output();
        if let Some(ref fp8) = self.out_proj_fp8w {
            // `w8a16_gemv` consumes `[N/BS,K/BS] BF16` block scales —
            // the canonical Qwen FP8 release format.
            fp8.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "ssm_forward::out_proj_fp8w → w8a16_gemv",
            );
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed_out,
                fp8.weight,
                fp8.row_scale,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed_out,
                dense_out,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                normed_out,
                &self.ssm.out_proj,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        }
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at out_proj");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "out-proj", out, 4);
        }

        // GDN HeadParallel: `out` is this rank's PARTIAL row-parallel out_proj
        // over its local value heads. Reduce across TP ranks to the complete
        // SSM output before the caller's residual add. Single-token path
        // (dense_gemv / w8a16_gemv / w4a16_gemv above → one position), so
        // num_tokens = 1. No-op at tp=1. Covers single-token decode
        // (trait_decode) and per-sequence multi-seq decode (trait_decode_multi_seq).
        self.ssm_tp_all_reduce(out, 1, ctx, stream)?;

        Ok(out)
    }
}
