// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq — recurrent inner (BA/gates → conv1d →
//! GDN → gated-norm) for the batched-projection SSM mixer.

use super::super::*;

/// How often the batched-recurrent fast path engaged vs fell back to the per-seq
/// loop. The precondition (pool slots contiguous AND in slice order) fails as
/// slots fragment, and it used to fail silently.
static BATCHED_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK_ONCE: std::sync::Once = std::sync::Once::new();

impl Qwen3SsmLayer {
    /// Recurrent inner of the batched-projection SSM mixer.
    ///
    /// Default: per-seq, byte-identical to `ssm_forward`. Experimental path:
    /// use existing batch dimensions for BA/gates, conv, GDN, and gated norm
    /// when the SSM pool states are contiguous slots `[0..n)`.
    ///
    /// `detail_t0` / `detail_parts` thread the caller's profiling state through
    /// so the `ATLAS_SSM_DETAIL` summary spans the whole mixer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_ms_ssm_recurrent<'a, 'b: 'a>(
        &self,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        n: usize,
        normed_base: DevicePtr,
        deinterleaved: DevicePtr,
        normed_out_base: DevicePtr,
        qkvz_size: usize,
        key_dim: usize,
        value_dim: usize,
        conv_dim: u32,
        qk_channels: u32,
        d_conv: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        vpg: usize,
        ba_size: u32,
        h: usize,
        bf16: usize,
        eps: f32,
        detail_profile: bool,
        detail_parts: &mut Vec<(&'static str, u128)>,
        detail_t0: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut rec_ba_us = 0u128;
        let mut rec_conv_us = 0u128;
        let mut rec_gdn_us = 0u128;
        let mut rec_norm_us = 0u128;
        macro_rules! detail_step {
            ($label:expr) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                    *detail_t0 = Some(std::time::Instant::now());
                }
            };
        }

        let batched_recurrent = if std::env::var("ATLAS_SSM_BATCHED_RECURRENT").ok().as_deref()
            == Some("1")
            && self.gdn_f32_strided_k.0 != 0
            && n > 1
        {
            let mut h_base = DevicePtr::NULL;
            let mut conv_base = DevicePtr::NULL;
            let mut contiguous = true;
            for i in 0..n {
                let ssm_state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                if i == 0 {
                    h_base = ssm_state.h_state;
                    conv_base = ssm_state.conv_state;
                } else {
                    contiguous &= ssm_state.h_state.0 == h_base.0 + (i * self.h_state_bytes) as u64;
                    contiguous &=
                        ssm_state.conv_state.0 == conv_base.0 + (i * self.conv_state_bytes) as u64;
                }
            }
            if contiguous {
                BATCHED_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some((h_base, conv_base))
            } else {
                // Falling back to the per-seq loop costs ~28% on this block
                // (measured n=16: batched 618 us/layer vs per-seq 853). It used
                // to happen SILENTLY, so a profile could show both paths in one
                // run with no way to tell how often or why. Report the first
                // one with the offending slot, and keep a count.
                let n_fb = FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                FALLBACK_ONCE.call_once(|| {
                    let mut broke_at = usize::MAX;
                    let mut delta = 0i64;
                    for i in 1..n {
                        if let Some(st) = states[i].as_any_mut().downcast_mut::<SsmLayerState>() {
                            let want = h_base.0 + (i * self.h_state_bytes) as u64;
                            if st.h_state.0 != want {
                                broke_at = i;
                                delta = st.h_state.0 as i64 - want as i64;
                                break;
                            }
                        }
                    }
                    tracing::info!(
                        "SSM batched recurrent DECLINED (n={n}): pool slots are not contiguous in \
                         slice order — seq {broke_at} sits {} slot(s) from where the batch axis \
                         expects it. The per-seq loop costs ~28% more on this block. Slots \
                         fragment as sequences finish, so this is expected to recur; count is \
                         logged at debug on every occurrence.",
                        delta / self.h_state_bytes.max(1) as i64
                    );
                });
                tracing::debug!("SSM batched recurrent fallback #{n_fb} (n={n})");
                None
            }
        } else {
            None
        };

        if let Some((h_state_base, conv_state_base)) = batched_recurrent {
            let gates = ctx.buffers.ssm_gates();
            let beta_fp32 = gates.offset(nv * 4);
            let gate_stride = (nv * 2) as u32;
            ops::dense_gemm_ba_gates_prefill(
                ctx.gpu,
                self.ba_gates_prefill_k,
                normed_base,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                n as u32,
                ba_size,
                h as u32,
                h as u32,
                gate_stride,
                nv as u32,
                vpg as u32,
                stream,
            )?;
            detail_step!("recurrent_batched_ba");

            let conv_out = ctx.buffers.ssm_conv_out_f32();
            // CONV INPUT-STRIDE: the conv reads `deinterleaved` (the QKVZ
            // projection output), whose rows are `qkvz_size` apart, and writes
            // a `conv_dim`-strided FP32 row. The plain kernel hardcodes BOTH
            // strides as `dim` (= conv_dim), so a batch=n launch would read
            // seq b>=1 from `b*conv_dim` instead of `b*qkvz_size` — landing in
            // the previous seq's Z-gate region and feeding garbage into the
            // GDN scan (correct at n=1, silently corrupt at n>=2).
            //
            // The strided kernel takes both strides explicitly, so the whole
            // batch goes in ONE launch. Kernel sets that predate it report
            // handle 0; there we keep the per-seq loop with pre-offset
            // pointers, which honours the same two strides one row at a time.
            if self.conv1d_l2norm_f32_strided_k.0 != 0 {
                ops::conv1d_update_l2norm_strided(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_strided_k,
                    conv_state_base,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_out,
                    conv_dim,
                    d_conv,
                    n as u32,
                    qk_channels,
                    kd as u32,
                    1e-6,
                    qkvz_size as u32, // input row stride (BF16 elements)
                    conv_dim,         // output row stride (FP32 elements)
                    stream,
                )?;
            } else {
                for i in 0..n {
                    ops::conv1d_update_l2norm(
                        ctx.gpu,
                        self.conv1d_l2norm_f32_k,
                        conv_state_base.offset(i * self.conv_state_bytes),
                        deinterleaved.offset(i * qkvz_size * bf16),
                        &self.ssm.conv1d,
                        conv_out.offset(i * conv_dim as usize * 4), // FP32, conv_dim-strided
                        conv_dim,
                        d_conv,
                        1,
                        qk_channels,
                        kd as u32,
                        1e-6,
                        stream,
                    )?;
                }
            }
            detail_step!("recurrent_batched_conv");

            if self.gdn_f32_strided_norm_k.0 != 0
                && std::env::var("ATLAS_GDN_FUSED_NORM").ok().as_deref() == Some("1")
            {
                let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
                fn gdn_half_reg_enabled() -> bool {
                    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    *ON.get_or_init(|| {
                        std::env::var("ATLAS_NO_GDN_HALF_REG").ok().as_deref() != Some("1")
                    })
                }
                let gdn_norm_k = if kd == 128
                    && vd == 128
                    && self.gdn_f32_strided_norm_half_k.0 != 0
                    && gdn_half_reg_enabled()
                {
                    self.gdn_f32_strided_norm_half_k
                } else {
                    self.gdn_f32_strided_norm_k
                };
                ops::gdn_decode_f32_strided_norm(
                    ctx.gpu,
                    gdn_norm_k,
                    h_state_base,
                    conv_out,
                    conv_out.offset(key_dim * 4),
                    conv_out.offset(key_dim * 2 * 4),
                    gates,
                    beta_fp32,
                    z_base,
                    self.ssm.norm.weight,
                    normed_out_base,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim,
                    conv_dim,
                    gate_stride,
                    qkvz_size as u32,
                    value_dim as u32,
                    eps,
                    stream,
                )?;
                detail_step!("recurrent_batched_gdn_norm");
            } else {
                let gdn_out = conv_out.offset(n * conv_dim as usize * 4);
                ops::gdn_decode_f32_strided(
                    ctx.gpu,
                    self.gdn_f32_strided_k,
                    h_state_base,
                    conv_out,
                    conv_out.offset(key_dim * 4),
                    conv_out.offset(key_dim * 2 * 4),
                    gates,
                    beta_fp32,
                    gdn_out,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim,
                    conv_dim,
                    gate_stride,
                    value_dim as u32,
                    stream,
                )?;
                detail_step!("recurrent_batched_gdn");

                for i in 0..n {
                    let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
                    let z_i = deint_i.offset((key_dim * 2 + value_dim) * bf16);
                    let gdn_out_i = gdn_out.offset(i * value_dim * 4);
                    let normed_out_i = normed_out_base.offset(i * value_dim * bf16);
                    ops::gated_rms_norm(
                        ctx.gpu,
                        self.gated_rms_norm_f32_k,
                        gdn_out_i,
                        z_i,
                        &self.ssm.norm,
                        normed_out_i,
                        nv as u32,
                        vd as u32,
                        vd as u32,
                        eps,
                        vd as u32,
                        stream,
                    )?;
                }
                detail_step!("recurrent_batched_norm");
            }
        } else {
            for i in 0..n {
                let normed_i = normed_base.offset(i * h * bf16);
                let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
                let z_i = deint_i.offset((key_dim * 2 + value_dim) * bf16);
                let normed_out_i = normed_out_base.offset(i * value_dim * bf16);

                let ssm_state = states[i]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

                let gates = ctx.buffers.ssm_gates();
                let beta_fp32 = gates.offset(nv * 4);
                let sub_t0 = if detail_profile {
                    ctx.gpu.synchronize(stream).ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                ops::dense_gemv_ba_gates(
                    ctx.gpu,
                    self.ba_gates_k,
                    normed_i,
                    &self.ssm.in_proj_ba,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates,
                    beta_fp32,
                    ba_size,
                    h as u32,
                    vpg as u32,
                    stream,
                )?;
                if let Some(t0) = sub_t0 {
                    ctx.gpu.synchronize(stream).ok();
                    rec_ba_us += t0.elapsed().as_micros();
                }

                let conv_out = ctx.buffers.ssm_conv_out_f32();
                // Fused conv+gdn+norm decode kernel (one launch instead of
                // conv1d_l2norm -> gdn -> norm). Race-free only for head_repeat=2
                // (Holo: 16 k / 32 v) + 128-dim heads. Skips the standalone conv
                // when active (the fused kernel does the conv internally).
                let use_fused_conv = self.gdn_f32_conv_norm_k.0 != 0
                    && nv == nk * 2
                    && kd == 128
                    && vd == 128
                    && std::env::var("ATLAS_GDN_FUSED_CONV").ok().as_deref() == Some("1");
                let sub_t0 = if detail_profile {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if !use_fused_conv {
                    ops::conv1d_update_l2norm(
                        ctx.gpu,
                        self.conv1d_l2norm_f32_k,
                        ssm_state.conv_state,
                        deint_i,
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
                if let Some(t0) = sub_t0 {
                    ctx.gpu.synchronize(stream).ok();
                    rec_conv_us += t0.elapsed().as_micros();
                }

                let gdn_out = conv_out.offset((key_dim * 2 + value_dim) * 4);
                let q_conv = conv_out;
                let k_conv = conv_out.offset(key_dim * 4);
                let v_conv = conv_out.offset(key_dim * 2 * 4);
                let sub_t0 = if detail_profile {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if use_fused_conv {
                    ops::gdn_decode_f32_conv_norm(
                        ctx.gpu,
                        self.gdn_f32_conv_norm_k,
                        ssm_state.h_state,
                        ssm_state.conv_state,
                        deint_i,
                        self.ssm.conv1d.weight,
                        gates,
                        beta_fp32,
                        z_i,
                        self.ssm.norm.weight,
                        normed_out_i,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim,
                        d_conv,
                        1e-6,
                        eps,
                        stream,
                    )?;
                    if let Some(t0) = sub_t0 {
                        ctx.gpu.synchronize(stream).ok();
                        rec_gdn_us += t0.elapsed().as_micros();
                    }
                } else if self.gdn_f32_norm_k.0 != 0
                    && std::env::var("ATLAS_GDN_FUSED_NORM").ok().as_deref() == Some("1")
                {
                    ops::gdn_decode_f32_norm(
                        ctx.gpu,
                        self.gdn_f32_norm_k,
                        ssm_state.h_state,
                        q_conv,
                        k_conv,
                        v_conv,
                        gates,
                        beta_fp32,
                        z_i,
                        self.ssm.norm.weight,
                        normed_out_i,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        eps,
                        stream,
                    )?;
                    if let Some(t0) = sub_t0 {
                        ctx.gpu.synchronize(stream).ok();
                        rec_gdn_us += t0.elapsed().as_micros();
                    }
                } else {
                    ops::gdn_decode(
                        ctx.gpu,
                        self.gdn_f32_k,
                        ssm_state.h_state,
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
                    if let Some(t0) = sub_t0 {
                        ctx.gpu.synchronize(stream).ok();
                        rec_gdn_us += t0.elapsed().as_micros();
                    }

                    let sub_t0 = if detail_profile {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    ops::gated_rms_norm(
                        ctx.gpu,
                        self.gated_rms_norm_f32_k,
                        gdn_out,
                        z_i,
                        &self.ssm.norm,
                        normed_out_i,
                        nv as u32,
                        vd as u32,
                        vd as u32,
                        eps,
                        vd as u32,
                        stream,
                    )?;
                    if let Some(t0) = sub_t0 {
                        ctx.gpu.synchronize(stream).ok();
                        rec_norm_us += t0.elapsed().as_micros();
                    }
                }
            }
            if detail_profile {
                detail_parts.push(("recurrent_ba", rec_ba_us));
                detail_parts.push(("recurrent_conv", rec_conv_us));
                detail_parts.push(("recurrent_gdn", rec_gdn_us));
                if rec_norm_us > 0 {
                    detail_parts.push(("recurrent_norm", rec_norm_us));
                }
                *detail_t0 = Some(std::time::Instant::now());
            }
        }

        Ok(())
    }
}
