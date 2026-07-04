// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_gdn_full.

use super::*;

impl Qwen3SsmLayer {
    /// Single-pass WY16 GDN for the K=16 DFlash verify (replaces WY4+replay).
    /// Reads gdn_bufs (post-conv q/k/v/gate/beta), starts from the pre-verify
    /// h_state (= checkpoint, since phase-1 conv doesn't touch h_state), and
    /// writes the final h_state live + Hi_0..Hi_14 into h_state_intermediates.
    /// Returns Ok(true) when the fused pass ran, Ok(false) to fall back.
    pub(super) fn prefill_gdn_wy16_inner(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if gdn_bufs.total_len != 16 || self.gdn_wy16_k.0 == 0 {
            return Ok(false);
        }
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let conv_dim = key_dim * 2 + nv * vd;
        let h_bytes = nv * vd * kd * 4;

        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let inter_stride_floats = (h_bytes / 4) as u32;

        // ATLAS_DFLASH_VERIFY_GDN_F32 prototype: closes the BF16-truncation
        // gap in the K=γ verify path's dominant dispatch branch (production
        // verify batches are total_len=γ+1=16, which this fast path owns
        // exclusively). Mirrors the split4_f32 dispatch below; sets
        // output_f32_written so phase3 knows the buffer holds fresh data.
        let use_f32_gdn = self.gdn_wy16_f32_k.0 != 0
            && !gdn_bufs.output_f32.is_null()
            && std::env::var("ATLAS_DFLASH_VERIFY_GDN_F32").ok().as_deref() == Some("1");
        let (kernel, out_ptr) = if use_f32_gdn {
            (self.gdn_wy16_f32_k, gdn_bufs.output_f32)
        } else {
            (self.gdn_wy16_k, gdn_bufs.output)
        };
        gdn_bufs.output_f32_written.set(use_f32_gdn);

        // gdn_decode_wy17 is kernel-agnostic; the wy16 handle drives a K_TOKENS=16
        // single pass writing final H (live) + Hi_0..Hi_14 (intermediates).
        ops::gdn_decode_wy17(
            ctx.gpu,
            kernel,
            ssm_state.h_state,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            out_ptr,
            ssm_state.h_state_intermediates[0],
            inter_stride_floats,
            1,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32,
            conv_dim as u32,
            (nv * 2) as u32,
            stream,
        )?;
        Ok(true)
    }

    pub(super) fn prefill_gdn_full_inner(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        // ATLAS_DFLASH_VERIFY_GDN_F32: default to "not written" for every
        // branch in this ladder; only the split4-f32 branch below flips it
        // back to true. Required because gdn_bufs (and its Cell) is shared
        // across SSM layers within a verify call — without resetting here,
        // a layer that falls through to WY32/persistent/chunked (all BF16)
        // could inherit a stale `true` left by prefill_gdn_wy16_inner's
        // dispatch on a previous layer.
        gdn_bufs.output_f32_written.set(false);

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let total = gdn_bufs.total_len as u32;

        // Packed QKV layout: Q at offset 0, K at key_dim, V at key_dim*2
        // Strides: qk_stride = conv_dim, v_stride = conv_dim (elements, not bytes)
        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);

        // Gate/beta: interleaved [total_len, 2*nv] FP32
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        // WY32 persistent: processes 32 tokens per WY iteration with H in
        // shared memory (~84KB). ~30× faster than per-token for 14k+ sequences.
        // Falls through to WY4 or sub-chunked persistent for shorter sequences.
        tracing::info!(
            "GDN prefill: total={total} wy32_k={} wy4_k={} persistent_k={} split4_k={}",
            self.gdn_prefill_wy32_k.0 != 0,
            self.gdn_prefill_persistent_wy4_k.0 != 0,
            self.gdn_prefill_persistent_k.0 != 0,
            self.gdn_prefill_split4_k.0 != 0
        );
        // gfx1151/SCALE (atlas_scale): every H-in-shared-memory GDN prefill
        // kernel exceeds RDNA3.5's hard 64KB LDS cap — WY32 ~84KB, WY4 =69688,
        // persistent =67584 (cuFuncSetAttribute(MAX_DYNAMIC_SHARED) →
        // CUDA_ERROR_INVALID_VALUE). Only split4 keeps the kd*vd H-state in
        // global memory (~2KB smem) and handles arbitrary length, so route
        // there for all sizes. Correctness-equivalent, lower throughput; the
        // smem-H fast paths are a Blackwell-only optimization. NVIDIA (cfg
        // unset) takes the full ladder below unchanged.
        if cfg!(atlas_scale) {
            return ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            );
        }
        if self.gdn_prefill_wy32_k.0 != 0 && total > 32 && !cfg!(atlas_scale) {
            // #110: dynamic smem must cover the FULL kernel layout (H + smem_k +
            // smem_q + smem_warp[4] + smem_kd[C*C] + smem_g[C] + smem_bt[C], C=32).
            // The old `+256` slack under-counted the smem_warp(16)+smem_g(128)+
            // smem_bt(128)=272 trailer by 16 B, so the kernel's smem_bt tail wrote
            // past the requested allocation → CUDA illegal access under live
            // occupancy (compute-sanitizer: Invalid __shared__ write at +0xce0).
            let smem =
                (kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 32 * 32 * 4 + (4 + 32 + 32) * 4) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_wy32_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if total > 4096 {
            // Sub-chunk fallback for >4096 tokens when WY32 isn't available.
            let chunk_max = 4096u32;
            let mut offset = 0u32;
            while offset < total {
                let chunk = (total - offset).min(chunk_max);
                let q_chunk = q_ptr.offset(offset as usize * conv_dim * bf16);
                let k_chunk = k_ptr.offset(offset as usize * conv_dim * bf16);
                let v_chunk = v_ptr.offset(offset as usize * conv_dim * bf16);
                let gate_chunk = gate_ptr.offset(offset as usize * gb_stride as usize * fp32);
                let beta_chunk = beta_ptr.offset(offset as usize * gb_stride as usize * fp32);
                let out_chunk = gdn_bufs.output.offset(offset as usize * value_dim * bf16);

                if self.gdn_prefill_persistent_k.0 != 0 && chunk >= 256 {
                    ops::gdn_prefill_persistent(
                        ctx.gpu,
                        self.gdn_prefill_persistent_k,
                        ssm_state.h_state,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        gate_chunk,
                        beta_chunk,
                        out_chunk,
                        1,
                        chunk,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                } else {
                    ops::gdn_prefill_split4(
                        ctx.gpu,
                        self.gdn_prefill_split4_k,
                        ssm_state.h_state,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        gate_chunk,
                        beta_chunk,
                        out_chunk,
                        1,
                        chunk,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                }
                offset += chunk;
            }
        } else if self.gdn_prefill_persistent_wy4_k.0 != 0 && !cfg!(atlas_scale) {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&total) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            // ATLAS_DFLASH_VERIFY_GDN_F32 prototype: closes the BF16-truncation
            // gap in the K=γ verify path's split4 dispatch (the branch this
            // model target's K=15 verify actually takes). Only covers this
            // ladder branch — WY32/persistent/chunked callers above stay BF16.
            let use_f32_gdn = self.gdn_prefill_split4_f32_k.0 != 0
                && !gdn_bufs.output_f32.is_null()
                && std::env::var("ATLAS_DFLASH_VERIFY_GDN_F32").ok().as_deref() == Some("1");
            gdn_bufs.output_f32_written.set(use_f32_gdn);
            if use_f32_gdn {
                ops::gdn_prefill_split4(
                    ctx.gpu,
                    self.gdn_prefill_split4_f32_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_bufs.output_f32,
                    1,
                    total,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    stream,
                )?;
            } else {
                ops::gdn_prefill_split4(
                    ctx.gpu,
                    self.gdn_prefill_split4_k,
                    ssm_state.h_state,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_bufs.output,
                    1,
                    total,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    stream,
                )?;
            }
        }

        Ok(())
    }

    /// Q12 Path B: batched GDN recurrence — mirrors prefill_gdn_full_inner
    /// dispatch ladder but routes to the `*_batched` kernel variants and
    /// passes `h_state_ptrs` (device array of N pointers) instead of a
    /// single h_state device pointer.
    ///
    /// Constraint: scheduler-enforced same-chunk-len across all N streams.
    /// `gdn_bufs.qkv` / `gate_beta` / `output` are stacked
    /// `[batch_size, chunk_len, *]` contiguous in memory. Each batch
    /// element's QKV starts at `b * chunk_len * conv_dim` (BF16).
    ///
    /// Validation status: kernels unvalidated against hardware.
    pub(super) fn prefill_gdn_full_batched_inner(
        &self,
        h_state_ptrs: spark_runtime::gpu::DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        // Mirror the single-stream dispatch ladder. Total tokens per stream
        // is `chunk_len`; the kernel internally processes `batch_size` such
        // streams (grid dim Y).
        if self.gdn_prefill_wy32_batched_k.0 != 0 && chunk_len > 32 {
            // #110: dynamic smem must cover the FULL kernel layout (H + smem_k +
            // smem_q + smem_warp[4] + smem_kd[C*C] + smem_g[C] + smem_bt[C], C=32).
            // The old `+256` slack under-counted the smem_warp(16)+smem_g(128)+
            // smem_bt(128)=272 trailer by 16 B, so the kernel's smem_bt tail wrote
            // past the requested allocation → CUDA illegal access under live
            // occupancy (compute-sanitizer: Invalid __shared__ write at +0xce0).
            let smem =
                (kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 32 * 32 * 4 + (4 + 32 + 32) * 4) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_wy32_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if self.gdn_prefill_persistent_wy4_batched_k.0 != 0 {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&chunk_len) && self.gdn_prefill_persistent_batched_k.0 != 0
        {
            ops::gdn_prefill_persistent_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if self.gdn_prefill_split4_batched_k.0 != 0 {
            ops::gdn_prefill_split4_batched(
                ctx.gpu,
                self.gdn_prefill_split4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            anyhow::bail!(
                "Qwen3SsmLayer::prefill_gdn_full_batched_inner: no batched GDN \
                 kernel handle is loaded for this target — caller should fall \
                 back to per-stream prefill_gdn_full."
            );
        }

        Ok(())
    }
}
