// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_gdn_full.

use super::*;

mod batched;

/// `ATLAS_GDN_BATCHED_FLA=1` → route the co-dispatch batched GDN scan through the
/// chunk-parallel FLA kernels at batch=N (fills chunk_delta_h's 32→32N CTAs)
/// instead of the occupancy-starved wy64 family. Default off (opt-in A/B).
pub(super) fn gdn_batched_fla_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_GDN_BATCHED_FLA").ok().as_deref() == Some("1"))
}

impl Qwen3SsmLayer {
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
        tracing::debug!(
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
        // FLA chunked GDN (recompute_wu → chunk_delta_h_ksplit → chunk_fwd_o) —
        // the same baked-default path single-stream prefill uses
        // (trait_prefill_recur.rs). Its grid [num_chunks, nv, batch] yields
        // num_chunks×nv blocks vs wy32's flat ~nv(32), so it fills GB10's 48 SMs
        // far better. The batched/co-dispatch per-request loop calls THIS fn, and
        // nsys showed it GDN-bound at ~45% of batched-prefill GPU time on the
        // occupancy-starved wy64 — routing per-request GDN through FLA is the
        // batching lever. Skipped on exact-replay (FLA's 64-tok regrouping drifts
        // vs a snapshot-anchored pass) and non-128-dim heads.
        // FlashInfer GDN (opt-in, ATLAS_GDN_FLASHINFER=1): tensor-core chunked delta-rule
        // scan, ~11× the scalar FLA chunk_delta_h at the Holo shape. Single-stream only;
        // takes Atlas's native packed-QKV + interleaved gate/beta directly (see
        // ops::gdn_flashinfer). FLA path below is the fallback when the flag/lib is absent.
        if !ctx.gdn_exact_replay && kd == 128 && vd == 128 && ops::gdn_flashinfer::available() {
            let scale = 1.0f32 / (kd as f32).sqrt();
            return ops::gdn_flashinfer::flashinfer_gdn_prefill(
                ctx.gpu,
                gdn_bufs.qkv,
                gdn_bufs.gate_beta,
                gdn_bufs.output,
                ssm_state.h_state,
                scale,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                gb_stride,
                1,
                stream,
            );
        }

        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        if !ctx.gdn_exact_replay
            && kd == 128
            && vd == 128
            && fla_scratch.0 != 0
            && self.gdn_prefill_fla_recompute_wu_k.0 != 0
            && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
            && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
        {
            let num_chunks = total.div_ceil(64);
            let nt = num_chunks as usize;
            let w_out = fla_scratch;
            let u_out = w_out.offset(nt * nv * 64 * kd * bf16);
            let s_out = u_out.offset(nt * nv * 64 * vd * bf16);
            let uc_out = s_out.offset(nt * nv * kd * vd * bf16);
            let gc_out = uc_out.offset(nt * nv * 64 * vd * bf16);
            return ops::gdn_prefill_fla(
                ctx.gpu,
                self.gdn_prefill_fla_recompute_wu_k,
                self.gdn_prefill_fla_chunk_delta_h_k,
                self.gdn_prefill_fla_chunk_delta_h_tc_vblock_k,
                self.gdn_prefill_fla_chunk_fwd_o_k,
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                w_out,
                u_out,
                s_out,
                uc_out,
                gc_out,
                1,
                total,
                num_chunks,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                false, // single-stream: contiguous h_state (not a pointer table)
                spark_runtime::gpu::DevicePtr::NULL, // cu_seqlens (unused)
                spark_runtime::gpu::DevicePtr::NULL, // cu_chunks (unused)
                false, // not varlen
                ctx.profile,
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

        Ok(())
    }
}
