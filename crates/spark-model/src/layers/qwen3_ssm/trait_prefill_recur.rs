// SPDX-License-Identifier: AGPL-3.0-only

//! GDN recurrence kernel dispatch for `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC
//! cap. [`Qwen3SsmLayer::prefill_gdn_recurrence`] mirrors the original
//! step 8 block 1:1 — same WY4-persistent / single-token persistent /
//! split4 dispatch, same env overrides, same kernel launches.

use super::*;

impl Qwen3SsmLayer {
    /// GDN prefill recurrence via the WY4-persistent kernel.
    ///
    /// Dispatch: FLA chunked prefill (baked default, 128-dim linear heads) →
    /// WY4-persistent (4 tokens/iter, H in shared memory) → single-token persistent
    /// (256..=4096) → split4 for unsupported configurations.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gdn_recurrence(
        &self,
        h_state: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gates_buf: DevicePtr,
        gdn_out_buf: DevicePtr,
        k: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        conv_dim: usize,
        midcap_idx: Option<usize>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let fp32 = 4usize;
        let gb_stride = (nv * 2) as u32;

        // gfx1151/SCALE (atlas_scale): every H-in-shared-memory GDN prefill
        // kernel exceeds RDNA3.5's 64KB LDS cap — FLA (C=64) ≈96KB, WY4 =69688,
        // persistent =67584. Only split4 keeps the kd*vd H-state in global
        // memory (~2KB smem) and handles arbitrary length, so route there for
        // all sizes. Correctness-equivalent, lower throughput; the smem-H fast
        // paths (and a future C=32 FLA variant) are Blackwell-only. NVIDIA
        // (cfg unset) takes the full FLA/WY ladder below unchanged.
        if cfg!(atlas_scale) {
            // MID-CHUNK tail capture: split the split4 recurrence at cap_local
            // (and, when present, cap_local - bs), D2D each captured h_state
            // into its reserved slot, then finish the trailing tokens with
            // h_state chained. Byte-identical to a single call — same algebra as
            // the >4096 sub-chunk loop below.
            if let (Some(cap), Some(idx)) = (ctx.midchunk_capture.as_ref(), midcap_idx) {
                let cl = cap.cap_local;
                if cl > 0 && (cl as u32) < k {
                    let bf16 = 2usize;
                    let value_dim = nv * vd;
                    // Run split4 over local tokens [start, start+len); h_state is
                    // the SAME (chained) across calls — offsets mirror the >4096
                    // sub-chunk loop's stride arithmetic.
                    let seg = |start: usize, len: u32| -> Result<()> {
                        let gate = gates_buf.offset(start * gb_stride as usize * fp32);
                        ops::gdn_prefill_split4(
                            ctx.gpu,
                            self.gdn_prefill_split4_k,
                            h_state,
                            q_ptr.offset(start * conv_dim * bf16),
                            k_ptr.offset(start * conv_dim * bf16),
                            v_ptr.offset(start * conv_dim * bf16),
                            gate,
                            gate.offset(nv * fp32),
                            gdn_out_buf.offset(start * value_dim * bf16),
                            1,
                            len,
                            nk as u32,
                            nv as u32,
                            kd as u32,
                            vd as u32,
                            conv_dim as u32,
                            conv_dim as u32,
                            gb_stride,
                            stream,
                        )
                    };
                    // Optional EARLIER capture at cap_local - bs (token tb - bs).
                    let mut start = 0usize;
                    if let Some(ce) = cap.cap_local_early {
                        seg(0, ce as u32)?;
                        ctx.gpu.copy_d2d_async(
                            h_state,
                            cap.h_dsts_early[idx],
                            cap.h_bytes,
                            stream,
                        )?;
                        start = ce;
                    }
                    // Capture h_state @ the tail boundary tb.
                    seg(start, (cl - start) as u32)?;
                    ctx.gpu
                        .copy_d2d_async(h_state, cap.h_dsts[idx], cap.h_bytes, stream)?;
                    // Trailing tokens [cap_local, k).
                    return seg(cl, k - cl as u32);
                }
            }
            return ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
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

        // FlashInfer GDN (opt-in, ATLAS_GDN_FLASHINFER=1): tensor-core chunked delta-rule
        // scan, ~11× the scalar FLA chunk_delta_h at the Holo shape. This is the live
        // single-stream prefill path (trait_prefill.rs -> prefill_gdn_recurrence). q_ptr is
        // the packed-QKV base, gates_buf the gate base — handed straight to the bit-exact
        // shim (ops::gdn_flashinfer). FLA ladder below is the fallback when flag/lib absent.
        if !ctx.gdn_exact_replay && kd == 128 && vd == 128 && ops::gdn_flashinfer::available() {
            let scale = 1.0f32 / (kd as f32).sqrt();
            return ops::gdn_flashinfer::flashinfer_gdn_prefill(
                ctx.gpu,
                q_ptr,
                gates_buf,
                gdn_out_buf,
                h_state,
                scale,
                k,
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

        // 2026-06-06: removed the concluded GDN-prefill experiment env flags
        // (ATLAS_GDN_CHUNK64 / ATLAS_FORCE_PERSISTENT / ATLAS_DISABLE_WY4) and their
        // dispatch branches. FLA is the baked default for 128-dim linear heads; the
        // WY4-persistent kernel is the unconditional fallback below.
        // FLA multi-kernel chunked prefill (recompute_wu → chunk_delta_h_ksplit →
        // chunk_fwd_o): 1.75x vs wy4 @16k, token-equal (cos=1.0 vs scalar). BAKED
        // DEFAULT 2026-06-06 (was gated behind ATLAS_GDN_FLA=1 — the env var is gone):
        // always taken for 128-dim linear-head GDN models when the FLA kernels & scratch
        // are present (scratch is allocated for exactly those models, sizes.rs). The wy4
        // branch below remains the fallback for other head dims / a guard miss.
        // Warm-hit replay (Marconi SSM snapshot restored): force the WY4
        // recurrence. FLA's chunked algebra is only token-equal when its
        // 64-token grid matches the pass that originally produced the cached
        // K/V; a replay anchored at an arbitrary snapshot offset regroups the
        // recurrence and its bf16 W/U/uc/S_c intermediates drift. The replay
        // range is rewritten into SHARED prefix-cache blocks, so non-exact
        // recompute poisons them and the drift ratchets across agentic turns
        // (token-stutter corruption, 2026-06-10). WY4 keeps H in FP32 SMEM
        // token-sequentially — same family as the decode kernel — and is the
        // path the clean pre-FLA baseline used. Replay segments are short
        // (suffix after a ≥10k skipped prefix), so the FLA speed loss is nil.
        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        if !ctx.gdn_exact_replay
            && kd == 128
            && vd == 128
            && fla_scratch.0 != 0
            && self.gdn_prefill_fla_recompute_wu_k.0 != 0
            && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
            && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
        {
            // One-time positive signal that the FLA path is live (vs silently
            // falling through to wy4 on a guard miss) — greppable in the server log.
            static FLA_LOG: std::sync::Once = std::sync::Once::new();
            FLA_LOG.call_once(|| {
                tracing::info!(
                    "GDN prefill: FLA chunked path ACTIVE (baked default: recompute_wu → chunk_delta_h_ksplit → chunk_fwd_o)"
                );
            });
            let num_chunks = k.div_ceil(64);
            let nt = num_chunks as usize;
            let w_out = fla_scratch;
            let u_out = w_out.offset(nt * nv * 64 * kd * 2);
            let s_out = u_out.offset(nt * nv * 64 * vd * 2);
            let uc_out = s_out.offset(nt * nv * kd * vd * 2);
            let gc_out = uc_out.offset(nt * nv * 64 * vd * 2);
            ops::gdn_prefill_fla(
                ctx.gpu,
                self.gdn_prefill_fla_recompute_wu_k,
                self.gdn_prefill_fla_chunk_delta_h_k,
                self.gdn_prefill_fla_chunk_delta_h_tc_vblock_k,
                self.gdn_prefill_fla_chunk_fwd_o_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                w_out,
                u_out,
                s_out,
                uc_out,
                gc_out,
                1,
                k,
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
            )?;
        } else if std::env::var_os("ATLAS_GDN_REGRESIDENT").is_some()
            && kd == 128
            && vd == 128
            && self.gdn_prefill_regresident_k.0 != 0
        {
            // Register-resident token-sequential recurrence — drop-in for WY4 on
            // the warm Marconi-replay path (this branch is only reached when the
            // FLA `if` above fell through, i.e. gdn_exact_replay). H lives in
            // registers (one warp per v-column, 4 k-rows/lane) instead of 64KB
            // smem, so >=2 CTA/SM and no per-token barriers. Token-equal to WY4
            // (cosine 1.0, max|dH|~1e-8 — same acceptance class) and ~2.9x faster
            // in isolation. Gated by ATLAS_GDN_REGRESIDENT until serve-validated.
            static RR_LOG: std::sync::Once = std::sync::Once::new();
            RR_LOG.call_once(|| {
                tracing::info!(
                    "GDN prefill: REGISTER-RESIDENT warm-replay path ACTIVE (ATLAS_GDN_REGRESIDENT; H in regs, no smem-H)"
                );
            });
            ops::gdn_prefill_regresident(
                ctx.gpu,
                self.gdn_prefill_regresident_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if self.gdn_prefill_persistent_wy4_k.0 != 0 {
            // WY4-persistent: H in shared memory, 4 tokens per iteration
            // smem = H[K_DIM*V_DIM] + 8*k/q buffers + warp sums + WY scalars
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
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
        } else if (256..=4096).contains(&k) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
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
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
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

    /// Conv1d prefill with optional MID-CHUNK tail capture. When capturing,
    /// splits the sliding-window conv at `cap_local`, D2D-copies the @tb
    /// conv_state into the reserved snapshot slot, then finishes the trailing
    /// tokens (conv_state chained). Byte-identical to a single call otherwise —
    /// the sliding window carried in conv_state makes the split exact (same
    /// contract multi-chunk prefill already relies on).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn conv1d_prefill_capture(
        &self,
        ctx: &ForwardContext,
        conv_state: DevicePtr,
        input: DevicePtr,
        output: DevicePtr,
        conv_dim: usize,
        d_conv: usize,
        k: u32,
        qkvz_size: usize,
        midcap_idx: Option<usize>,
        stream: u64,
    ) -> Result<()> {
        if let (Some(cap), Some(idx)) = (ctx.midchunk_capture.as_ref(), midcap_idx) {
            let cl = cap.cap_local;
            if cl > 0 && (cl as u32) < k {
                let bf16 = 2usize;
                // Run the sliding-window conv over local tokens [start, start+len);
                // conv_state is chained across calls (the same contract multi-chunk
                // prefill relies on), so the split is byte-exact.
                let seg = |start: usize, len: u32| -> Result<()> {
                    ops::conv1d_update_prefill(
                        ctx.gpu,
                        self.conv1d_prefill_k,
                        conv_state,
                        input.offset(start * qkvz_size * bf16),
                        &self.ssm.conv1d,
                        DevicePtr::NULL,
                        output.offset(start * conv_dim * bf16),
                        conv_dim as u32,
                        d_conv as u32,
                        len,
                        qkvz_size as u32,
                        conv_dim as u32,
                        stream,
                    )
                };
                // Optional EARLIER capture at cap_local - bs (token tb - bs).
                let mut start = 0usize;
                if let Some(ce) = cap.cap_local_early {
                    seg(0, ce as u32)?;
                    ctx.gpu.copy_d2d_async(
                        conv_state,
                        cap.conv_dsts_early[idx],
                        cap.conv_bytes,
                        stream,
                    )?;
                    start = ce;
                }
                // Capture conv_state @ the tail boundary tb.
                seg(start, (cl - start) as u32)?;
                ctx.gpu
                    .copy_d2d_async(conv_state, cap.conv_dsts[idx], cap.conv_bytes, stream)?;
                // Trailing tokens [cap_local, k).
                seg(cl, k - cl as u32)?;
                return Ok(());
            }
        }
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            conv_state,
            input,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            output,
            conv_dim as u32,
            d_conv as u32,
            k,
            qkvz_size as u32,
            conv_dim as u32,
            stream,
        )
    }
}
