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
            let uc_out = s_out.offset(nt * nv * kd * vd * 4);
            ops::gdn_prefill_fla(
                ctx.gpu,
                self.gdn_prefill_fla_recompute_wu_k,
                self.gdn_prefill_fla_chunk_delta_h_k,
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
}
