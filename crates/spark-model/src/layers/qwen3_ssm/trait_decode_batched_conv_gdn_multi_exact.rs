// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-verify body for the batched (`GdnStates::Multi`) MTP verify
//! (issue #435 route (a), PR1).
//!
//! Per token position `t`, TWO strided launches cover all `n` sequences:
//!
//! 1. `causal_conv1d_update_l2norm_f32_strided` at batch = n — the SAME
//!    kernel the batched-recurrent decode path runs, whose per-sequence math
//!    is line-for-line the non-strided `causal_conv1d_update_l2norm_f32`
//!    (strides only move base addresses).
//! 2. `gated_delta_rule_decode_f32_strided_norm_snap` at batch = n — the
//!    strided fused-norm decode kernel (again per-sequence identical to the
//!    single-seq `..._f32_norm`) with the per-token h rollback snapshot
//!    written inline.
//!
//! Both per-sequence bodies are bitwise the single-token decode chain, so
//! this arm is bitwise-equal to the per-sequence exact loop it accelerates —
//! and to sequential decode — regardless of whether spec-off decode at C=n
//! runs the per-sequence (`ssm_forward`) or batched-recurrent strided form.
//!
//! Conv rollback snapshots stay `copy_d2d_async` per sequence (n·(K-1)
//! copies); the h snapshots ride inline in the `_snap` launch.
//!
//! `Ok(false)` declines — non-contiguous pool slots, missing kernels
//! (`_snap` is model-shadow staged), or `--gdn-fused-norm` absent (the
//! strided twin only exists on the fused-norm arm) — and the caller runs the
//! per-sequence loop, which under `verify_exact_enabled()` is the per-token
//! exact arm: byte-identical math, more launches. All decisions are pure
//! functions of (flags, handles, the slot vector's fixed addresses), so the
//! outcome is CUDA-graph-stable for a slot-vector-keyed graph.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::LayerState;
use crate::layers::ops;

/// Per-sequence device pointers gathered (and validated) before any launch.
struct ExactMultiSeq {
    conv_state: DevicePtr,
    conv_inter: Vec<DevicePtr>,
}

impl Qwen3SsmLayer {
    /// Attempt the two-launch-per-token exact batched verify. See the module
    /// docs; `args` carries the run's base (row-offset) buffers with
    /// `args.num_tokens = k`.
    pub(super) fn decode_batched_conv_gdn_multi_exact(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
        ctx: &crate::layer::ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<bool> {
        let n = states.len();
        let kk = args.num_tokens;
        let conv_bytes = self.conv_state_bytes;
        let h_bytes = self.h_state_bytes;

        // The strided fused-norm snap twin is the only batched form of the
        // exact chain; everything else declines to the per-sequence exact
        // loop (still exact — see module docs).
        if n < 2
            || !crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
            || self.conv1d_l2norm_f32_strided_k.0 == 0
            || self.gdn_f32_strided_norm_snap_k.0 == 0
        {
            return Ok(false);
        }

        // ── Preconditions on the ACTUAL pool pointers (never assumed) ──
        // The strided kernels infer DENSE per-sequence strides for conv_state
        // (conv_dim·d_conv FP32) and h_state (nv·kd·vd FP32), so both must sit
        // on consecutive slots; the h intermediates must be intra-slot
        // contiguous with a UNIFORM cross-sequence stride.
        let mut seqs: Vec<ExactMultiSeq> = Vec::with_capacity(n);
        let mut conv_base = DevicePtr::NULL;
        let mut h_base = DevicePtr::NULL;
        let mut h_inter_base = DevicePtr::NULL;
        let mut h_inter_seq_stride = 0u64;
        for (i, state) in states.iter().enumerate() {
            let Some(st) = state.as_any().downcast_ref::<SsmLayerState>() else {
                return Ok(false);
            };
            // h side: kk-1 intermediates (the kernel writes indices
            // 0..kk-2 and NULL-skips the dead kk-1 — see below; the pool
            // allocates exactly K-1 per slot since the K-1 shrink).
            if st.conv_state_intermediates.len() < kk || st.h_state_intermediates.len() < kk - 1 {
                return Ok(false);
            }
            let hi0 = st.h_state_intermediates[0];
            for t in 1..kk - 1 {
                if st.h_state_intermediates[t].0 != hi0.0 + (t * h_bytes) as u64 {
                    return Ok(false);
                }
            }
            if i == 0 {
                conv_base = st.conv_state;
                h_base = st.h_state;
                h_inter_base = hi0;
            } else {
                if st.conv_state.0 != conv_base.0 + (i * conv_bytes) as u64
                    || st.h_state.0 != h_base.0 + (i * h_bytes) as u64
                {
                    return Ok(false);
                }
                if i == 1 {
                    h_inter_seq_stride = hi0.0.wrapping_sub(h_inter_base.0);
                    // The kernel writes kk-1 dense snapshots per sequence;
                    // they must not overlap the next sequence's region.
                    if h_inter_seq_stride < ((kk - 1) * h_bytes) as u64 {
                        return Ok(false);
                    }
                } else if hi0.0 != h_inter_base.0 + (i as u64) * h_inter_seq_stride {
                    return Ok(false);
                }
            }
            seqs.push(ExactMultiSeq {
                conv_state: st.conv_state,
                conv_inter: st.conv_state_intermediates[..kk].to_vec(),
            });
        }

        let ConvGdnArgs {
            deinterleaved,
            gates_buf,
            normed_out,
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
            ..
        } = *args;
        let eps = ctx.config.rms_norm_eps as f32;
        // FP32 conv scratch: one row per SEQUENCE at qkvz_size FP32 stride,
        // reused across token positions (each is consumed by the GDN launch
        // that follows it on the same stream). n rows ≤ the buffer's n·k-row
        // capacity by construction.
        let conv_scratch = ctx.buffers.ssm_conv_out_f32();

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "EXACT batched MTP verify ENGAGED (#435, opt-in --exact-verify): \
                 per-token strided conv_f32 + strided fused-norm snap at batch=n \
                 (2 launches + n conv d2d per position); omit the flag for the \
                 default WY arms"
            );
        });

        for t in 0..kk {
            // ── 1 launch: conv1d+SiLU+L2norm, FP32 out, all n sequences ──
            ops::conv1d_update_l2norm_strided(
                ctx.gpu,
                self.conv1d_l2norm_f32_strided_k,
                conv_base,
                deinterleaved.offset(t * qkvz_size * bf16),
                &self.ssm.conv1d,
                conv_scratch,
                conv_dim as u32,
                d_conv as u32,
                n as u32,
                qk_ch,
                kd as u32,
                1e-6,
                (kk * qkvz_size) as u32, // input seq stride (BF16 elems)
                qkvz_size as u32,        // output seq stride (FP32 elems)
                stream,
            )?;

            // ── 1 launch: GDN + gated norm + inline h snapshot, batch = n ──
            let snapshot = t + 1 < kk;
            let (h_inter_t, h_inter_stride_elems) = if snapshot {
                (h_inter_base.offset(t * h_bytes), h_inter_seq_stride / 4)
            } else {
                // Index kk-1 has no reader (see the reader enumeration in
                // trait_decode_batched_conv_gdn.rs) — NULL skips the stores.
                (DevicePtr::NULL, 0)
            };
            let gate_t = gates_buf.offset(t * nv * 2 * fp32);
            ops::gdn_decode_f32_strided_norm_snap(
                ctx.gpu,
                self.gdn_f32_strided_norm_snap_k,
                h_base,
                conv_scratch,
                conv_scratch.offset(key_dim * fp32),
                conv_scratch.offset(key_dim * 2 * fp32),
                gate_t,
                gate_t.offset(nv * fp32),
                deinterleaved.offset(t * qkvz_size * bf16 + conv_dim * bf16),
                self.ssm.norm.weight,
                normed_out.offset(t * value_dim * bf16),
                h_inter_t,
                h_inter_stride_elems,
                n as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,        // qk_stride (FP32 conv rows)
                qkvz_size as u32,        // v_stride
                (kk * nv * 2) as u32,    // gb_stride (seq-major gate rows)
                (kk * qkvz_size) as u32, // z_stride (seq-major deint rows)
                (kk * value_dim) as u32, // out_stride (seq-major normed rows)
                eps,
                stream,
            )?;

            // ── Conv rollback snapshots (d2d per sequence; index kk-1 dead) ──
            if snapshot {
                for seq in &seqs {
                    ctx.gpu.copy_d2d_async(
                        seq.conv_state,
                        seq.conv_inter[t],
                        conv_bytes,
                        stream,
                    )?;
                }
            }
        }

        Ok(true)
    }
}
