// SPDX-License-Identifier: AGPL-3.0-only

//! Exact MTP-verify conv+GDN arm (issue #435 route (a)) — reached from
//! `--exact-verify`, and (2026-09-03) from any pass that declares
//! `ForwardContext::gdn_exact_replay`.
//!
//! For most verify bodies this arm does NOT run by default: they take the
//! WY/fused chunkwise arms, so spec-on output is NOT bitwise-equal to spec-off
//! at temp 0 — #435's divergence remains with default settings, and
//! `--exact-verify` is the flag that removes it (measured decode-step cost
//! ~+35% at n=8/K=4, ~+22% at n=16/K=2, ~+36% at n=32/K=2; every surveyed
//! production engine ships exactness as the same kind of opt-in).
//!
//! The mHC MTP verify (`model/trait_impl/verify_hc.rs`) is the exception: its
//! row 0 re-processes an already-committed token, so it declares
//! `gdn_exact_replay` and gets this arm unconditionally. Measured on
//! qwen3.8-flash-next (native EXL3, gamma=1, 40-token probe): with the WY arms
//! the verify's row-0 logits matched serial decode 0/38; with this arm plus
//! the other two row-exact legs, 38/38. Kill switch `ATLAS_NO_VERIFY_ROW_GDN`.
//!
//! The WY/fused verify arms diverge in two ways: the verify conv runs the
//! BF16-output kernel where sequential decode runs the FP32 one (h-state
//! relL2 ~8.6e-4 after one K=4 window — the dominant term), and the WY
//! chunkwise reordering differs from the recurrent update by ~3.4e-8 — four
//! orders smaller but nonzero, and an argmax flip only needs a per-logit
//! error above a thin top-2 margin.
//!
//! This arm runs, per verify token, EXACTLY the kernel chain `ssm_forward`
//! (the single-token decode reference) runs — same handles, same launch
//! geometry, same argument values — so the h-state, conv-state and normed
//! output are bitwise what sequential decode would have produced from the
//! same inputs. Decode dispatch is untouched anywhere in this change, so
//! spec-OFF is bit-unchanged by construction.
//!
//! Rollback snapshots: the `_snap` kernel twin
//! (`gated_delta_rule_decode_f32_norm_snap`, model-shadow staged) writes the
//! per-token h-state intermediate inline — the same bits the recurrence just
//! committed, following the `gdn_verify_fused_conv_kn.cu` /
//! `gated_delta_rule_wy4.cu` precedent — and the FP32 fused verify conv
//! (`gdn_verify_fused_conv_kn_f32`) writes the conv snapshots inline. Where
//! either handle is 0 the arm falls back to the parent kernel plus a
//! `copy_d2d_async` per token: identical bits, more launches.
//!
//! The arm also writes the FINAL gated-RMS-norm output (fused or per-token,
//! mirroring the decode dispatch), so `decode_batched_inner` must SKIP its
//! phase-8 norm when `verify_exact_enabled()` — both sites read that one
//! predicate.
//!
//! Kill switch: omit `--exact-verify` (the default) to run the WY/fused arms.
//! Every dispatch decision below is a pure function of (process-static flags,
//! kernel handles, the slot's fixed intermediate addresses), so it is
//! CUDA-graph-stable.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Byte offsets for verify token `t` in the exact arm's buffers, plus whether
/// `t` writes rollback intermediates. Pure — stride arithmetic is the failure
/// mode of this arm, so it is factored out and unit-tested below.
pub(super) struct ExactRow {
    /// Into `deinterleaved` (BF16 bytes): token `t`'s [Q|K|V|Z] row.
    pub qkv_in: usize,
    /// Into `ssm_conv_out_f32` (bytes): token `t`'s FP32 conv row. Rows are
    /// `qkvz_size` FP32 elements apart — NOT `conv_dim` — so each row keeps
    /// its Z-region tail free for the non-fused arm's FP32 GDN output,
    /// mirroring `ssm_forward`'s single-row layout.
    pub conv_out_f32: usize,
    /// Into `ssm_conv_out_f32` (bytes): token `t`'s FP32 GDN output scratch
    /// (the row's Z-region tail, exactly `value_dim` FP32 elements).
    pub gdn_out_f32: usize,
    /// Into `deinterleaved` (BF16 bytes): token `t`'s Z gate.
    pub z: usize,
    /// Into `gates_buf` (bytes): token `t`'s gate / beta (FP32).
    pub gate: usize,
    pub beta: usize,
    /// Into the `normed_out` BASE (BF16 bytes): token `t`'s final normed
    /// output row — the row phase 9 (out_proj) reads. NOTE the base differs
    /// from `conv_out_buf` for Multi runs (`row0 * value_dim` vs
    /// `row0 * conv_dim` — see `ConvGdnArgs::normed_out`).
    pub normed_out: usize,
    /// Whether token `t` writes h/conv rollback intermediates. The index
    /// `num_tokens - 1` has NO reader (see the reader enumeration in
    /// `trait_decode_batched_conv_gdn.rs`), so the last token skips it.
    pub snapshot: bool,
}

/// See [`ExactRow`]. `fp32`/`bf16` are element sizes in bytes.
pub(super) fn exact_row(
    t: usize,
    num_tokens: usize,
    qkvz_size: usize,
    conv_dim: usize,
    value_dim: usize,
    nv: usize,
) -> ExactRow {
    let (bf16, fp32) = (2usize, 4usize);
    let gate = t * nv * 2 * fp32;
    ExactRow {
        qkv_in: t * qkvz_size * bf16,
        conv_out_f32: t * qkvz_size * fp32,
        gdn_out_f32: t * qkvz_size * fp32 + conv_dim * fp32,
        z: t * qkvz_size * bf16 + conv_dim * bf16,
        gate,
        beta: gate + nv * fp32,
        normed_out: t * value_dim * bf16,
        snapshot: t + 1 < num_tokens,
    }
}

impl Qwen3SsmLayer {
    /// The sequential-decode-exact conv+GDN+norm chain over `num_tokens`
    /// verify positions of ONE sequence. Bitwise contract: after this call,
    /// `h_state`, `conv_state`, the rollback intermediates 0..K-2 and the
    /// normed output rows hold exactly the bits `num_tokens` invocations of
    /// `ssm_forward`'s conv/GDN/norm chain would have produced.
    pub(super) fn decode_batched_conv_gdn_exact(
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
            normed_out,
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
        let eps = ctx.config.rms_norm_eps as f32;

        // ── Kernel selection: the SAME decision tree as `ssm_forward` ──
        let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
        let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
        let fused_gdn_norm = use_f32_gdn
            && self.gdn_f32_norm_k.0 != 0
            && crate::layers::qwen3_ssm::gdn_fused_norm_enabled();
        let snap = fused_gdn_norm && self.gdn_f32_norm_snap_k.0 != 0;
        let f32_conv_base = ctx.buffers.ssm_conv_out_f32();

        // FP32 fused verify conv: one launch for all K positions with the
        // conv-state snapshots written inline — requires the pool-contiguous
        // intermediate layout it writes to (always true for ssm_pool slots;
        // CHECKED, and constant for a given slot ⇒ graph-stable).
        let conv_inter_base = ssm_state.conv_state_intermediates[0];
        let fused_conv = use_f32_conv
            && self.gdn_verify_fused_conv_kn_f32_k.0 != 0
            && ssm_state
                .conv_state_intermediates
                .iter()
                .take(num_tokens)
                .enumerate()
                .all(|(t, p)| p.0 == conv_inter_base.0 + (t * conv_bytes) as u64);

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "EXACT MTP verify ENGAGED (#435 route (a); --exact-verify, or a pass \
                 that declares gdn_exact_replay — the mHC MTP verify does): \
                 per-token sequential-decode kernel chain (f32_conv={use_f32_conv}, \
                 fused_gdn_norm={fused_gdn_norm}, snap_twin={snap}, \
                 fused_f32_conv={fused_conv}); ATLAS_NO_VERIFY_ROW_GDN restores the \
                 WY arms for the pass-scoped case"
            );
        });

        if fused_conv {
            ops::gdn_verify_fused_conv_kn_f32(
                ctx.gpu,
                self.gdn_verify_fused_conv_kn_f32_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                f32_conv_base,
                conv_inter_base,
                num_tokens as u32,
                conv_dim as u32,
                d_conv as u32,
                qk_ch,
                kd as u32,
                qkvz_size as u32, // input stride (BF16 elems between positions)
                qkvz_size as u32, // output stride (FP32 elems between positions)
                (conv_bytes / 4) as u32, // snapshot stride (FP32 elems)
                1e-6,
                stream,
            )?;
        }

        for t in 0..num_tokens {
            let row = exact_row(t, num_tokens, qkvz_size, conv_dim, value_dim, nv);
            let qkv_t = deinterleaved.offset(row.qkv_in);

            // ── Conv1d + SiLU + L2 norm (same kernel as ssm_forward) ──
            let (conv_out_t, conv_elem) = if use_f32_conv {
                (f32_conv_base.offset(row.conv_out_f32), fp32)
            } else {
                (conv_out_buf.offset(t * conv_dim * bf16), bf16)
            };
            if !fused_conv {
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    if use_f32_conv {
                        self.conv1d_l2norm_f32_k
                    } else {
                        self.conv1d_l2norm_k
                    },
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
                if row.snapshot {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            let q_t = conv_out_t;
            let k_t = conv_out_t.offset(key_dim * conv_elem);
            let v_t = conv_out_t.offset(key_dim * 2 * conv_elem);
            let gate_t = gates_buf.offset(row.gate);
            let beta_t = gates_buf.offset(row.beta);
            let z_t = deinterleaved.offset(row.z);
            let normed_t = normed_out.offset(row.normed_out);

            // ── GDN + gated RMS norm (same arm as ssm_forward) ──
            if fused_gdn_norm {
                if snap {
                    // Inline h snapshot: same bits, no d2d.
                    let h_inter = if row.snapshot {
                        ssm_state.h_state_intermediates[t]
                    } else {
                        DevicePtr::NULL
                    };
                    ops::gdn_decode_f32_norm_snap(
                        ctx.gpu,
                        self.gdn_f32_norm_snap_k,
                        ssm_state.h_state,
                        q_t,
                        k_t,
                        v_t,
                        gate_t,
                        beta_t,
                        z_t,
                        self.ssm.norm.weight,
                        normed_t,
                        h_inter,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        eps,
                        stream,
                    )?;
                } else {
                    ops::gdn_decode_f32_norm(
                        ctx.gpu,
                        self.gdn_f32_norm_k,
                        ssm_state.h_state,
                        q_t,
                        k_t,
                        v_t,
                        gate_t,
                        beta_t,
                        z_t,
                        self.ssm.norm.weight,
                        normed_t,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        eps,
                        stream,
                    )?;
                    if row.snapshot {
                        ctx.gpu.copy_d2d_async(
                            ssm_state.h_state,
                            ssm_state.h_state_intermediates[t],
                            h_bytes,
                            stream,
                        )?;
                    }
                }
            } else {
                // Unfused arm: FP32 GDN + FP32-input gated norm when linked
                // (ssm_forward's `use_f32_gdn`), BF16 pair otherwise.
                let (gdn_kernel, norm_kernel, gdn_out_t) = if use_f32_gdn {
                    (
                        self.gdn_f32_k,
                        self.gated_rms_norm_f32_k,
                        f32_conv_base.offset(row.gdn_out_f32),
                    )
                } else {
                    (
                        self.gdn_k,
                        self.gated_rms_norm_k,
                        gdn_out_buf.offset(t * value_dim * bf16),
                    )
                };
                ops::gdn_decode(
                    ctx.gpu,
                    gdn_kernel,
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
                ops::gated_rms_norm(
                    ctx.gpu,
                    norm_kernel,
                    gdn_out_t,
                    z_t,
                    &self.ssm.norm,
                    normed_t,
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
                if row.snapshot {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.h_state,
                        ssm_state.h_state_intermediates[t],
                        h_bytes,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::exact_row;

    // Real 27B GDN shapes: nk=16, nv=48, kd=vd=128, d_conv=4.
    const QKVZ: usize = 16384; // conv_dim + value_dim
    const CONV_DIM: usize = 10240; // 2*2048 + 6144
    const VALUE_DIM: usize = 6144; // 48*128
    const NV: usize = 48;

    /// POSITIVE: token offsets stride by the ROW sizes of each buffer — qkvz
    /// rows in `deinterleaved`/`ssm_conv_out_f32`, value_dim rows in the
    /// normed output, 2*nv FP32 in gates — on the real 27B shapes.
    #[test]
    fn exact_row_strides_match_27b_shapes() {
        let r2 = exact_row(2, 4, QKVZ, CONV_DIM, VALUE_DIM, NV);
        assert_eq!(r2.qkv_in, 2 * QKVZ * 2, "deinterleaved rows are qkvz BF16");
        assert_eq!(r2.conv_out_f32, 2 * QKVZ * 4, "f32 conv rows are qkvz FP32");
        assert_eq!(
            r2.gdn_out_f32,
            2 * QKVZ * 4 + CONV_DIM * 4,
            "f32 gdn scratch is the conv row's Z-region tail"
        );
        assert_eq!(r2.z, 2 * QKVZ * 2 + CONV_DIM * 2, "Z sits after [Q|K|V]");
        assert_eq!(r2.gate, 2 * NV * 2 * 4, "gates rows are [gate|beta] FP32");
        assert_eq!(r2.beta, r2.gate + NV * 4);
        assert_eq!(
            r2.normed_out,
            2 * VALUE_DIM * 2,
            "normed rows are value_dim BF16"
        );
        // The FP32 GDN scratch must fit inside the row's tail: exactly
        // value_dim FP32 elements between conv_dim and qkvz_size.
        assert_eq!(QKVZ - CONV_DIM, VALUE_DIM);
    }

    /// POSITIVE + NEGATIVE: rollback intermediates are written for tokens
    /// 0..K-2 and NOT for K-1 (no reader exists for that index — writing it
    /// is the dead d2d the K=4 arm removed; skipping earlier ones would break
    /// rollback).
    #[test]
    fn exact_row_snapshot_skips_only_last_token() {
        for k in 2..=8usize {
            for t in 0..k {
                let row = exact_row(t, k, QKVZ, CONV_DIM, VALUE_DIM, NV);
                assert_eq!(
                    row.snapshot,
                    t + 1 < k,
                    "snapshot policy wrong at t={t}, k={k}"
                );
            }
        }
    }
}
