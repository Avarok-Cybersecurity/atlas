// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 5-7 of `Qwen3SsmLayer::decode_batched_inner`: Conv1d + L2 norm +
//! GDN per-token (with intermediate checkpoints). Extracted from
//! `trait_decode_batched.rs` to keep the parent file under 500 LoC.
//! Dispatches one of the fused K=2/3/4 paths, the pool-layout WY arm
//! (K∈{5..16} chain verify and K=17 DFlash — see
//! `trait_decode_batched_conv_gdn_wyn.rs`), or the sequential per-token
//! fallback. All buffers + state are owned by the caller; this
//! function only mutates `ssm_state.h_state`, `ssm_state.conv_state`,
//! their intermediates, `conv_out_buf`, and `gdn_out_buf`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

// The `OnceLock<bool>` static that lived here is now a field on
// `layers::ops::ModelLevers` — resolved when the model is built and carried
// on `ForwardContext`, because a static outlives the model whose flags it
// encodes.

/// Kill switch for the register-resident wy2 twin. PRESENCE check per the
/// house convention (`ATLAS_NO_GDN_WY2_RESIDENT=0` is NOT off), read once
/// per process.
fn wy2_resident_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_GDN_WY2_RESIDENT").is_none())
}

/// Kill switch for the register-resident wy3 twin. Independent of wy2's so
/// each lever attributes on its own A/B leg. PRESENCE check per the house
/// convention (`=0` is NOT off), read once per process.
fn wy3_resident_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_GDN_WY3_RESIDENT").is_none())
}

/// Minimum verify batch width (sequences in the launch) for the
/// register-resident wy twins (SSOT for wy2 AND wy3 — same
/// `__launch_bounds__(128,1)` occupancy trade). The resident kernel is
/// 1 block/SM by construction; for wy2, n=32 buys +22 tok/s (+9.6%,
/// matched-out 256.2 vs 235.2) and C=16 matched-out measures +8.6 tok/s at
/// n=16 (validator r1, 184.4 vs 175.8 @15882). The apparent -2 tok/s at
/// n=8 that motivated this gate was later attributed to CROSS-BOOT NOISE
/// (fixer r1: gated legs = the kill-leg dispatch by construction, yet read
/// inside the resident-ON band; validator r1 concurred), so n < 16 is
/// UNPROVEN either way, not a measured loss. 16 stays the floor because it
/// is the smallest width validated on the winning side — and for wy3 it is
/// exactly the 16:2 default rung's width (n=16 x k=3 rows) — below it the
/// base kernel is dispatched (strictly free at wide widths, protective at
/// narrow ones).
fn wy_resident_min_width() -> usize {
    16
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
    /// Final normed-output base for THIS call's rows: the phase-8/9 buffer
    /// (== conv_out_buf's UN-offset base) advanced by `row0 * value_dim`
    /// BF16 — NOT by `row0 * conv_dim` like `conv_out_buf`, because phase 9
    /// reads normed rows at `value_dim` stride from row 0. Only the exact
    /// verify arm writes through this (its norm runs in-loop); the WY arms
    /// leave the norm to phase 8, which derives its own destination.
    pub normed_out: DevicePtr,
    /// The h pool's BYTE PITCH — `Qwen3SsmLayer::h_slot_stride_bytes()`, not
    /// the FP32 `h_state_bytes`. Every reader below strides or byte-copies
    /// pool h memory with it (per-token intermediates, slot-to-slot bases),
    /// and under the f16-SIZED pool those regions are 2 bytes/element. An
    /// FP32 value here would overrun the neighbouring slot rather than fail.
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

    /// Select the K=2 verify WY kernel: the register-resident twin
    /// (`gated_delta_rule_wy2_resident`, Pass 2 served from registers —
    /// 2R+2W -> 1R+2W of the 64KB/head FP32 state) when it is linked, the
    /// head shape matches its compile-time k_dim (kd == vd == 128, the only
    /// production GDN shape), the launch is WIDE enough to carry its
    /// 1-block/SM occupancy (`n >= wy_resident_min_width()` — the wave-10
    /// validator measured the resident kernel LOSING ~2 tok/s at n=8 while
    /// buying +22 at n=32), and the kill switch is absent; the base
    /// `gated_delta_rule_wy2` otherwise. Identical launch contract — call
    /// sites (single-seq arm here with n=1, batched-verify arm in
    /// `trait_decode_batched_conv_gdn_multi` with n=batch width) just swap
    /// the handle, keeping this the ONE dispatch decision point. The choice
    /// is a pure function of (shape, n, process-static handle/env), and
    /// verify graphs are keyed by the slot vector (which fixes n), so it is
    /// CUDA-graph-stable. Byte-identical numerics (bitwise parity leg in
    /// gdn_wy_verify_microtest). The first dispatch of each arm logs WITH
    /// the handle — try_kernel misses are a silent handle 0, so the ENGAGED
    /// log is the resolution proof.
    pub(super) fn wy2_kernel(
        &self,
        kd: usize,
        vd: usize,
        n: usize,
    ) -> spark_runtime::gpu::KernelHandle {
        let wide_enough = n >= wy_resident_min_width();
        let eligible = kd == 128
            && vd == 128
            && wide_enough
            && self.gdn_wy2_resident_k.0 != 0
            && wy2_resident_enabled();
        static LOGGED_ENGAGED: std::sync::Once = std::sync::Once::new();
        static LOGGED_BASE: std::sync::Once = std::sync::Once::new();
        // ATLAS_SSM_H_FP16 stage 2: under the flag the h-state in the pool is
        // FP16, so the FP16 twin is the ONLY correct kernel — an FP32 twin
        // here would read half-width data as floats and emit fluent garbage.
        // Selection is otherwise identical (same residency/width/shape rules),
        // which keeps this the one decision point per K. A zero handle is
        // returned as zero on purpose: the call sites turn that into a hard
        // error rather than a silent FP32 fallback.
        if super::ssm_h_fp16_enabled() {
            return if eligible && self.gdn_wy2_resident_f16_k.0 != 0 {
                self.gdn_wy2_resident_f16_k
            } else {
                self.gdn_wy2_f16_k
            };
        }
        if eligible {
            LOGGED_ENGAGED.call_once(|| {
                tracing::info!(
                    "GDN wy2 REGISTER-RESIDENT ENGAGED (handle {:#x}, n={n}): K=2 verify \
                     Pass 2 served from registers — state traffic 2R+2W -> 1R+2W; \
                     width-gated n >= {}; kill switch ATLAS_NO_GDN_WY2_RESIDENT (presence)",
                    self.gdn_wy2_resident_k.0,
                    wy_resident_min_width(),
                );
            });
            self.gdn_wy2_resident_k
        } else {
            LOGGED_BASE.call_once(|| {
                tracing::info!(
                    "GDN wy2 register-resident twin NOT engaged at this dispatch (kd={kd}, \
                     vd={vd}, n={n} vs min width {}, handle {:#x}, kill_switch_present={}): \
                     base gated_delta_rule_wy2 in use (wider K=2 launches re-decide)",
                    wy_resident_min_width(),
                    self.gdn_wy2_resident_k.0,
                    !wy2_resident_enabled(),
                );
            });
            self.gdn_wy2_k
        }
    }

    /// Select the K=3 verify WY kernel: the register-resident twin
    /// (`gated_delta_rule_wy3_resident`, Pass 2 served from registers —
    /// 2R+3W -> 1R+3W of the 64KB/head FP32 state) under exactly the wy2
    /// twin's conditions (kd == vd == 128, `n >= wy_resident_min_width()`,
    /// handle linked, kill switch absent); base `gated_delta_rule_wy3`
    /// otherwise. K=3 is the 16:2 default ladder rung's row shape (2 drafts
    /// = 3 rows/seq — the +6% C=16 winner, 2026-07-30) and the 24:2/32:2
    /// env rungs'; the 16:2 win was measured ON THE BASE wy3, so this twin
    /// STACKS on it (the rung no longer forfeits the residency lever that
    /// was wy2-only). Same ONE-decision-point / graph-stability / bitwise-
    /// parity contract as `wy2_kernel` above; the ENGAGED log is the
    /// try_kernel resolution proof.
    pub(super) fn wy3_kernel(
        &self,
        kd: usize,
        vd: usize,
        n: usize,
    ) -> spark_runtime::gpu::KernelHandle {
        let wide_enough = n >= wy_resident_min_width();
        let eligible = kd == 128
            && vd == 128
            && wide_enough
            && self.gdn_wy3_resident_k.0 != 0
            && wy3_resident_enabled();
        static LOGGED_ENGAGED: std::sync::Once = std::sync::Once::new();
        static LOGGED_BASE: std::sync::Once = std::sync::Once::new();
        // ATLAS_SSM_H_FP16 stage 2 — see `wy2_kernel` for the rationale.
        if super::ssm_h_fp16_enabled() {
            return if eligible && self.gdn_wy3_resident_f16_k.0 != 0 {
                self.gdn_wy3_resident_f16_k
            } else {
                self.gdn_wy3_f16_k
            };
        }
        if eligible {
            LOGGED_ENGAGED.call_once(|| {
                tracing::info!(
                    "GDN wy3 REGISTER-RESIDENT ENGAGED (handle {:#x}, n={n}): K=3 verify \
                     Pass 2 served from registers — state traffic 2R+3W -> 1R+3W; \
                     width-gated n >= {}; kill switch ATLAS_NO_GDN_WY3_RESIDENT (presence)",
                    self.gdn_wy3_resident_k.0,
                    wy_resident_min_width(),
                );
            });
            self.gdn_wy3_resident_k
        } else {
            LOGGED_BASE.call_once(|| {
                tracing::info!(
                    "GDN wy3 register-resident twin NOT engaged at this dispatch (kd={kd}, \
                     vd={vd}, n={n} vs min width {}, handle {:#x}, kill_switch_present={}): \
                     base gated_delta_rule_wy3 in use (wider K=3 launches re-decide)",
                    wy_resident_min_width(),
                    self.gdn_wy3_resident_k.0,
                    !wy3_resident_enabled(),
                );
            });
            self.gdn_wy3_k
        }
    }

    /// Select the K=4 verify WY kernel. There is no register-resident K=4
    /// twin, so this is only ever the base kernel or — under
    /// `ATLAS_SSM_H_FP16` — its FP16 h-state twin. K=4 is the widths-1..8
    /// shape of the default ladder (`4:3,8:3,16:2,32:1`, 3 drafts = 4 rows),
    /// i.e. exactly the low rungs the no-regression gate covers.
    pub(super) fn wy4_kernel(&self) -> spark_runtime::gpu::KernelHandle {
        if super::ssm_h_fp16_enabled() {
            return self.gdn_wy4_f16_k;
        }
        self.gdn_wy4_k
    }

    /// Refuse to run the verify path with an FP16 pool and no FP16 kernel.
    ///
    /// The selectors above return handle 0 when the twin for this K did not
    /// link, and every caller's existing reaction to a zero handle is to fall
    /// back — to the base kernel, or to the sequential per-token loop. Both
    /// fallbacks are FP32 readers, and an FP32 reader over an FP16 pool does
    /// not fault: it reinterprets pairs of halves as floats and produces
    /// plausible-looking, wrong numbers. That is the single silent failure
    /// mode of this design, so it is converted into a boot-time-visible error
    /// at the first verify dispatch instead. Preflight makes it unreachable in
    /// a supported configuration; this is the backstop for the unsupported
    /// ones.
    pub(super) fn require_wy_f16(
        &self,
        kk: usize,
        wy_k: spark_runtime::gpu::KernelHandle,
    ) -> Result<()> {
        if super::ssm_h_fp16_enabled() && wy_k.0 == 0 {
            anyhow::bail!(
                "ATLAS_SSM_H_FP16: no FP16 h-state twin resolved for the K={kk} MTP verify \
                 WY kernel. Falling back to the FP32 kernel would read the FP16 pool as \
                 floats and emit fluent garbage, so this refuses instead. Run without \
                 --speculative, or unset ATLAS_SSM_H_FP16."
            );
        }
        Ok(())
    }

    /// Run conv1d_update_l2norm + GDN over `num_tokens` (multi-token decode
    /// / MTP verify). Picks the K=2/3/4, K∈{5..8} (wyN) or K=17 fused WY
    /// path if available (wyN covers K∈{5..16} since 2026-08-29), otherwise
    /// falls back to the sequential per-token gdn_decode loop.
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
            normed_out: _,
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

        // ── Issue #435 route (a), OPT-IN via `--exact-verify`: the
        // sequential-decode-exact chain (bitwise-equal to spec-off decode).
        // All K widths. The WY / fused BF16-conv arms below are the DEFAULT —
        // fast, but NOT bitwise-equal to spec-off (#435's divergence remains
        // unless the flag is given; measured decode-step cost of exact is
        // ~+22-36% at the n=8/16/32 rungs). They are also mandatory under
        // `--ssm-h-dtype f16`, whose FP16 pool the exact arm's FP32 kernels
        // must never read (verify_exact_enabled() is false when h_f16 is
        // set). Phase 8 in decode_batched_inner reads the SAME predicate to
        // skip its norm — the exact arm writes the final normed rows itself.
        //
        // OWED (default-divergence mitigation, investigated 2026-08-09, not
        // implemented — needs GPU-validated kernel twins): the DOMINANT #435
        // term (~8.6e-4 of the total; the chunkwise reordering term is only
        // ~3.4e-8) is the BF16 conv-output STORE, not the WY algebra. The
        // arms below consume `causal_conv1d_update_l2norm` /
        // `gdn_verify_fused_conv_kn` whose only difference from the `_f32`
        // twins sequential decode prefers (ssm_forward.rs:222) is
        // `__float2bfloat16(silu)` at the store; that rounding of k/v is then
        // committed into the FP32 H state by the WY update. Feeding these
        // arms FP32 conv rows (the `_f32` conv kernels already exist) and
        // adding FP32-input twins of the WY family (wy2/wy3/wy4 + resident +
        // strided) would cut the default divergence ~4 orders of magnitude
        // for ~1% extra traffic — the WY kernels are bandwidth-bound on H
        // state (~16 MB/layer/step vs ~120 KB of extra conv-row bytes). It
        // would NOT make spec-on bitwise-equal to spec-off; only
        // `--exact-verify` does that.
        //
        // PASS-SCOPED (2026-09-03): the same exact chain also runs, WITHOUT
        // the global flag, whenever the PASS declares
        // `ForwardContext::gdn_exact_replay` — today only the mHC MTP verify
        // (`model/trait_impl/verify_hc.rs`), whose row 0 re-processes an
        // already-committed token and therefore MUST reproduce that token's
        // serial `decode()` bit for bit. Measured on qwen3.8-flash-next
        // (native EXL3, gamma=1): with the WY/BF16-conv arms the verify's
        // row-0 logits matched serial decode 0/N; the exact chain is one of
        // the three legs that takes it to N/N. Kill switch
        // `ATLAS_NO_VERIFY_ROW_EXACT`. Phase 8 in `decode_batched_inner`
        // reads the SAME predicate to skip its norm.
        if super::verify_row_exact_leg(ctx.gdn_exact_replay, super::RowExactLeg::ConvGdn) {
            return self.decode_batched_conv_gdn_exact(ssm_state, ctx, args);
        }

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
                // Skip t == K-1: no reader exists, and that is ENFORCED, not
                // merely argued. `commit_accepted_prefix` now bails on both
                // `num_accepted == 0` and `num_accepted > k` and early-returns
                // on `num_accepted == k`, so its reachable intermediate index
                // is exactly [0, k-2] (async_chkpt.rs). The other two readers
                // are bounded by their callers: `rollback_ssm_states` is only
                // called from the self-spec path under
                // `if a.seq.seq_len > expected_seq_len` (spec_step.rs:158),
                // which means at least one draft was REJECTED, so
                // `num_accepted + 1 <= K-1` and the index is <= K-2; and
                // `start_rollback_and_checkpoint_async` is only ever called
                // with 1..=K-1 (impl_a2.rs:450-509, spec_step.rs:340).
                // DFlash cannot reach these branches at all: it dispatches
                // only at `drafts.len() >= 4` (mtp_step.rs:308), i.e. verify
                // width >= 5, which lands on K=17 or the sequential fallback
                // (which skips the dead t = K-1 write the same way since the
                // K-1 h-intermediate shrink).
                // Writing it cost a conv_bytes D2D per SSM layer per verify
                // step for nothing (measured: 0.14% of decode GPU time).
                if t + 1 < 4 {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            // WY-chunkwise GDN: 2-pass algorithm for 4-token verification.
            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy4(
                ctx.gpu,
                self.wy4_kernel(),
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
                false,           // contiguous state — this site is batch_size=1
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
                // Skip t == K-1 (dead write — see the K=4 branch above).
                if t + 1 < 3 {
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
            ops::gdn_decode_wy3(
                ctx.gpu,
                self.wy3_kernel(kd, vd, 1),
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
                false,           // contiguous state — this site is batch_size=1
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
                // intermediates[1] (= K-1) is NOT written: the committed
                // post-t1 window is already live in conv_state and the
                // full-accept path early-returns without reading it. See the
                // K=4 branch for the reader enumeration.
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
                // intermediates[1] (= K-1) is NOT written — dead write, see
                // the K=4 branch for the reader enumeration.
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy2(
                ctx.gpu,
                self.wy2_kernel(kd, vd, 1),
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
                false,           // contiguous state — this site is batch_size=1
                stream,
            )?;
        } else if num_tokens == 17
            && self.gdn_wy17_k.0 != 0
            && ctx.levers.gdn_wy17
            && !super::ssm_h_fp16_enabled()
        {
            // (f16 guard 2026-08-29: wy17 has no FP16 twin — under the f16
            // pool this arm must not fire; K=17 then reaches the sequential
            // fallback below, which REFUSES under f16 rather than reading
            // the FP16 pool with FP32 kernels.)
            // ── K=17 (DFlash γ+1): fused WY-Chunkwise path ──
            //
            // Shared pool-layout arm (fused conv_kn epilogue + one wy17
            // launch) — body lives in trait_decode_batched_conv_gdn_wyn.rs,
            // dispatched identically for the chain-verify K∈{5..8} widths
            // below.
            self.decode_batched_conv_gdn_wyn(ssm_state, ctx, args, self.gdn_wy17_k)?;
        } else if let Some(wyn_k) = self.wyn_kernel(num_tokens, ctx.levers.gdn_wyn).filter(|_| {
            // The wyN launch writes Hi_t at h_state_intermediates[0] +
            // t*h_bytes — require the pool-contiguous layout it assumes
            // (always true for ssm_pool slots); fail safe to the sequential
            // fallback otherwise instead of corrupting memory.
            let h_base = ssm_state.h_state_intermediates[0];
            ssm_state
                .h_state_intermediates
                .iter()
                .take(num_tokens - 1)
                .enumerate()
                .all(|(t, p)| p.0 == h_base.0 + (t * h_bytes) as u64)
        }) {
            // ── K∈{5..8} chain verify: fused WY-Chunkwise path (wy5..wy8,
            // one K-templated kernel source). Removes the serial per-token
            // GDN fallback at these widths. Kill-switch: ATLAS_GDN_WYN=0. ──
            self.decode_batched_conv_gdn_wyn(ssm_state, ctx, args, wyn_k)?;
        } else {
            // ── No fused arm (K>17, wyN absent/killed, or non-pool
            // intermediates): sequential per-token path ──
            //
            // f16 backstop (2026-08-29): every kernel below is an FP32
            // h-state reader. Over an FP16 pool that does not fault — it
            // emits fluent garbage — so refuse loudly instead, mirroring
            // `require_wy_f16`. Reachable only when a width's f16 twin is
            // missing/killed or intermediates are non-pool; supported
            // configs never land here (wy2/3/4 + wyn f16 twins cover
            // K<=16, and validate.rs bounds --dflash-gamma under f16).
            if super::ssm_h_fp16_enabled() {
                anyhow::bail!(
                    "ATLAS_SSM_H_FP16: no FP16 fused arm for K={num_tokens} \
                     GDN verify (twin missing/killed or non-pool \
                     intermediates). The sequential fallback's FP32 kernels \
                     would read the FP16 pool as floats and emit fluent \
                     garbage, so this refuses instead. Run without \
                     --speculative at this width, or unset ATLAS_SSM_H_FP16."
                );
            }
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

                // Skip t == K-1 (dead write — no reader exists; see the
                // reader enumeration in the K=4 branch above). Required for
                // the h side since the pool allocates K-1 h intermediates
                // (index K-1 no longer exists); the conv skip saves the
                // same dead copy the fused K=2/3/4 arms already skip.
                if (t as usize) + 1 < num_tokens {
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
        }

        Ok(())
    }
}
