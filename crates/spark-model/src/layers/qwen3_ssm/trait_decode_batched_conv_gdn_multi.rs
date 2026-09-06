// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched conv+WY body for the batched K-row MTP verify
//! (`GdnStates::Multi`, k = 2..=4 from the K-vs-batch ladder). Collapses the
//! per-sequence loop — n × (k `conv1d_update_l2norm` + (k-1) conv-state
//! `copy_d2d_async` + 1 `gdn_decode_wy{k}`) = n(2k-1) launches per GDN layer
//! — into TWO launches:
//!
//! 1. `gdn_verify_fused_conv_kn_batched` (gridDim.y = n): all n sequences ×
//!    K positions of conv1d+SiLU+L2norm in one launch, every per-token
//!    rollback snapshot written inline. Bit-identical to n separate per-token
//!    sequences (independent conv windows; only base addresses move — see
//!    the kernel header in `kernels/gb10/common/gdn_verify_fused_conv_kn.cu`).
//!    Contract delta vs the per-token loop: the kernel ALSO writes the dead
//!    t = K-1 snapshot (the per-token path skips it). Harmless: the pool
//!    allocates `num_intermediates = K` snapshots per slot, and the
//!    preconditions below verify index K-1 exists and is intra-slot
//!    contiguous before the launch.
//! 2. `gdn_decode_wy{2,3,4}` at `batch_size = n` with `state_is_table = true`:
//!    ONE launch over device pointer tables (one entry per sequence) for
//!    h_state + the k-1 Hi intermediates — the table form that sidesteps the
//!    intermediates-stride corruption the contiguous form has at n > 1
//!    (see the ensure! in `ops::gdn_decode_wy4`). Per-sequence math is
//!    byte-identical (`b` only selects base addresses).
//!
//! K-GENERIC (2026-07-28). This path was gated to k == 4 because only wy4
//! carried the pointer-table form; every K-vs-batch ladder step below 4
//! therefore ran the per-sequence loop — n launches per GDN layer instead of
//! 2 — which is a named suspect for the measured "K=1 verify step costs ~1.9x
//! a plain batch-16 decode step" (break-even at p1~0.72 is 1.72x). wy2/wy3
//! now take the same `state_is_table` flag, so k in {2,3,4} all take the
//! two-launch path. The conv kernel was already K-generic (`num_tokens` is a
//! runtime arg).
//!
//! The conv launch needs UNIFORM per-sequence strides, which holds iff the
//! batch occupies CONSECUTIVE ssm-pool slots in batch order. That is checked
//! against the ACTUAL state pointers every call; any mismatch falls back to
//! the per-sequence loop (logged once, counted — same pattern as
//! `ssm_batched_recurrent`). The WY tables are staged by the model
//! (`upload_verify_wy_tables`, verify_e2.rs) into a fixed device buffer
//! refreshed pre-replay, so both launches are CUDA-graph-stable.
//!
//! Kill switch `ATLAS_NO_VERIFY_GDN_BATCH` (PRESENCE check per the house
//! convention — `=0` is NOT off) forces the per-sequence loop for A/B.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::{LayerState, VERIFY_WY_TABLE_STRIDE_BYTES};
use crate::layers::ops;

/// How often the batched conv+WY fast path engaged vs fell back (slot
/// fragmentation breaks the consecutive-slot precondition as sequences
/// finish, so fallbacks are expected to recur).
static BATCHED_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bit `k` set once the ENGAGED / DECLINED line has been logged for verify
/// width `k`. Per-`k` (not a single `Once`) so a serve log PROVES which
/// ladder widths actually took the two-launch path — a single first-hit line
/// at k=4 would say nothing about the k=2/k=3 steps this change exists for.
static ENGAGED_KMASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static DECLINED_KMASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Set bit `k` in `mask`; returns true iff this call set it (first time).
fn first_for_k(mask: &std::sync::atomic::AtomicU32, k: usize) -> bool {
    let bit = 1u32 << (k & 31);
    (mask.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit) == 0
}

/// Periodic engaged-vs-declined RATE at INFO, under the existing
/// `ATLAS_MTP_ACCEPT_DEBUG` gate (checked FIRST — a default serve pays one
/// `OnceLock` load and nothing else).
///
/// The per-`k` first-occurrence lines above prove WHICH widths ever took
/// each arm; they say nothing about how often. That distinction is the whole
/// question when the batched verify carries a per-step cost the n==1 path
/// does not: a declined call runs `n*(2k-1)` launches per GDN layer instead
/// of 2 (768 vs 96 at n=2, k=4 over 48 GDN layers), so a decline RATE is a
/// millisecond-scale term while a single decline is noise.
fn record_multi_rate(n: usize, kk: usize) {
    use std::sync::atomic::Ordering;
    const PERIOD: u64 = 2048;
    static SINCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !crate::speculative::mtp_accept_debug() {
        return;
    }
    if SINCE.fetch_add(1, Ordering::Relaxed) + 1 >= PERIOD {
        SINCE.store(0, Ordering::Relaxed);
        let ok = BATCHED_OK.load(Ordering::Relaxed);
        let fb = FALLBACK.load(Ordering::Relaxed);
        tracing::info!(
            "batched-verify GDN conv+WY [last n={n} k={kk}]: engaged={ok} declined={fb} \
             declined_frac={:.3} (a declined call is {} launches/layer, not 2)",
            fb as f64 / (ok + fb).max(1) as f64,
            n * (kk + kk - 1),
        );
    }
}

/// Kill switch, PRESENCE check (`=0` is NOT off), read once per process.
fn verify_gdn_batch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_VERIFY_GDN_BATCH").is_none())
}

impl Qwen3SsmLayer {
    /// Attempt the cross-sequence batched conv+WY fast path for the
    /// `GdnStates::Multi` arm. `args` carries the BASE (un-offset) buffers
    /// with `args.num_tokens = k` (per-sequence verify width); `states` is
    /// the n per-sequence SSM states in batch order; `wy_tables` is this
    /// layer's slice of the model's staged pointer tables (NULL → decline).
    ///
    /// Returns `Ok(false)` when any precondition fails — the caller runs the
    /// existing per-sequence loop, which is byte-identical math. All checks
    /// are host pointer arithmetic over n <= 4 sequences (no GPU work), and
    /// their outcome is a pure function of the ssm-slot vector + process-
    /// static kernel handles/env, so the decision is stable across CUDA
    /// graph capture/replay for a slot-vector-keyed graph.
    pub(super) fn decode_batched_conv_gdn_multi(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
        wy_tables: DevicePtr,
        ctx: &crate::layer::ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<bool> {
        let n = states.len();
        let kk = args.num_tokens;
        // Write-on-accept request: clear this layer's engaged word FIRST,
        // inside the same capture, so it describes this launch only (a
        // previous step that ran the twin and never folded cannot leak its
        // stash into this step's fold). Before every decline below on
        // purpose: a declined arm runs the parent per-sequence loop, and the
        // fold must then see 0.
        if ctx.gdn_write_on_accept {
            self.woa_clear_at_entry(ctx.gpu, args.stream)?;
        }
        // ── Issue #435 route (a), OPT-IN via `--exact-verify`: exact batched
        // verify (strided sequential-decode chain at batch=n, bitwise-equal
        // to spec-off decode). Ok(false) → the caller's per-sequence loop,
        // which under the same predicate runs the per-token exact arm —
        // identical bits. The WY fast path below is the DEFAULT; without the
        // flag, spec-on output is NOT bitwise-equal to spec-off (#435).
        if super::verify_exact_enabled() {
            return self.decode_batched_conv_gdn_multi_exact(states, ctx, args);
        }
        // wy2/wy3/wy4 all carry the `state_is_table` pointer-table form, so
        // every K-vs-batch ladder width takes the two-launch path. The handle
        // is selected here (try_kernel misses are a silent 0 — gate on it,
        // never assume resolution). k=2/k=3 route through `wy2_kernel` /
        // `wy3_kernel` (the ONE resident-vs-base decision point per K; base
        // kernel when the resident twin is unlinked/killed/shape-mismatched/
        // too narrow (n below wy_resident_min_width()), so this stays non-0
        // either way).
        let wy_k = match kk {
            2 => self.wy2_kernel(args.kd, args.vd, n),
            3 => self.wy3_kernel(args.kd, args.vd, n),
            4 => self.wy4_kernel(),
            // 2026-09-01: gamma-width verifies join the two-launch path via
            // the wyN pointer-table twins (state_is_table compiled in).
            // None (out of range / module absent / ATLAS_GDN_WYN=0 / f16
            // pool without twin) falls back per-sequence, byte-identical —
            // the pre-twin behavior for these widths.
            5..=16 => match self.wyn_table_kernel(kk, ctx.levers.gdn_wyn) {
                Some(h) => h,
                None => return Ok(false),
            },
            _ => return Ok(false),
        };
        // ATLAS_SSM_H_FP16: a zero handle below turns into `Ok(false)` and the
        // caller runs the per-sequence FP32 loop — which, over an FP16 pool,
        // is silent corruption rather than an error. Refuse first.
        self.require_wy_f16(kk, wy_k)?;
        if n < 2
            || !verify_gdn_batch_enabled()
            || self.gdn_verify_fused_conv_kn_batched_k.0 == 0
            || wy_k.0 == 0
            || wy_tables.is_null()
        {
            return Ok(false);
        }

        let conv_bytes = self.conv_state_bytes;
        // ── Preconditions: consecutive-slot layout, verified on the actual
        // pointers (never assumed from the scheduler invariant) ──
        let mut conv_base = DevicePtr::NULL;
        let mut inter_base = DevicePtr::NULL;
        let mut inter_seq_stride = 0u64;
        for i in 0..n {
            let Some(st) = states[i].as_any().downcast_ref::<SsmLayerState>() else {
                return Ok(self.gdn_multi_decline(n, kk));
            };
            // The batched conv writes snapshots t = 0..K-1 at stride
            // conv_bytes; every index must exist and be intra-slot contiguous.
            // h side: the wy4 table entries (staged by the model) need
            // intermediates 0..2 — existence re-checked here (defense in
            // depth; the model declines the upload on the same condition).
            if st.conv_state_intermediates.len() < kk || st.h_state_intermediates.len() < kk - 1 {
                return Ok(self.gdn_multi_decline(n, kk));
            }
            let i0 = st.conv_state_intermediates[0];
            for t in 1..kk {
                if st.conv_state_intermediates[t].0 != i0.0 + (t * conv_bytes) as u64 {
                    return Ok(self.gdn_multi_decline(n, kk));
                }
            }
            if i == 0 {
                conv_base = st.conv_state;
                inter_base = i0;
            } else {
                if st.conv_state.0 != conv_base.0 + (i * conv_bytes) as u64 {
                    return Ok(self.gdn_multi_decline(n, kk));
                }
                if i == 1 {
                    inter_seq_stride = i0.0.wrapping_sub(inter_base.0);
                    // The per-sequence snapshot regions must not overlap the
                    // kernel's K writes (pool stride is num_intermediates ×
                    // conv_bytes with num_intermediates >= K, so this holds
                    // for pool-backed states; checked, not assumed).
                    if inter_seq_stride < (kk * conv_bytes) as u64 {
                        return Ok(self.gdn_multi_decline(n, kk));
                    }
                } else if i0.0 != inter_base.0 + (i as u64) * inter_seq_stride {
                    return Ok(self.gdn_multi_decline(n, kk));
                }
            }
        }

        let ConvGdnArgs {
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            qkvz_size,
            conv_dim,
            key_dim,
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

        // ── 1 launch: all n sequences × K conv positions + inline snapshots ──
        ops::gdn_verify_fused_conv_kn_batched(
            ctx.gpu,
            self.gdn_verify_fused_conv_kn_batched_k,
            conv_base,
            deinterleaved,
            &self.ssm.conv1d,
            conv_out_buf,
            inter_base,
            kk as u32,
            conv_dim as u32,
            d_conv as u32,
            qk_ch,
            kd as u32,
            qkvz_size as u32,        // input stride (BF16 elems between positions)
            conv_dim as u32,         // output stride (BF16 elems between positions)
            (conv_bytes / 4) as u32, // snapshot stride (FP32 elems)
            1e-6,
            n as u32,
            (conv_bytes / 4) as u32, // conv_state seq stride (FP32 elems)
            (kk * qkvz_size) as u32, // input seq stride (BF16 elems)
            (kk * conv_dim) as u32,  // output seq stride (BF16 elems)
            (inter_seq_stride / 4) as u32, // snapshot seq stride (FP32 elems)
            stream,
        )?;

        // ── 1 launch: WY over all n sequences via pointer tables ──
        // Activation rows are seq-major (`r = b*k + t`), exactly the kernels'
        // `(b*K+T)*stride` indexing; the state args become device pointer
        // tables (h | Hi0 | ..), one `VERIFY_WY_TABLE_STRIDE_BYTES` slab
        // each, staged by the model pre-graph. Only the first `kk` slabs are
        // read (wy2 takes h+Hi0, wy3 h+Hi0+Hi1, wy4 h+Hi0..Hi2).
        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
        let gate_ptr = gates_buf;
        let beta_ptr = gates_buf.offset(nv * fp32);
        let hi = |t: usize| wy_tables.offset(t * VERIFY_WY_TABLE_STRIDE_BYTES);
        // Write-on-accept (2026-09-03): the K=4 twin writes no state; the
        // model folds the accepted rows after the verdict (`gdn_fold_accepted`).
        // Engaged ONLY on the caller's request (`ctx.gdn_write_on_accept`,
        // the DFlash batched step); the decision is `woa::woa_decision`.
        let woa_now = self.woa_now(ctx.gdn_write_on_accept, kk, n);
        if woa_now {
            self.woa_launch(
                ctx.gpu,
                wy_tables,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                n,
                conv_dim,
                stream,
            )?;
        } else {
            match kk {
                2 => ops::gdn_decode_wy2(
                    ctx.gpu,
                    wy_k,
                    wy_tables,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    hi(1),
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32, // qk_stride
                    conv_dim as u32, // v_stride
                    (nv * 2) as u32, // gb_stride
                    true,            // state_is_table — one table entry per sequence
                    stream,
                )?,
                3 => ops::gdn_decode_wy3(
                    ctx.gpu,
                    wy_k,
                    wy_tables,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    hi(1),
                    hi(2),
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    (nv * 2) as u32,
                    true,
                    stream,
                )?,
                4 => ops::gdn_decode_wy4(
                    ctx.gpu,
                    wy_k,
                    wy_tables,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    hi(1),
                    hi(2),
                    hi(3),
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    (nv * 2) as u32,
                    true,
                    stream,
                )?,
                // K=5..16: the wyN table twin reads the Hi slabs itself — slab 0
                // (h) as the state table, slab 1 (Hi0) as the intermediates
                // table, and VERIFY_WY_TABLE_SEQS pointer entries between
                // consecutive Hi slabs (the kernel's reinterpreted stride).
                _ => ops::gdn_decode_wyn_table(
                    ctx.gpu,
                    wy_k,
                    wy_tables,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_out_buf,
                    hi(1),
                    crate::layer::VERIFY_WY_TABLE_SEQS as u32,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    (nv * 2) as u32,
                    stream,
                )?,
            }
        }

        let ok = BATCHED_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        record_multi_rate(n, kk);
        if first_for_k(&ENGAGED_KMASK, kk) {
            tracing::info!(
                "batched-verify GDN conv+WY ENGAGED (n={n}, k={kk}): per-layer {} launches \
                 -> 2 (batched conv kernel + table-form wy{kk}); count logged at debug",
                n * (kk + kk - 1),
            );
        }
        if ok.is_multiple_of(1024) {
            tracing::debug!("batched-verify GDN conv+WY engaged x{ok}");
        }
        Ok(true)
    }

    /// Count + first-occurrence log for a declined batched conv+WY call.
    /// Always returns `false` so call sites read
    /// `return Ok(self.gdn_multi_decline(n, kk))`.
    fn gdn_multi_decline(&self, n: usize, kk: usize) -> bool {
        let n_fb = FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        record_multi_rate(n, kk);
        if first_for_k(&DECLINED_KMASK, kk) {
            tracing::info!(
                "batched-verify GDN conv+WY DECLINED (n={n}, k={kk}): batch is not on \
                 consecutive ssm-pool slots (or intermediates/tables unavailable) — running \
                 the per-sequence loop. Slots fragment as sequences finish, so this can \
                 recur; count logged at debug."
            );
        }
        tracing::debug!("batched-verify GDN conv+WY fallback #{n_fb} (n={n}, k={kk})");
        false
    }
}
