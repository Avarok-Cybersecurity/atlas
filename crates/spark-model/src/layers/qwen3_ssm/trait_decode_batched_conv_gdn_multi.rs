// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched conv+WY body for the batched K=4 MTP verify
//! (`GdnStates::Multi`). Collapses the per-sequence loop — n × (4
//! `conv1d_update_l2norm` + 3 conv-state `copy_d2d_async` + 1
//! `gdn_decode_wy4`) = 8n launches per GDN layer — into TWO launches:
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
//! 2. `gdn_decode_wy4` at `batch_size = n` with `state_is_table = true`:
//!    ONE launch over device pointer tables (one entry per sequence) for
//!    h_state + the 3 Hi intermediates — the table form that sidesteps the
//!    intermediates-stride corruption the contiguous form has at n > 1
//!    (see the ensure! in `ops::gdn_decode_wy4`). Per-sequence math is
//!    byte-identical (`b` only selects base addresses).
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
static ENGAGED_ONCE: std::sync::Once = std::sync::Once::new();
static FALLBACK_ONCE: std::sync::Once = std::sync::Once::new();

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
        // wy4 is the only WY kernel with the pointer-table form (wy2/wy3
        // hard-refuse batch > 1); K != 4 declines.
        if kk != 4
            || n < 2
            || !verify_gdn_batch_enabled()
            || self.gdn_verify_fused_conv_kn_batched_k.0 == 0
            || self.gdn_wy4_k.0 == 0
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
                return Ok(self.gdn_multi_decline(n));
            };
            // The batched conv writes snapshots t = 0..K-1 at stride
            // conv_bytes; every index must exist and be intra-slot contiguous.
            // h side: the wy4 table entries (staged by the model) need
            // intermediates 0..2 — existence re-checked here (defense in
            // depth; the model declines the upload on the same condition).
            if st.conv_state_intermediates.len() < kk || st.h_state_intermediates.len() < kk - 1 {
                return Ok(self.gdn_multi_decline(n));
            }
            let i0 = st.conv_state_intermediates[0];
            for t in 1..kk {
                if st.conv_state_intermediates[t].0 != i0.0 + (t * conv_bytes) as u64 {
                    return Ok(self.gdn_multi_decline(n));
                }
            }
            if i == 0 {
                conv_base = st.conv_state;
                inter_base = i0;
            } else {
                if st.conv_state.0 != conv_base.0 + (i * conv_bytes) as u64 {
                    return Ok(self.gdn_multi_decline(n));
                }
                if i == 1 {
                    inter_seq_stride = i0.0.wrapping_sub(inter_base.0);
                    // The per-sequence snapshot regions must not overlap the
                    // kernel's K writes (pool stride is num_intermediates ×
                    // conv_bytes with num_intermediates >= K, so this holds
                    // for pool-backed states; checked, not assumed).
                    if inter_seq_stride < (kk * conv_bytes) as u64 {
                        return Ok(self.gdn_multi_decline(n));
                    }
                } else if i0.0 != inter_base.0 + (i as u64) * inter_seq_stride {
                    return Ok(self.gdn_multi_decline(n));
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

        // ── 1 launch: WY4 over all n sequences via pointer tables ──
        // Activation rows are seq-major (`r = b*4 + t`), exactly the kernel's
        // `(b*4+T)*stride` indexing; the 4 state args become device pointer
        // tables (h | Hi0 | Hi1 | Hi2), one `VERIFY_WY_TABLE_STRIDE_BYTES`
        // slab each, staged by the model pre-graph.
        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
        let gate_ptr = gates_buf;
        let beta_ptr = gates_buf.offset(nv * fp32);
        ops::gdn_decode_wy4(
            ctx.gpu,
            self.gdn_wy4_k,
            wy_tables,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_out_buf,
            wy_tables.offset(VERIFY_WY_TABLE_STRIDE_BYTES),
            wy_tables.offset(2 * VERIFY_WY_TABLE_STRIDE_BYTES),
            wy_tables.offset(3 * VERIFY_WY_TABLE_STRIDE_BYTES),
            n as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32, // qk_stride
            conv_dim as u32, // v_stride
            (nv * 2) as u32, // gb_stride
            true,            // state_is_table — pointer tables, one entry per sequence
            stream,
        )?;

        let ok = BATCHED_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        ENGAGED_ONCE.call_once(|| {
            tracing::info!(
                "batched-verify GDN conv+WY ENGAGED (n={n}): per-layer 8n launches -> 2 \
                 (batched conv kernel + table-form wy4); count logged at debug"
            );
        });
        if ok.is_multiple_of(1024) {
            tracing::debug!("batched-verify GDN conv+WY engaged x{ok}");
        }
        Ok(true)
    }

    /// Count + first-occurrence log for a declined batched conv+WY call.
    /// Always returns `false` so call sites read `return Ok(self.gdn_multi_decline(n))`.
    fn gdn_multi_decline(&self, n: usize) -> bool {
        let n_fb = FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        FALLBACK_ONCE.call_once(|| {
            tracing::info!(
                "batched-verify GDN conv+WY DECLINED (n={n}): batch is not on consecutive \
                 ssm-pool slots (or intermediates/tables unavailable) — running the \
                 per-sequence loop. Slots fragment as sequences finish, so this can recur; \
                 count logged at debug."
            );
        });
        tracing::debug!("batched-verify GDN conv+WY fallback #{n_fb} (n={n})");
        false
    }
}
