// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone B: the stateful conv1d + SSM-scan inner of the multi-sequence
//! decode body, batched into ONE launch pair over a dense prefix of rows.
//!
//! Milestone A left this inner as a per-row loop (`2 * n` tiny launches per
//! layer per step) because `causal_conv1d_update` and `mamba2_ssm_decode`
//! hardcode every batch address to a dense stride, while the batched decode
//! feeds them rows that are `in_proj_size`-strided (dt, xBC inside the
//! projection row) and `d_xbc`-strided (the conv output). The strided kernel
//! twins added in milestone B take those strides explicitly, so all rows go in
//! one launch.
//!
//! ## Why a dense PREFIX and not the whole batch
//!
//! Neither strided kernel takes a stride for the recurrent state: `conv_state`
//! keeps `(b * dim + ch) * d_conv` and `h_state` keeps
//! `(b * num_heads + head) * head_dim * state_size`, matching the GDN family
//! rule that only activations are strided. Both are therefore INFERRED as
//! `base + i * slot_bytes`, which is only true while the pool slots backing
//! rows `0..p` are contiguous and in slice order.
//!
//! Two things break that, and the same check catches both:
//!
//!   * **Pad rows.** `decode_a2` points every row in `n_real..padded_n` at
//!     `ssm_pool.dummy_slot()` (slot index `max_slots`). A kernel that inferred
//!     the address of a pad row would write it into a REAL sequence's slot —
//!     silent cross-sequence state corruption. A shared dummy slot cannot
//!     satisfy `base + i * slot_bytes` for `i < max_slots`, so the prefix
//!     always stops at or before the first pad.
//!   * **Slot fragmentation.** Slots are reclaimed as sequences finish, so a
//!     mid-batch hole is a NORMAL steady state, not an error. Rows from the
//!     hole onward fall back to the per-row loop, which is what milestone A
//!     did for every row.
//!
//! Both kernels are bit-identical to the per-row loop they replace (only base
//! addresses change; every FMA, warp shuffle and clamp is untouched — proven
//! byte-for-byte by `examples/conv1d_biased_strided_microtest.rs` and
//! `examples/mamba2_strided_microtest.rs`), so this arm can never move
//! numerics. The batched/per-row split here is a pure launch-count decision.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

/// How often the strided conv/scan arm engaged vs declined to the per-row
/// loop. The precondition (pool slots contiguous in slice order) fails as
/// slots fragment, and it must not fail silently — same contract as the GDN
/// family's `ssm_batched_recurrent`.
static BATCHED_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FALLBACK_ONCE: std::sync::Once = std::sync::Once::new();

impl NemotronMamba2Layer {
    /// Whether both strided twins resolved on this target. Milestone-A kernel
    /// sets report handle 0 and keep the per-row loop.
    pub(super) fn has_strided_ssm_decode(&self) -> bool {
        self.conv1d_update_strided_k.0 != 0 && self.mamba2_ssm_strided_k.0 != 0
    }

    /// Length of the leading run of rows whose SSM pool slots are contiguous
    /// AND in slice order — the rows a strided launch may legally cover.
    ///
    /// Returns `0` if any row is not an `SsmLayerState` at all (the caller
    /// then takes the per-row loop, which reports that as a typed error).
    pub(super) fn dense_state_prefix(
        &self,
        n: usize,
        states: &mut [&mut (dyn LayerState + 'static)],
    ) -> usize {
        let mut h_base = DevicePtr::NULL;
        let mut conv_base = DevicePtr::NULL;
        for i in 0..n {
            let Some(st) = states[i].as_any_mut().downcast_mut::<SsmLayerState>() else {
                return 0;
            };
            if i == 0 {
                h_base = st.h_state;
                conv_base = st.conv_state;
                continue;
            }
            let want_h = h_base.0 + (i * self.h_state_bytes) as u64;
            let want_conv = conv_base.0 + (i * self.conv_state_bytes) as u64;
            if st.h_state.0 != want_h || st.conv_state.0 != want_conv {
                return i;
            }
        }
        n
    }

    /// Conv1d + SSM scan for rows `0..n`.
    ///
    /// Rows `0..p` (the dense prefix) go in ONE strided launch pair; rows
    /// `p..n` keep the milestone-A per-row form. `proj` is the `[n,
    /// in_proj_size]` projection output, `xbc_base` the `[n, d_xbc]` conv
    /// output, `y_base` the `[n, d_inner]` scan output.
    pub(super) fn decode_ms_conv_scan(
        &self,
        proj: DevicePtr,
        xbc_base: DevicePtr,
        y_base: DevicePtr,
        n: usize,
        states: &mut [&mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        let gs = self.n_groups * self.state_size;

        let p = if n >= 2 && self.has_strided_ssm_decode() {
            let p = self.dense_state_prefix(n, states);
            if p >= 2 {
                BATCHED_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                self.report_conv_scan_decline(n, p);
            }
            p
        } else {
            0
        };

        if p >= 2 {
            // Row 0's slots are the base of the dense run by construction.
            let (h_base, conv_base) = {
                let st = states[0]
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq 0"))?;
                (st.h_state, st.conv_state)
            };
            // xBC lives INSIDE the projection row (after the z gate), so the
            // conv input stride is in_proj_size, not d_xbc. The non-strided
            // kernel would read row b>=1 at `b * d_xbc`, landing in the
            // previous row's dt/z region — correct at n=1, corrupt at n>=2.
            ops::conv1d_update_biased_strided(
                ctx.gpu,
                self.conv1d_update_strided_k,
                conv_base,
                proj.offset(self.d_inner * bf16),
                &self.ssm.conv1d_weight,
                self.ssm.conv1d_bias.weight,
                xbc_base,
                self.d_xbc as u32,
                self.d_conv as u32,
                p as u32,
                self.in_proj_size as u32, // input row stride (BF16 elements)
                self.d_xbc as u32,        // output row stride (BF16 elements)
                stream,
            )?;
            ops::mamba2_ssm_decode_strided(
                ctx.gpu,
                self.mamba2_ssm_strided_k,
                h_base,
                xbc_base,                                        // x
                xbc_base.offset(self.d_inner * bf16),            // B
                xbc_base.offset((self.d_inner + gs) * bf16),     // C
                proj.offset((self.d_inner + self.d_xbc) * bf16), // dt
                self.ssm.a_log.weight,
                self.ssm.d_param.weight,
                self.ssm.dt_bias.weight,
                y_base,
                p as u32,
                self.num_heads as u32,
                self.head_dim as u32,
                self.state_size as u32,
                self.n_groups as u32,
                1e-9,                     // dt_min — no effective clamp, matches `ssm_decode`
                1e9,                      // dt_max
                self.d_xbc as u32,        // x_stride
                self.d_xbc as u32,        // bc_stride
                self.in_proj_size as u32, // dt_stride (dt is in the proj row)
                self.d_inner as u32,      // y_stride
                stream,
            )?;
        }

        for i in p..n {
            let st = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
            let proj_i = proj.offset(i * self.in_proj_size * bf16);
            let xbc_out_i = xbc_base.offset(i * self.d_xbc * bf16);
            self.conv1d_update_biased(
                ctx.gpu,
                st.conv_state,
                proj_i.offset(self.d_inner * bf16), // xBC within this row
                xbc_out_i,
                self.d_xbc as u32,
                self.d_conv as u32,
                1,
                stream,
            )?;
            self.ssm_decode(
                ctx.gpu,
                st.h_state,
                xbc_out_i,                                         // x
                xbc_out_i.offset(self.d_inner * bf16),             // B
                xbc_out_i.offset((self.d_inner + gs) * bf16),      // C
                proj_i.offset((self.d_inner + self.d_xbc) * bf16), // dt
                y_base.offset(i * self.d_inner * bf16),            // y row i
                1,
                stream,
            )?;
        }
        Ok(())
    }

    /// First decline gets a named `info!` with the offending row; every
    /// decline bumps a counter and logs at debug. Silent declines are what
    /// made the GDN version of this arm unprofilable.
    fn report_conv_scan_decline(&self, n: usize, p: usize) {
        let count = FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        FALLBACK_ONCE.call_once(|| {
            tracing::info!(
                "nemotron mamba2 strided conv/scan DECLINED (n={n}, dense prefix={p}): SSM pool \
                 slots are not contiguous in slice order from row {p}. Falling back to the \
                 per-row loop (2 launches/row/layer instead of 2/layer). Pad rows alias the \
                 pool dummy slot and ALWAYS break the prefix, so a batch whose real rows are \
                 fewer than its padded rung declines by design; slots also fragment as \
                 sequences finish, so this is expected to recur."
            );
        });
        tracing::debug!("nemotron mamba2 strided conv/scan fallback #{count} (n={n}, prefix={p})");
    }
}
