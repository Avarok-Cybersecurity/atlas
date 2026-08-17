// SPDX-License-Identifier: AGPL-3.0-only

//! Batched K=4 verify support (verify_e.rs): WY pointer-table staging +
//! CUDA-graph gating helpers. Split out to keep verify_e under the LoC cap.

#![allow(dead_code)]

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layer::{SsmLayerState, VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS};
use crate::traits::SequenceState;

/// Bound on cached batched-verify graphs (one per distinct (ssm-slot
/// vector, k) key). Slot vectors churn as sequences finish; at the cap the
/// least-recently-used graph is destroyed and replaced (verify_e.rs), so
/// the cap bounds graph memory WITHOUT ever pinning the path eager —
/// the pre-LRU insert-only cache went permanently eager after 32 distinct
/// vectors, which a long serve is guaranteed to produce.
///
/// The cap is only sane because the key space is bounded: since the
/// canonical depth→slot assignment (`speculative::verify_key`) a step's key
/// is (slot set, depth MULTISET), not the arrangement. Keyed on the
/// arrangement, D-Cut alone produced 266 keys at n=8 against this 32 —
/// 89% of steps re-captured, 23.2 ms/step of instantiate+destroy.
pub(super) const VERIFY_BATCHED_GRAPH_CAP: usize = 32;

/// Verify row-buffer capacity R = Σ ks — the exact capacity of the batched
/// verify's metadata gaps (verify_e.rs layout: positions 384 B | seq_slot
/// 384 B | slots 768 B | seq_lens 384 B | bt at +2048), the `bt_rows`
/// staging and the logits rows (`sizes.rs`). 96 = the wave-11 depth-at-width
/// envelope: 32:2 = n=32 × k=3 rows hits it dead on (24:2 = 72); previously
/// 64 (the 32:1 rung's n=32 × k=2), 32 before that (n=16 × k=2). Sequence
/// count stays bounded at `VERIFY_WY_TABLE_SEQS` = 32 — this cap widens
/// ROWS (depth at width), not width. The scheduler-side `VERIFY_ROW_BUDGET`
/// (`mtp_dcut.rs`) mirrors this bound — keep them in lock-step.
pub(in crate::model) const VERIFY_ROW_CAP: usize = 96;

/// Batched-verify CUDA graphs: ON by default, disabled by PRESENCE of
/// `ATLAS_NO_MTP_VERIFY_GRAPHS` (house convention — `=0` is NOT off).
/// Read once per process.
pub(super) fn verify_graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_VERIFY_GRAPHS").is_none())
}

/// House VALUE convention: a switch is armed by the literal `"1"` and by
/// nothing else — `=0`, `=true`, `=` and mere presence all leave it OFF.
/// SSOT for the three verify-path switches below, which the batched verify
/// used to spell out inline as `std::env::var(..).ok().as_deref() ==
/// Some("1")` / `== Ok("1")` at three separate sites. One expression, one
/// test, no chance of a PRESENCE/VALUE mix-up when a site is edited.
fn value_switch_armed(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Read a VALUE switch from the environment under [`value_switch_armed`].
fn read_value_switch(name: &str) -> bool {
    value_switch_armed(std::env::var(name).ok().as_deref())
}

/// Per-layer stream-sync diagnostic (`ATLAS_K4_DIAG=1`, VALUE check — this
/// one predates the presence convention and `=1` is its documented form).
///
/// Read ONCE per process. The raw `std::env::var` sat in the batched verify
/// hot path, so every n>=2 verify step paid a `getenv` + a `String`
/// allocation for a switch that cannot change after launch — a cost the
/// single-sequence path does not carry. Behaviourally identical for any
/// process that does not mutate its own environment mid-run: nothing in the
/// tree does, and `std::env::set_var` is `unsafe` since Rust 2024.
pub(super) fn k4_diag_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| read_value_switch("ATLAS_K4_DIAG"))
}

/// Verify argmax D2H arm: `ATLAS_VERIFY_D2H_DEFAULT_STREAM=1` restores the
/// original default-stream copy. Read once per process (was a per-step
/// `std::env::var` in the D2H tail).
pub(super) fn verify_d2h_default_stream() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| read_value_switch("ATLAS_VERIFY_D2H_DEFAULT_STREAM"))
}

/// Verify argmax D2H arm: `ATLAS_NO_PINNED_VERIFY_D2H=1` forces the pageable
/// on-stream copy. Read once per process (was a per-step `std::env::var`).
pub(super) fn verify_d2h_no_pinned() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| read_value_switch("ATLAS_NO_PINNED_VERIFY_D2H"))
}

impl TransformerModel {
    /// Batched-verify graph key: each sequence's ssm-pool slot in batch
    /// order — every SSM pointer the graph bakes (h/conv state, rollback
    /// intermediates, WY table contents) is a pure function of this vector;
    /// all other captured addresses (hidden/logits/scratch/meta) are fixed
    /// buffers refreshed pre-replay. The per-sequence ROW VECTOR `ks` is
    /// interleaved into the key because a graph bakes both the R = Σ ks launch
    /// dimensions AND, under D-Cut, WHICH sequence got which depth (the GDN
    /// runs are grouped by depth) — the same slot vector at a different ladder
    /// step or a different D-Cut shape must not replay. A
    /// wy-tables-present sentinel is appended so a table-less capture can
    /// never replay a table-full step or vice versa.
    ///
    /// The scheduler dispatches each chunk in the ONE canonical order
    /// (`speculative::verify_key::verify_batch_order` — depths descending
    /// paired with slots ascending), so the key is a pure function of
    /// (slot set, depth multiset, sentinel): at n=8 the 266 D-Cut
    /// ARRANGEMENTS that thrashed this 32-entry cache collapse onto the 3
    /// multisets behind them. Kill switch `ATLAS_NO_CANONICAL_VERIFY_KEY`
    /// (scheduler side) restores the arrangement-keyed behaviour. Key BYTES
    /// live in `verify_key` so the ordering rule and the key it produces
    /// cannot drift apart. `None` → no graph (a sequence without a pool
    /// slot).
    pub(super) fn verify_batched_graph_key(
        &self,
        seqs: &[&mut SequenceState],
        ks: &[usize],
        wy_tables_null: bool,
    ) -> Option<Vec<u32>> {
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(seqs.len());
        for (i, s) in seqs.iter().enumerate() {
            pairs.push((s.ssm_slot_idx()? as u32, *ks.get(i)? as u32));
        }
        Some(crate::speculative::verify_key::verify_graph_key(
            &pairs,
            wy_tables_null,
        ))
    }

    /// Stage the per-GDN-layer WY pointer tables (`[h|Hi0|Hi1|Hi2]` ×
    /// `VERIFY_WY_TABLE_SEQS` u64 entries per layer, batch entries filled,
    /// tail zero) into the fixed `verify_wy_tables` device buffer. Runs
    /// PRE-graph every batched verify step so a replayed graph reads tables
    /// refreshed for the current batch (contents are constant per slot
    /// vector; refreshing keeps replay correct by construction, not by
    /// invariant).
    ///
    /// `k` is this step's verify width (rows per sequence, 2..=4 from the
    /// ladder). Exactly `k` tables are filled — `[h | Hi_0 .. Hi_{k-2}]` —
    /// because `gdn_decode_wy{2,3,4}` read one h table plus k-1 intermediate
    /// tables. Table STRIDES are `k`-independent, so a slice offset never
    /// depends on the ladder step.
    ///
    /// Returns NULL — uploading nothing — unless EVERY GDN layer × sequence
    /// provides h_state + ≥ k-1 h intermediates (the layer-side batched arm
    /// re-checks per layer; defense in depth). NULL keeps the per-sequence
    /// WY loop, which is byte-identical math.
    /// `ghosts` — drain-tail borrow only (`graph_borrow.rs`): `(slot, k)`
    /// pairs for the borrowed graph's baked tail rows, appended after the
    /// batch entries. A captured entry is a pure function of `(layer, slot)`
    /// (`SsmLayerState` pointers ARE the pool accessors), so ghost entries
    /// are synthesized straight from the pool — reproducing byte-for-byte
    /// the pointers the departed sequence's capture staged. Empty on every
    /// non-borrow step.
    pub(super) fn upload_verify_wy_tables(
        &self,
        seqs: &[&mut SequenceState],
        k: usize,
        ghosts: &[(u32, u32)],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = seqs.len();
        if self.verify_wy_tables.is_null()
            || n + ghosts.len() > VERIFY_WY_TABLE_SEQS
            || !(2..=crate::layer::VERIFY_WY_TABLES_PER_LAYER).contains(&k)
        {
            return Ok(DevicePtr::NULL);
        }
        let num_ssm = self.config.num_ssm_layers();
        if num_ssm == 0 {
            return Ok(DevicePtr::NULL);
        }
        let entries_per_layer = VERIFY_WY_LAYER_STRIDE_BYTES / 8;
        let mut host = vec![0u64; num_ssm * entries_per_layer];
        let mut ssm_idx = 0usize;
        for layer_idx in 0..self.layers.len() {
            if self.config.layer_type(layer_idx) != LayerType::LinearAttention {
                continue;
            }
            let base = ssm_idx * entries_per_layer;
            for (i, seq) in seqs.iter().enumerate() {
                let Some(st) = seq.layer_states[layer_idx]
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                else {
                    return Ok(DevicePtr::NULL);
                };
                if st.h_state.is_null() || st.h_state_intermediates.len() < k - 1 {
                    return Ok(DevicePtr::NULL);
                }
                host[base + i] = st.h_state.0;
                for t in 0..k - 1 {
                    host[base + (t + 1) * VERIFY_WY_TABLE_SEQS + i] = st.h_state_intermediates[t].0;
                }
            }
            for (gi, &(slot, gk)) in ghosts.iter().enumerate() {
                let i = n + gi;
                let s = slot as usize;
                host[base + i] = self.ssm_pool.h_state(ssm_idx, s).0;
                for t in 0..(gk as usize).saturating_sub(1) {
                    host[base + (t + 1) * VERIFY_WY_TABLE_SEQS + i] =
                        self.ssm_pool.h_intermediate(ssm_idx, s, t).0;
                }
            }
            ssm_idx += 1;
        }
        // Pageable-source async H2D per house pattern (the driver stages the
        // host bytes before returning, same as the metadata uploads).
        // SAFETY: the length is derived from the source — `host.len() * 8 ==
        // size_of_val(&host[..])` — so the read stops at `len` and never
        // enters the `Vec`'s spare capacity. `host` is `vec![0u64; num_ssm *
        // entries_per_layer]`, fully zero-initialised at construction, so the
        // entries the fill loop leaves untouched (the `n..VERIFY_WY_TABLE_SEQS`
        // tail of each table) are initialised zeros, not garbage. `u64` is POD.
        let bytes =
            unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 8) };
        self.gpu
            .copy_h2d_async(bytes, self.verify_wy_tables, stream)?;
        Ok(self.verify_wy_tables)
    }
}

#[cfg(test)]
#[path = "verify_e2_tests.rs"]
mod tests;
