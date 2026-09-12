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
/// verify's metadata gaps (verify_e.rs layout, every offset DERIVED from
/// this constant: positions 4R | seq_slot 4R | slots 8R | seq_lens 4R | bt
/// at 24R), the `bt_rows` staging and the logits rows (`sizes.rs`). History:
/// 32 (n=16 × k=2), 64 (32:1), 96 (wave-11 depth-at-width, 32:2 = n=32 × k=3
/// dead on), now 160: the DFlash uniform K=γ+1=8 shape needs n×8 rows, and
/// 96 capped the batched verify at n=12 — a C=16 serve chunked 12+4 (better
/// than the serial-all it did before, still one extra weight sweep per step)
/// and C=20 fits exactly at 160. Cost is the logits arena: rows × vocab × 2 B
/// = ~79.5 MB at vocab 248320, +32 MB over the 96-row arena. Sequence count
/// stays bounded at `VERIFY_WY_TABLE_SEQS` = 32 — this cap widens ROWS
/// (depth at width), not width. The scheduler-side `VERIFY_ROW_BUDGET`
/// (`mtp_dcut.rs`) and `bt_rows`/`logits_tokens` (`sizes.rs`) mirror this
/// bound — keep all four in lock-step.
pub(in crate::model) const VERIFY_ROW_CAP: usize = 160;

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

/// WY-table staging cache: ON by default, disabled by PRESENCE of
/// `ATLAS_NO_VERIFY_WY_CACHE` (house convention — `=0` is NOT off).
/// Read once per process. OFF restores the unconditional per-step re-stage.
pub(super) fn verify_wy_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_VERIFY_WY_CACHE").is_none())
}

/// Encode the COMPLETE input set of `upload_verify_wy_tables`'s staged bytes.
///
/// ★ The proof obligation for the cache is that two calls with equal keys
/// stage byte-identical tables. Every entry the fill loops write is
/// `SsmLayerState::h_state` / `h_state_intermediates[t]` for a batch
/// sequence, or `ssm_pool.h_state` / `h_intermediate` for a ghost — and
/// those four are the SAME pool accessors:
///
/// * `h_state` is only ever assigned `ssm_pool.h_state(ssm_layer_idx, slot)`
///   (`meta.rs` at alloc, `sequence.rs` at compaction) or `DevicePtr(0)` on
///   free, and `ssm_pool.h_state` is `h_state_pools[layer].offset(slot *
///   h_stored_bytes)` — pure address arithmetic over pool bases fixed at
///   model construction.
/// * `h_state_intermediates[t]` is only ever
///   `ssm_pool.h_intermediate(ssm_layer_idx, slot, t)`, and its LENGTH is
///   `ssm_pool.h_inter_count(slot)` — slot-keyed (tiered pools) and
///   `has_mtp`-gated, where `has_mtp` is a MODEL property
///   (`proposer.is_some() || self_speculative`), not a per-sequence one.
///
/// So a table entry is a pure function of `(layer, slot, t)`, the layer set
/// is `config.num_ssm_layers()` (fixed), and the only step-varying inputs are
/// the ones encoded here:
///   1. `k` — how many of the four per-layer tables are filled.
///   2. `slots.len()` — how many batch entries are filled, hence where the
///      zero tail of each table starts.
///   3. `slots` — the per-sequence ssm-pool slot, IN BATCH ORDER (entry `i`
///      of every table is sequence `i`).
///   4. `ghosts` — the drain-tail borrow's `(slot, depth)` pairs, in order,
///      appended after the batch entries.
///
/// Everything else (pool base addresses, `h_stored_bytes`, the tiered
/// `h_inter_offsets`, the GDN layer set) is fixed for the process at model
/// construction.
///
/// The encoding is injective: `[k, n, slots[0..n], (slot, depth) * g]` — `n`
/// disambiguates the slot run from the ghost tail, and the ghost count falls
/// out of the remaining length.
pub(super) fn verify_wy_cache_key(slots: &[u32], k: usize, ghosts: &[(u32, u32)]) -> Vec<u64> {
    let mut key = Vec::with_capacity(2 + slots.len() + 2 * ghosts.len());
    key.push(k as u64);
    key.push(slots.len() as u64);
    key.extend(slots.iter().map(|&s| s as u64));
    for &(slot, depth) in ghosts {
        key.push(slot as u64);
        key.push(depth as u64);
    }
    key
}

/// What the batched verify did with its CUDA graph on one step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum VerifyGraphOutcome {
    /// Exact slot-vector hit — the whole forward replayed.
    Replay,
    /// Drain-tail borrow: a WIDER captured key replayed with ghost rows.
    Borrow,
    /// Miss: the forward ran eagerly UNDER CAPTURE and a graph was
    /// instantiated (the expensive outcome — `verify_key`'s module docs
    /// price instantiate+destroy at 23.2 ms/step at an 89% recapture rate).
    Capture,
    /// No graph at all: `ATLAS_NO_MTP_VERIFY_GRAPHS`, `ATLAS_K4_DIAG`, or a
    /// batch with a slotless sequence.
    Eager,
}

/// Periodic INFO summary of the batched-verify graph outcomes, under the
/// existing `ATLAS_MTP_ACCEPT_DEBUG` gate (checked FIRST, so a default serve
/// pays one `OnceLock` load and no atomics).
///
/// ★ Why this exists. The n>=2 verify path carries a per-step FIXED cost the
/// n==1 path does not, and the two candidates of the right magnitude — graph
/// RE-capture, and the batched GDN conv+WY fast path declining into the
/// per-sequence loop — were both unanswerable from a ladder log. Captures
/// only logged one INFO line each (thousands of lines, no rate) and the GDN
/// engage/decline counters only surfaced at `debug`. A rate, and the live key
/// count next to it, is what distinguishes "the key space collapsed and the
/// path replays" from "this rung re-captures every step".
pub(super) fn record_verify_graph_outcome(n: usize, live_keys: usize, outcome: VerifyGraphOutcome) {
    use std::sync::atomic::{AtomicU64, Ordering};
    const PERIOD: u64 = 200;
    static STEPS: AtomicU64 = AtomicU64::new(0);
    static REPLAY: AtomicU64 = AtomicU64::new(0);
    static BORROW: AtomicU64 = AtomicU64::new(0);
    static CAPTURE: AtomicU64 = AtomicU64::new(0);
    static EAGER: AtomicU64 = AtomicU64::new(0);
    if !crate::speculative::mtp_accept_debug() {
        return;
    }
    match outcome {
        VerifyGraphOutcome::Replay => &REPLAY,
        VerifyGraphOutcome::Borrow => &BORROW,
        VerifyGraphOutcome::Capture => &CAPTURE,
        VerifyGraphOutcome::Eager => &EAGER,
    }
    .fetch_add(1, Ordering::Relaxed);
    if STEPS.fetch_add(1, Ordering::Relaxed) + 1 >= PERIOD {
        let steps = STEPS.swap(0, Ordering::Relaxed).max(1);
        let (replay, borrow) = (
            REPLAY.swap(0, Ordering::Relaxed),
            BORROW.swap(0, Ordering::Relaxed),
        );
        let (capture, eager) = (
            CAPTURE.swap(0, Ordering::Relaxed),
            EAGER.swap(0, Ordering::Relaxed),
        );
        tracing::info!(
            "batched-verify graphs [{steps} steps, last n={n}]: replay={replay} \
             borrow={borrow} CAPTURE={capture} eager={eager} capture_frac={:.3} \
             live_keys={live_keys}/{}",
            capture as f64 / steps as f64,
            VERIFY_BATCHED_GRAPH_CAP,
        );
    }
}

impl TransformerModel {
    /// EXL3 graph-capture veto for every verify site: the native lm_head and
    /// native MoE experts launch COOPERATIVELY (grid.sync / spin barriers),
    /// which is illegal under CUDA graph capture — the arms' own `ensure!`s
    /// refuse `graph_capture`, so an uncovered site fails loudly mid-serve
    /// instead of corrupting. False on every non-EXL3 boot, so the sites'
    /// `use_graphs` terms are byte-identical to before with the gates off.
    pub(super) fn exl3_graph_veto(&self) -> bool {
        self.lm_head_exl3.is_some() || self.layers.iter().any(|l| l.exl3_graph_veto())
    }

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
    /// paired with slots ascending) at batch widths >=
    /// `verify_key::CANONICAL_KEY_MIN_WIDTH`, so the key is a pure function of
    /// (slot set, depth multiset, sentinel) there: at n=8 the 266 D-Cut
    /// ARRANGEMENTS that thrashed this 32-entry cache collapse onto the 3
    /// multisets behind them. Below that width the key stays
    /// arrangement-shaped ON PURPOSE — the space is 2 keys at n=2 and 10 at
    /// n=4, so there is nothing to collapse and the assignment measured net
    /// negative (`CANONICAL_KEY_MIN_WIDTH` carries the A/B table). Kill switch
    /// `ATLAS_NO_CANONICAL_VERIFY_KEY` (scheduler side) restores the
    /// arrangement-keyed behaviour at every width. Key BYTES live in
    /// `verify_key` so the ordering rule and the key it produces cannot drift
    /// apart. `None` → no graph (a sequence without a pool slot).
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
    /// PRE-graph on every batched verify step whose table content differs
    /// from what is already staged, so a replayed graph reads tables valid
    /// for the current batch.
    ///
    /// ★ CACHED since `perf/verify-fixed-cost`. This used to re-stage
    /// unconditionally — a 48 KB zeroed host `Vec` + `num_ssm × n`
    /// `Any` downcasts + a 48 KB pageable H2D on EVERY n>=2 verify step,
    /// which the single-sequence path (n==1, `verify_c2`) never pays. The
    /// staged bytes are a pure function of `(k, slot vector, ghosts)` —
    /// `verify_wy_cache_key` carries the enumeration and the proof — so when
    /// that key matches `verify_wy_cache` the device buffer already holds
    /// exactly these bytes and both the build and the copy are skipped.
    /// Nothing else writes `verify_wy_tables` (allocation memsets it to zero
    /// once; this is its only writer), so "same key ⇒ same device content"
    /// holds for the buffer, not merely for the host image.
    ///
    /// The trade is explicit: correctness moves from "by construction"
    /// (re-upload every step) to "by invariant" (the enumeration above).
    /// The invariant is backstopped twice — the per-layer batched arm
    /// re-checks each state's intermediate capacity before reading a table
    /// (`trait_decode_batched_conv_gdn_multi.rs`), and the wy-tables-present
    /// sentinel is in the CUDA-graph key — and `ATLAS_NO_VERIFY_WY_CACHE`
    /// restores the unconditional re-stage for A/B.
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
        // Cache probe. A sequence without a pool slot is unkeyable (and would
        // fail the `h_state` gate below anyway) — such a batch simply stages
        // uncached, exactly as before.
        let cache_key: Option<Vec<u64>> = if verify_wy_cache_enabled() {
            seqs.iter()
                .map(|s| s.ssm_slot_idx().map(|v| v as u32))
                .collect::<Option<Vec<u32>>>()
                .map(|slots| verify_wy_cache_key(&slots, k, ghosts))
        } else {
            None
        };
        if let Some(key) = cache_key.as_deref()
            && self.verify_wy_cache.lock().as_deref() == Some(key)
        {
            return Ok(self.verify_wy_tables);
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
        // Recorded only AFTER the copy is enqueued: every early return above
        // (downcast miss / null h_state / short intermediates) leaves the
        // device buffer — and therefore the cache — untouched and still
        // describing the last successful stage.
        if let Some(key) = cache_key {
            *self.verify_wy_cache.lock() = Some(key);
        }
        Ok(self.verify_wy_tables)
    }
}

#[cfg(test)]
#[path = "verify_e2_tests.rs"]
mod tests;
