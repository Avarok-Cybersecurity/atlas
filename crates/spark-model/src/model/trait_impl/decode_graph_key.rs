// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence decode CUDA-graph cache keying (`decode_a2.rs` helpers).
//!
//! A captured batched-decode graph bakes exactly ONE class of per-sequence
//! device address: the SSM `h_state` / `conv_state` pointers of each row's
//! layer states (`trait_impl/sequence.rs` states this explicitly; attention
//! layers hand out `EmptyLayerState`, and every other captured address —
//! hidden/residual/logits/scratch, attention metadata, KV block tables — is a
//! fixed buffer refreshed BEFORE replay). Those pointers are a pure function
//! of the row's SSM pool slot, so the slot VECTOR is the exact graph key.
//!
//! It replaced a `padded_n`-only key, which was sound only under the
//! scheduler invariant "active sequences occupy contiguous SSM pool slots
//! [0..n) in batch order" AND `n == padded_n`. Two live callers break it:
//!
//! * the **MTP Phase-A bootstrap** (`mtp_bootstrap_step.rs`) passes a SUBSET
//!   of the active set (the draftless sequences, sorted by slot), e.g. slots
//!   `{0,2,5}` — a graph captured for that subset would then be replayed for
//!   a different subset, or for the full batch, with the first subset's GDN
//!   state pointers baked in: silent cross-sequence state corruption;
//! * any `n < padded_n` batch, whose rows `n..padded_n` bake the *dummy* slot
//!   — replaying that graph at a larger `n` runs a real sequence's row into
//!   the dummy slot (its recurrent state never advances), and the converse
//!   replay writes a real slot that is not in this batch.
//!
//! Both are reachable under continuous load. Keying on the slot vector makes
//! replay correct by construction instead of by invariant, so `free_sequence`
//! retains this cache (LRU at [`batch_decode_graph_cap`] bounds memory). A
//! blanket drain on every completion would throw away the measured +2.8/+3.2%
//! and recapture an identical slot-vector graph for the next occupant.
//!
//! Multi-seq decode graphs are DEFAULT-ON since 2026-07-27
//! (`ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1` disables), validated: C=8
//! 65.75 -> 67.6 (+2.8%), C=16 92.6 -> 95.6 (+3.2%), emitted-text SHA
//! unchanged, 2 reps/cell. That measurement RETIRED a planned rewrite: the
//! attention branch has ~2,300 per-sequence launches/step and hand-batching
//! them was estimated at 4-9 ms; graphs capture all of them wholesale for
//! +3.2%, so the individual batching work cannot beat what the flag already
//! gets. (Batching the gate-mul by hand beforehand measured flat.)

use spark_runtime::gpu::{GpuBackend, GraphHandle};
use std::collections::HashMap;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// Bound on cached batched-decode graphs (one per distinct SSM-slot vector).
/// Slot vectors churn as sequences finish; at the cap the least-recently-used
/// graph is destroyed and replaced, so the cap bounds graph memory without
/// ever pinning the path eager (the `verify_batched_graphs` LRU precedent —
/// an insert-only cache goes permanently eager on a long serve).
/// 32 -> 48 with the bs=32 ladder (padded_n 24/32): the C=32 drain tail
/// mints slot-vector keys for n=17..32 compositions ON TOP of the <=16 set;
/// at 32 the LRU would thrash and re-capture the steady-state key.
///
/// DERIVED for native bs>32 (wave-14a): `16 + decode-meta rows` — exactly
/// the historical 48 at the 32-row floor (every bs<=32 boot unchanged), and
/// the same +16 headroom over the drain-tail composition set at bs=64
/// (cap 80). Pure LRU bound — bounds graph memory, never pins eager.
pub(super) fn batch_decode_graph_cap(decode_meta_rows: usize) -> usize {
    16 + decode_meta_rows
}

/// Insert `graph` at `key`, evicting the LRU entry when at `cap` and the key
/// is new. Returns handles the caller must `destroy_graph` (evicted LRU
/// and/or the previous occupant of `key`).
pub(super) fn lru_insert_graph(
    cache: &mut (HashMap<Vec<u32>, (GraphHandle, u64)>, u64),
    cap: usize,
    key: Vec<u32>,
    graph: GraphHandle,
) -> Vec<GraphHandle> {
    let mut drop = Vec::new();
    if cache.0.len() >= cap
        && !cache.0.contains_key(&key)
        && let Some(evict) = cache
            .0
            .iter()
            .min_by_key(|(_, entry)| entry.1)
            .map(|(k, _)| k.clone())
        && let Some((old, _)) = cache.0.remove(&evict)
    {
        drop.push(old);
    }
    cache.1 += 1;
    let tick = cache.1;
    if let Some((old, _)) = cache.0.insert(key, (graph, tick)) {
        drop.push(old);
    }
    drop
}

/// Graph the batches the padded_n-keyed cache could not legally cover — the
/// MTP bootstrap's slot SUBSET and any `n < padded_n` batch: **ON** by
/// default, disabled by PRESENCE of `ATLAS_NO_MTP_BOOT_GRAPH` (house
/// convention — `=0` is NOT off). Disabled, those batches run EAGER and only
/// the canonical `slots == [0..n)` with `n == padded_n` batch is graphed,
/// which is the pre-slot-key behaviour minus its unsound replays.
/// Read once per process.
pub(super) fn boot_graph_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_BOOT_GRAPH").is_none())
}

impl TransformerModel {
    /// Batched-decode graph key: each row's SSM pool slot in batch order,
    /// `padded_n` entries long (rows `n..padded_n` bake the dummy slot, so
    /// they carry it). The length encodes `padded_n`; the entries encode
    /// every per-sequence pointer a capture bakes.
    ///
    /// Models with no SSM layers bake nothing per-sequence (their layer
    /// states are empty), so they keep a single-entry `[padded_n]` key —
    /// byte-identical cache behaviour to the old `padded_n` map. Length 1 vs
    /// length `padded_n >= 2` keeps the two families from colliding.
    ///
    /// `None` → run eager: an SSM model whose row has no pool slot has
    /// per-SEQUENCE layer-state pointers that no slot vector can describe.
    pub(super) fn batch_decode_graph_key(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
    ) -> Option<Vec<u32>> {
        if self.config.num_ssm_layers() == 0 {
            return Some(vec![padded_n as u32]);
        }
        let n = seqs.len();
        let mut key: Vec<u32> = Vec::with_capacity(padded_n);
        for s in seqs.iter() {
            match s.ssm_slot_idx() {
                Some(idx) => key.push(idx as u32),
                None => {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        tracing::warn!(
                            "multi-seq decode graphs OFF for this step: a sequence has no SSM \
                             pool slot, so its baked layer-state pointers are per-sequence"
                        );
                    });
                    return None;
                }
            }
        }
        let dummy = self.ssm_pool.dummy_slot() as u32;
        for _ in n..padded_n {
            key.push(dummy);
        }
        // Kill switch: keep ONLY the canonical batch graphed.
        if !boot_graph_enabled()
            && (n != padded_n || key.iter().enumerate().any(|(i, &s)| s != i as u32))
        {
            return None;
        }
        Some(key)
    }

    /// Insert a freshly captured graph, evicting the least-recently-used
    /// entry at [`batch_decode_graph_cap`].
    ///
    /// Destroying the evicted graph is safe: the scheduler consumes every
    /// decode step's logits with a blocking D2H before submitting the next
    /// step, so no earlier replay is still in flight on this stream (the
    /// `verify_batched_graphs` eviction argument).
    pub(super) fn insert_batch_decode_graph(
        &self,
        cache: &mut (HashMap<Vec<u32>, (GraphHandle, u64)>, u64),
        key: Vec<u32>,
        graph: GraphHandle,
    ) {
        let cap = batch_decode_graph_cap(self.buffers.decode_meta().rows());
        for old in lru_insert_graph(cache, cap, key, graph) {
            if let Err(e) = self.gpu.destroy_graph(old) {
                tracing::warn!("batched-decode graph evict: {e:#}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::GraphHandle;

    fn empty_cache() -> (HashMap<Vec<u32>, (GraphHandle, u64)>, u64) {
        (HashMap::new(), 0)
    }

    #[test]
    fn lru_insert_below_cap_drops_nothing() {
        let mut cache = empty_cache();
        let drop = lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(1));
        assert!(drop.is_empty());
        assert_eq!(cache.0.len(), 1);
        let (stored, tick) = cache.0.get(&vec![0]).unwrap();
        assert_eq!(stored.0, 1);
        assert_eq!(*tick, 1);
        assert_eq!(cache.1, 1);
    }

    #[test]
    fn lru_insert_at_cap_evicts_oldest_not_the_new_key() {
        let mut cache = empty_cache();
        lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(10));
        lru_insert_graph(&mut cache, 2, vec![1], GraphHandle(11));
        let drop = lru_insert_graph(&mut cache, 2, vec![2], GraphHandle(12));
        assert_eq!(drop.iter().map(|g| g.0).collect::<Vec<_>>(), vec![10]);
        assert!(cache.0.contains_key(&vec![1]));
        assert!(cache.0.contains_key(&vec![2]));
        assert!(!cache.0.contains_key(&vec![0]));
    }

    #[test]
    fn replacing_an_existing_key_destroys_the_old_handle_without_evicting_peers() {
        let mut cache = empty_cache();
        lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(10));
        lru_insert_graph(&mut cache, 2, vec![1], GraphHandle(11));
        let drop = lru_insert_graph(&mut cache, 2, vec![0], GraphHandle(99));
        assert_eq!(drop.iter().map(|g| g.0).collect::<Vec<_>>(), vec![10]);
        assert_eq!(cache.0.len(), 2);
        assert_eq!(cache.0.get(&vec![0]).unwrap().0.0, 99);
        assert_eq!(cache.0.get(&vec![1]).unwrap().0.0, 11);
        assert_eq!(cache.0.get(&vec![0]).unwrap().1, 3, "replacement is MRU");

        let drop = lru_insert_graph(&mut cache, 2, vec![2], GraphHandle(12));
        assert_eq!(
            drop.iter().map(|handle| handle.0).collect::<Vec<_>>(),
            vec![11],
            "the untouched peer is LRU"
        );
        assert_eq!(cache.0.get(&vec![0]).unwrap().0.0, 99);
        assert_eq!(cache.0.get(&vec![2]).unwrap().0.0, 12);
        assert!(!cache.0.contains_key(&vec![1]));
    }

    #[test]
    fn cap_is_headroom_over_decode_meta_rows() {
        assert_eq!(batch_decode_graph_cap(0), 16);
        assert_eq!(batch_decode_graph_cap(1), 17);
        assert_eq!(batch_decode_graph_cap(32), 48);
        assert_eq!(batch_decode_graph_cap(64), 80);
    }
}
