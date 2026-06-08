// SPDX-License-Identifier: AGPL-3.0-only

//! Phase-C decode-time SSM-snapshot ring.
//!
//! For hybrid (attention + Mamba/SSM) models the boundary rollback
//! ([`super::rollback::rollback_to_boundary`]) cannot simply lower a
//! cursor the way it rewinds the paged attention KV cache: the SSM
//! `h_state` + `conv_state` are recurrent, advanced in-place by every
//! decoded token. Undoing them requires a *snapshot* of the state as it
//! was at an earlier token.
//!
//! This ring records, per active sequence, a small bounded set of SSM
//! snapshots taken **at boundary tokens during normal decode** — the
//! same boundary tokens [`super::rollback::find_last_boundary`] selects.
//! Each entry pairs the absolute generated-token position with the
//! decode-rollback snapshot *slot* the model wrote the GPU-side state
//! into (the GPU memory itself is owned by the model's
//! `SsmSnapshotPool`; this struct only tracks slot indices).
//!
//! Snapshotting only at boundary tokens — not every token — keeps the
//! cost bounded: at most `capacity` D2D copies are live per sequence,
//! and the ring evicts oldest-first so a long generation never grows
//! the set. `capacity` is the model's
//! `decode_rollback_ring_slots()` — sized `ROLLBACK_RESTEER_CAP + 1`
//! so every permitted rollback has a distinct snapshot plus the current
//! boundary.
//!
//! Memory cost: each slot stores `h_state + conv_state` for **all** SSM
//! layers. Per marconi.md that is ~77 MB/slot for a 122B/36-layer model;
//! a 35B-A3B hybrid is far smaller. The full decode-rollback region is
//! `capacity × max_batch_size` slots, allocated once at model init.

/// One recorded boundary snapshot: the generated-token position it was
/// taken at, and the model-side snapshot slot holding the GPU state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmRingEntry {
    /// Length of `output_tokens` at the moment the snapshot was taken —
    /// i.e. the snapshot reflects SSM state *after* generating exactly
    /// this many tokens. A rollback that keeps `keep_len` tokens needs
    /// the entry whose `token_position == keep_len`.
    pub token_position: usize,
    /// Decode-rollback snapshot slot index in `[0, capacity)` that the
    /// model wrote this sequence's SSM state into.
    pub snapshot_slot: usize,
}

/// Bounded ring of decode-time SSM snapshots for one active sequence.
///
/// Pure data structure — no GPU / I/O. The scheduler drives the actual
/// `save`/`restore` D2D copies through the `Model` trait; this struct
/// only decides *which* snapshot slot to (re)use and *which* boundary a
/// rollback can reach.
#[derive(Debug, Clone)]
pub struct SsmDecodeRing {
    /// Live entries, oldest first. Length never exceeds `capacity`.
    entries: Vec<SsmRingEntry>,
    /// Maximum live entries == number of decode-rollback snapshot slots
    /// reserved for this sequence. `0` disables the ring entirely
    /// (pure-attention models, or SSM models with no reserved region).
    capacity: usize,
    /// Round-robin cursor over `[0, capacity)` for slot assignment.
    /// Because the ring evicts oldest-first and slots are reused in the
    /// same order, advancing this modulo `capacity` always yields a slot
    /// not referenced by any live entry once the ring is full.
    next_slot: usize,
}

impl SsmDecodeRing {
    /// Create a ring with room for `capacity` snapshots. `capacity == 0`
    /// produces a permanently-disabled ring (every `record` is a no-op,
    /// every lookup returns `None`).
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            next_slot: 0,
        }
    }

    /// Whether this ring can hold snapshots. `false` for pure-attention
    /// models — the scheduler then skips all SSM snapshot work.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    /// Reserve the snapshot slot the next boundary should be written
    /// into, registering the `(token_position, slot)` entry.
    ///
    /// When the ring is full this **evicts the oldest entry** and
    /// returns its slot for reuse — the freed GPU snapshot is about to
    /// be overwritten by the caller's `save`. Returns `None` only when
    /// the ring is disabled (`capacity == 0`).
    ///
    /// The caller MUST, on a non-`None` return, issue the model-side
    /// `save_decode_ssm_snapshot` into the returned slot; otherwise the
    /// entry would point at stale GPU state.
    pub fn record(&mut self, token_position: usize) -> Option<usize> {
        if self.capacity == 0 {
            return None;
        }
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.capacity;

        if self.entries.len() == self.capacity {
            // Full: drop the oldest entry; its slot is `slot` (the ring
            // reuses slots in insertion order, so the oldest live entry
            // always holds the slot we are about to hand out).
            debug_assert_eq!(self.entries[0].snapshot_slot, slot);
            self.entries.remove(0);
        }
        self.entries.push(SsmRingEntry {
            token_position,
            snapshot_slot: slot,
        });
        Some(slot)
    }

    /// Find the snapshot slot for the entry whose `token_position`
    /// exactly equals `keep_len` — i.e. the snapshot of SSM state right
    /// after the boundary token a rollback wants to resume from.
    ///
    /// Returns `None` when no live snapshot matches; the caller MUST
    /// then decline the rollback (the SSM state cannot be restored).
    pub fn slot_for_position(&self, keep_len: usize) -> Option<usize> {
        self.entries
            .iter()
            .find(|e| e.token_position == keep_len)
            .map(|e| e.snapshot_slot)
    }

    /// The set of token positions that currently have a live snapshot,
    /// most-recent last. Used by boundary selection to restrict the
    /// rollback target to a boundary that can actually be restored.
    pub fn snapshot_positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.iter().map(|e| e.token_position)
    }

    /// Drop every entry whose `token_position` is strictly greater than
    /// `keep_len`. Called after a successful rollback: snapshots taken
    /// in the now-discarded degenerate tail are invalid and their slots
    /// must become reusable. The snapshot at exactly `keep_len` (the one
    /// just restored) is kept — generation resumes from it.
    pub fn truncate_after(&mut self, keep_len: usize) {
        self.entries.retain(|e| e.token_position <= keep_len);
    }

    /// Number of live snapshots. Test/diagnostic helper.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
#[path = "ssm_decode_ring_tests.rs"]
mod ssm_decode_ring_tests;
