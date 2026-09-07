// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-rollback AUX ring — the companion to the SSM decode ring.
//!
//! # Why this exists
//!
//! The content-loop watchdog's rollback (`scheduler/rollback.rs`) restores the
//! SSM recurrent state from the decode ring and lowers `seq_len`, but nothing
//! rewound the auxiliary per-sequence state that hybrid models also carry: the
//! QSA indexer cursor and the PLE n-gram history. On Qwen3.8-Flash-Next the
//! very next decode then failed
//!
//! ```text
//! QSA: decode at pos 2864 but 2932 tokens ingested — the indexer cache lost sync
//! ```
//!
//! and the request died with a 500. The only aux rewind anywhere in the tree
//! was the speculative-verify commit hook; the watchdog path never touched aux
//! state at all.
//!
//! # The shape
//!
//! Mirror the SSM treatment with the AUDITED SNAPSHOT PATH rather than a new
//! arithmetic rewind of the cursors. `collect_aux_states` / `apply_aux_states`
//! are the same `snapshot_aux` / `restore_aux` layer hooks Marconi prefix
//! caching already uses, so a restore is byte-exact by construction and cannot
//! get cursor arithmetic subtly wrong.
//!
//! Entries are keyed by the SAME `(seq.slot_idx, ring_slot)` pair as the SSM
//! ring, so the two halves cannot drift: `snapshot_boundary_if_ssm` saves both
//! or drops the ring entry, and `rollback_to_boundary` restores both or
//! declines with `RollbackFallback::NoAuxSnapshot` (a stream-safe hard stop —
//! the same honesty as the existing `NoSsmSnapshot`).
//!
//! Blobs live on the HOST. They are small and bounded: one per aux-carrying
//! layer per live `(slot, ring_slot)`, and the ring is
//! `DECODE_ROLLBACK_RING_SLOTS` deep per sequence.

use std::collections::HashMap;

use anyhow::Result;
use parking_lot::Mutex;

use crate::traits::SequenceState;

/// Host-side aux blobs for the decode-rollback ring, keyed by
/// `(sequence slot, ring slot)`.
#[derive(Default)]
pub(crate) struct DecodeAuxRing {
    entries: Mutex<HashMap<(usize, usize), Vec<(u32, Vec<u8>)>>>,
}

impl DecodeAuxRing {
    /// Store `blobs` for `(slot, ring_slot)`, replacing any previous entry.
    pub(crate) fn put(&self, slot: usize, ring_slot: usize, blobs: Vec<(u32, Vec<u8>)>) {
        self.entries.lock().insert((slot, ring_slot), blobs);
    }

    /// Take the blobs for `(slot, ring_slot)`, if any. Cloned rather than
    /// removed: a declined rollback must be able to retry, and the entry is
    /// invalidated explicitly by [`Self::forget_from`] / [`Self::forget_slot`].
    pub(crate) fn get(&self, slot: usize, ring_slot: usize) -> Option<Vec<(u32, Vec<u8>)>> {
        self.entries.lock().get(&(slot, ring_slot)).cloned()
    }

    /// Drop the entry for exactly `(slot, ring_slot)` — used when the SSM half
    /// failed to save, so no orphan aux blob is left for a later boundary to
    /// match against.
    pub(crate) fn forget(&self, slot: usize, ring_slot: usize) {
        self.entries.lock().remove(&(slot, ring_slot));
    }

    /// Drop every entry belonging to `slot` (sequence teardown / slot reuse).
    pub(crate) fn forget_slot(&self, slot: usize) {
        self.entries.lock().retain(|&(s, _), _| s != slot);
    }

    /// Number of live entries — diagnostics and tests only.
    pub(crate) fn len(&self) -> usize {
        self.entries.lock().len()
    }
}

impl super::types::TransformerModel {
    /// Save `seq`'s aux state into ring slot `ring_slot`. No-op (and no entry)
    /// when the model carries no aux state, so non-hybrid models are untouched.
    pub(in crate::model) fn save_decode_aux_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.requires_aux_state() {
            return Ok(());
        }
        let blobs = self.collect_aux_states(seq, self.gpu.default_stream())?;
        // An aux-carrying model that produced NO blobs would silently restore
        // nothing later; treat it as a failure so the caller drops the ring
        // entry rather than banking a half-snapshot.
        if blobs.is_empty() {
            anyhow::bail!(
                "decode aux snapshot: model requires aux state but no layer produced a blob"
            );
        }
        self.decode_aux_ring.put(seq.slot_idx, ring_slot, blobs);
        Ok(())
    }

    /// Restore the aux state saved for `ring_slot`. `Err` when the companion is
    /// missing — the caller MUST decline the rollback then, never proceed.
    pub(in crate::model) fn restore_decode_aux_snapshot_dispatch(
        &self,
        seq: &mut SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.requires_aux_state() {
            return Ok(());
        }
        let Some(blobs) = self.decode_aux_ring.get(seq.slot_idx, ring_slot) else {
            anyhow::bail!(
                "decode aux snapshot missing for slot {} ring_slot {ring_slot} — \
                 rolling back without the aux rewind would desync the QSA indexer",
                seq.slot_idx
            );
        };
        self.apply_aux_states(seq, &blobs, self.gpu.default_stream())
    }

    /// Forget the aux companion for `(seq.slot_idx, ring_slot)`.
    pub(in crate::model) fn forget_decode_aux_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) {
        if self.requires_aux_state() {
            self.decode_aux_ring.forget(seq.slot_idx, ring_slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeAuxRing;

    #[test]
    fn ring_is_keyed_by_slot_and_ring_slot_and_forgets_precisely() {
        let r = DecodeAuxRing::default();
        r.put(1, 0, vec![(0, vec![1, 2, 3])]);
        r.put(1, 1, vec![(0, vec![4])]);
        r.put(2, 0, vec![(0, vec![5])]);
        assert_eq!(r.len(), 3);
        // Distinct sequences never collide on the same ring slot — the bug
        // class this key exists to prevent.
        assert_eq!(r.get(1, 0).unwrap()[0].1, vec![1, 2, 3]);
        assert_eq!(r.get(2, 0).unwrap()[0].1, vec![5]);
        // A get does not consume: a declined rollback may retry.
        assert!(r.get(1, 0).is_some());
        assert_eq!(r.len(), 3);
        // forget removes exactly one entry.
        r.forget(1, 0);
        assert!(r.get(1, 0).is_none());
        assert!(r.get(1, 1).is_some());
        assert!(r.get(2, 0).is_some());
        // forget_slot removes every entry of that sequence and no other.
        r.forget_slot(1);
        assert!(r.get(1, 1).is_none());
        assert!(r.get(2, 0).is_some());
        assert_eq!(r.len(), 1);
        // A missing companion reads as None — the caller turns that into a
        // declined rollback, never a silent proceed.
        assert!(r.get(9, 9).is_none());
    }
}
