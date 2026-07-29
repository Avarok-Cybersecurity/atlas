// SPDX-License-Identifier: AGPL-3.0-only

//! [`TierEvict`] — what an SSM spill-tier eviction decided to do with the
//! victim. Split out of `prefix_cache.rs` (500-LoC cap).

/// Outcome of [`crate::prefix_cache::PrefixCache::evict_snapshot_to_tier`].
///
/// The spill side of the tier has a cost gate (`ATLAS_SSM_SPILL_MIN_TOKENS`)
/// mirroring the fault-in side's `ATLAS_SSM_FAULT_MIN_TOKENS`, and the gate has
/// to be applied at VICTIM SELECTION, not at the byte move. Gating only the
/// byte move would leave the index entry marked `tiered` — findable by
/// `lookup_tiered` — with no blob behind it, so every warm turn would pay a
/// full-blob allocation plus a `store.get` only to learn it misses.
///
/// Both arms free the victim's HBM slot, so the caller's invariant "a reclaim
/// always yields a slot" is preserved either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierEvict {
    /// Move the victim's bytes to the tier and keep its index entry findable
    /// (`tiered = true`) so a warm turn faults it back instead of recomputing.
    Spill {
        /// The freed HBM snapshot slot (spill FROM here, then free it).
        slot: usize,
        /// Tier key = the entry's prefix hash.
        key: u64,
        /// Victim depth in tokens — what the spill's cost buys back.
        depth: usize,
    },
    /// Too shallow to be worth a spill: the index entry was REMOVED, no tier
    /// key exists, and `tier_spills` was not incremented. Identical to the
    /// pre-tier drop path.
    Drop {
        /// The freed HBM snapshot slot.
        slot: usize,
        /// Victim depth in tokens (for the gate-skip log line).
        depth: usize,
    },
}

impl TierEvict {
    /// The freed HBM slot, whichever arm was taken.
    pub fn slot(&self) -> usize {
        match *self {
            TierEvict::Spill { slot, .. } | TierEvict::Drop { slot, .. } => slot,
        }
    }
}
