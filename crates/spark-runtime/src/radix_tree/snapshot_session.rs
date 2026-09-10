// SPDX-License-Identifier: AGPL-3.0-only

//! The snapshot lookup's session gate — ONE predicate, two callers.
//!
//! Its own module because it is the shared thing: `snapshot::lookup` and
//! `snapshot_tier::lookup_tiered` both read it, and it used to exist as two
//! copies of the same condition in those two files. A shared rule that lives
//! inside one of its callers invites the next person to edit the copy in front
//! of them. (It also took `snapshot.rs` past the repository's 500-line cap,
//! which is the cheaper reason and the one CI notices.)

use super::snapshot::SnapshotEntry;

/// Does the session gate reject this entry for this lookup?
///
/// ★ ONE predicate, read by BOTH lookups. It was two copies of the same
/// condition — here and in `snapshot_tier::lookup_tiered` — and only the
/// tier-aware one is on the serving path. A change applied to the copy inside
/// `lookup` (which is `#[allow(dead_code)]`) would have looked correct, passed
/// review, and changed nothing that runs.
///
/// The shipped rule gates ONLY `is_tail`, on the ground stated on
/// [`SnapshotEntry::is_tail`]: a non-tail entry stores state at EXACTLY
/// `token_count`, so it is "a pure function of the verified token prefix" and
/// safe cross-session like the KV radix. That reasoning holds in EXACT
/// arithmetic and fails in floating point. A recurrent state at `token_count`
/// is path-dependent — the bits depend on how the prefix was chunked and
/// batched on the pass that produced them, which depends on what else the
/// scheduler had in flight. Two entries with identical
/// `(token_count, prefix_hash)` can hold different bits, and the scans take
/// the deepest under a STRICT `>`, so among equal-depth entries the lowest
/// index wins — and index order is pool insertion/eviction history. Which
/// state a request restores is thus a function of what ran before it.
///
/// In production that is the right trade, deliberately made: the blanket gate
/// was removed because `session_hash` is `hash(first min(len, 1024) prompt
/// tokens)` and is UNSTABLE across turns of a growing conversation, so it
/// rejected every valid non-tail anchor from a prior turn — the
/// warm-TTFT-climb root cause.
///
/// Under `--hermetic` the trade goes the other way: a known-answer test must
/// not read state produced by another request, and warm TTFT is not what a KAT
/// measures. Hermetic therefore gates EVERY entry and pays the warm-turn cost.
/// `hermetic` is a PARAMETER, not a read of the global, so both answers are
/// reachable from a test in one process — `hermetic_enabled()` caches in a
/// `OnceLock` and cannot be flipped once read. Call sites resolve it once
/// before their scan and pass it down.
pub(super) fn session_gate_blocks(
    entry: &SnapshotEntry,
    session_hash: u64,
    hermetic: bool,
) -> bool {
    if !entry.is_tail && !hermetic {
        return false;
    }
    session_hash == 0 || entry.session_hash != session_hash
}

#[cfg(test)]
#[path = "tests/snapshot_session_gate.rs"]
mod snapshot_session_gate;
