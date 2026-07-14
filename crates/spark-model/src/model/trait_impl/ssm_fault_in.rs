// SPDX-License-Identifier: AGPL-3.0-only

//! Shared SSM-snapshot tier fault-in helper.
//!
//! When the radix prefix match finds a spilled anchor (`ssm_snapshot` is
//! `None` but `ssm_snapshot_tier_key` is present), fault its bytes back into a
//! resident Marconi slot and treat it as resident for the restore — converting
//! a full-prefix SSM recompute into a tier restore.
//!
//! Extracted so all three prefill paths share one implementation
//! (`prefill_b_prefix_lookup`, `prefill_a`/`prefill_dispatch`, and
//! `prefill_c`/`prefill_twophase_dispatch`). Before this the logic lived inline
//! in `prefill_b/prefix_lookup.rs` only, so `prefill_a`/`prefill_c` ignored the
//! tier key and always recomputed (handoff task #6). The cost-aware depth gate
//! (task #5) lives here too, so it applies uniformly across the three paths.

#![allow(dead_code)]

use spark_runtime::prefix_cache::PrefixMatch;

use super::super::types::TransformerModel;

/// Minimum tier-snapshot depth (in tokens) below which a fault-in is skipped in
/// favour of recompute. Overridable via `ATLAS_SSM_FAULT_MIN_TOKENS`; `0`
/// disables the gate (always fault when a tier key exists).
///
/// Cost model: a fault-in is a fixed ~28 ms — the full spill blob (every SSM
/// layer's `h`+`conv`, ~66 MB on Holo-3.1-35B) is read back over RDMA/NVMe at
/// ~2.5 GB/s (~26 ms) plus an H2D scatter + sync — and then the suffix prefill
/// still replays SSM over `[snap_tok, matched)`. It only pays off when it
/// elides MORE prefix SSM recompute than it costs. Below the threshold the
/// matched prefix is so shallow that recomputing its SSM from scratch is
/// cheaper than the fixed blob read, so we skip the fault and recompute.
///
/// Default `256` (16 blocks) is conservative: GDN prefill over a few hundred
/// tokens is well under 28 ms, so below it recompute is unambiguously cheaper,
/// while the tuned tail-clustered checkpoints (near the ~15 K prompt tail) sit
/// far above it and always fault in. A low threshold minimises wrongly skipping
/// a beneficial deep fault. Tune per template via the miss-depth histogram.
const DEFAULT_FAULT_MIN_TOKENS: usize = 256;

fn fault_in_min_tokens() -> usize {
    parse_fault_min_tokens(std::env::var("ATLAS_SSM_FAULT_MIN_TOKENS").ok())
}

/// Pure parse of `ATLAS_SSM_FAULT_MIN_TOKENS`: an unset or unparseable value
/// falls back to [`DEFAULT_FAULT_MIN_TOKENS`]; `0` disables the gate.
fn parse_fault_min_tokens(raw: Option<String>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_FAULT_MIN_TOKENS)
}

/// Whether a tier snapshot of `depth` tokens is too shallow to be worth a
/// fixed-cost fault-in vs. recomputing the prefix (task #5). `min_depth == 0`
/// disables the gate.
fn should_skip_fault_for_depth(depth: usize, min_depth: usize) -> bool {
    depth < min_depth
}

impl TransformerModel {
    /// Resolve the **effective** SSM snapshot for a prefix match, folding a
    /// resident hit with a spill-tier fault-in. Shared by all three prefill
    /// paths (`prefill_a`/`prefill_b`/`prefill_c`).
    ///
    /// When the anchor was SPILLED (`ssm_snapshot` is `None` but a tier key is
    /// present), [`try_fault_in_ssm_snapshot`](Self::try_fault_in_ssm_snapshot)
    /// faults its bytes into a resident slot — converting a full-prefix SSM
    /// recompute into a tier restore. Returns `(eff_snapshot, eff_depth)`:
    /// - `eff_snapshot` = the resident slot, else the faulted-in slot, else
    ///   `None` (nothing to restore → recompute the prefix).
    /// - `eff_depth` = the restored state's token depth. A faulted anchor's
    ///   resident `ssm_snapshot_tokens` field is `0`, so its real depth lives in
    ///   `ssm_snapshot_tier_tokens`; callers MUST use this folded depth as the
    ///   skip point or a tier restore skips nothing (warm hit slower than a
    ///   plain recompute).
    ///
    /// A faulted slot is byte-identical to the snapshot it spilled from, so the
    /// restore/skip logic at each call site is unchanged.
    pub(in crate::model) fn eff_ssm_snapshot(
        &self,
        prefix_match: &PrefixMatch,
        session_hash: u64,
        stream: u64,
    ) -> (Option<usize>, usize) {
        let faulted_snap = self.try_fault_in_ssm_snapshot(prefix_match, session_hash, stream);
        let eff_snapshot = prefix_match.ssm_snapshot.or(faulted_snap);
        let eff_snapshot_tokens = if faulted_snap.is_some() {
            prefix_match.ssm_snapshot_tier_tokens
        } else {
            prefix_match.ssm_snapshot_tokens
        };
        (eff_snapshot, eff_snapshot_tokens)
    }

    /// Attempt to fault a spilled SSM snapshot for `prefix_match` back into a
    /// resident Marconi slot, re-homing it onto `session_hash`. Returns the
    /// faulted-in slot on success, or `None` when there is nothing to fault
    /// (resident snapshot already present, no tier store, no tier key), the
    /// cost-aware depth gate rejects it (task #5), the pool is fully mid-flight,
    /// or the blob is gone from the tier (miss → caller recomputes).
    ///
    /// On success the caller must treat the returned slot as the effective
    /// snapshot with depth `prefix_match.ssm_snapshot_tier_tokens` (see
    /// `eff_snapshot`/`eff_snapshot_tokens` at the call sites): the resident
    /// `ssm_snapshot_tokens` field is `0` for a faulted-in anchor.
    pub(in crate::model) fn try_fault_in_ssm_snapshot(
        &self,
        prefix_match: &PrefixMatch,
        session_hash: u64,
        stream: u64,
    ) -> Option<usize> {
        // Only relevant when the radix match had no RESIDENT snapshot but did
        // carry a tier key — otherwise the resident path handles it.
        if prefix_match.ssm_snapshot.is_some() {
            return None;
        }
        let store = self.ssm_tier_store.as_deref()?;
        let key = prefix_match.ssm_snapshot_tier_key?;

        // #5 cost-aware gate: skip the fixed-cost fault when the matched prefix
        // is shallow enough that recomputing its SSM is cheaper than the blob
        // read + replay.
        let depth = prefix_match.ssm_snapshot_tier_tokens;
        let min_depth = fault_in_min_tokens();
        if should_skip_fault_for_depth(depth, min_depth) {
            tracing::info!(
                "SSM tier fault-in SKIPPED (cost gate): tier snapshot depth {depth} < \
                 ATLAS_SSM_FAULT_MIN_TOKENS={min_depth} — recomputing the shallow prefix \
                 is cheaper than a ~28ms blob fault + replay"
            );
            return None;
        }

        // `acquire_or_spill_slot` spills a resident victim to make room when the
        // pool is full, so a warm hit isn't lost to a busy pool; `None` only if
        // every slot is mid-flight.
        let slot = self.ssm_snapshots.acquire_or_spill_slot(
            self.prefix_cache.as_ref(),
            store,
            self.gpu.as_ref(),
        )?;
        match self
            .ssm_snapshots
            .fault_in_slot(slot, key, store, self.gpu.as_ref(), stream)
        {
            Ok(true) => {
                self.prefix_cache.promote_snapshot(key, slot);
                // Re-home the session owner onto the fresh slot. Without this
                // the slot is untagged (or carries a spill victim's stale tag)
                // and the `session_matches` gate at the call site rejects the
                // just-faulted state → full recompute. `lookup`/`lookup_tiered`
                // already filtered by session, so `session_hash` is the
                // rightful owner.
                self.ssm_snapshots.tag_session(slot, session_hash);
                tracing::info!(
                    "SSM tier fault-in: restored spilled snapshot at token {depth} into slot {slot}"
                );
                Some(slot)
            }
            // Miss (blob gone) or error: return the slot, recompute.
            _ => {
                self.ssm_snapshots.free(slot);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_FAULT_MIN_TOKENS, parse_fault_min_tokens, should_skip_fault_for_depth};

    #[test]
    fn min_tokens_defaults_when_unset_or_garbage() {
        assert_eq!(parse_fault_min_tokens(None), DEFAULT_FAULT_MIN_TOKENS);
        assert_eq!(
            parse_fault_min_tokens(Some("not-a-number".into())),
            DEFAULT_FAULT_MIN_TOKENS
        );
        assert_eq!(
            parse_fault_min_tokens(Some("".into())),
            DEFAULT_FAULT_MIN_TOKENS
        );
    }

    #[test]
    fn min_tokens_parses_explicit_values() {
        assert_eq!(parse_fault_min_tokens(Some("0".into())), 0);
        assert_eq!(parse_fault_min_tokens(Some("1024".into())), 1024);
    }

    #[test]
    fn gate_skips_shallow_faults_only() {
        let min = DEFAULT_FAULT_MIN_TOKENS;
        // Shallow: recompute is cheaper → skip the fault.
        assert!(should_skip_fault_for_depth(0, min));
        assert!(should_skip_fault_for_depth(min - 1, min));
        // At/above threshold: fault in.
        assert!(!should_skip_fault_for_depth(min, min));
        assert!(!should_skip_fault_for_depth(min + 1, min));
        assert!(!should_skip_fault_for_depth(15_000, min));
    }

    #[test]
    fn gate_disabled_never_skips() {
        // min_depth == 0 disables the gate: even a depth-0 tier key faults in.
        assert!(!should_skip_fault_for_depth(0, 0));
        assert!(!should_skip_fault_for_depth(1, 0));
    }
}
