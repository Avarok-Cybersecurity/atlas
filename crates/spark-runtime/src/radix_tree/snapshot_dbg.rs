// SPDX-License-Identifier: AGPL-3.0-only

//! Per-lookup SSM-snapshot candidate diagnostics (`ATLAS_SNAP_LOOKUP_DBG`).
//!
//! Answers one question and only that question: at the instant a warm turn
//! looks the index up, WHICH entries were selectable, and why were the others
//! not? Evaluated in-place inside the lookup's critical section, so it cannot
//! be confounded by log ordering, by a save that had not landed yet, or by an
//! entry belonging to another sequence — every rejection is attributed to the
//! filter that actually rejected it.
//!
//! Off unless the env var is set; the classification recomputes a prefix hash
//! per entry, which is O(depth) and must never run on the serving path.

use super::hash_token_prefix;
use super::snapshot::SsmSnapshotIndex;

impl SsmSnapshotIndex {
    /// Emit the candidate breakdown for one `lookup_tiered` call.
    ///
    /// `sel` is the depth the lookup chose (`None` on a miss).
    ///
    /// - `ok`   : passed every filter, i.e. the set the selector chose from.
    ///            The deepest of these MUST equal `sel` — a mismatch is a
    ///            genuine selection defect.
    /// - `above`: prefix-valid, session-valid, tail-valid entries that were
    ///            rejected ONLY because their depth exceeds the block-aligned
    ///            KV `matched_tokens`. These are NOT selectable today (the
    ///            prefill's SSM skip point doubles as the KV write floor), so
    ///            they are the design cap, not a selection bug.
    pub(super) fn log_lookup_candidates(
        &self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
        adapter_id: u64,
        sel: Option<usize>,
    ) {
        let (mut ok, mut above): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let (mut n_tier, mut n_sess, mut n_tail, mut n_hash, mut n_long) = (0, 0, 0, 0, 0);
        for e in &self.entries {
            if e.tiered {
                n_tier += 1;
            }
            if session_hash != 0 && e.session_hash != 0 && e.session_hash != session_hash {
                n_sess += 1;
                continue;
            }
            if e.is_tail && (session_hash == 0 || e.session_hash != session_hash) {
                n_tail += 1;
                continue;
            }
            if e.token_count > tokens.len() {
                n_long += 1;
                continue;
            }
            if hash_token_prefix(tokens, e.token_count, adapter_id) != e.prefix_hash {
                n_hash += 1;
                continue;
            }
            if e.token_count > matched_tokens {
                above.push(e.token_count);
            } else {
                ok.push(e.token_count);
            }
        }
        ok.sort_unstable();
        above.sort_unstable();
        tracing::info!(
            "snap-lookup: total={} matched={matched_tokens} sel={sel:?} n={} \
             ok={ok:?} above={above:?} \
             rej_tiered={n_tier} rej_sess={n_sess} rej_tail={n_tail} rej_hash={n_hash} \
             rej_longer_than_prompt={n_long}",
            tokens.len(),
            self.entries.len(),
        );
    }
}
