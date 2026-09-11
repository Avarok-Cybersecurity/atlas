// SPDX-License-Identifier: AGPL-3.0-only

//! Whether a speculative step blocks fusing a prefill chunk into decode.
//!
//! SSOT for a predicate that had THREE copies (`phase_continue_prefills`'s
//! `single_active_with_spec`, reused by the always-mixed gate and the Q12
//! mixed-batch gate, and `run_standard`'s `spec_step_this_tick`), each
//! carrying the same justification comment.
//!
//! # ★ The premise those comments state is no longer true
//!
//! All three read, in substance: *"those `step_*` paths require
//! `active.len() == 1` … spec is off by construction when `active.len() >= 2`,
//! so the mixed branch is safe there."* That was correct when the MTP
//! dispatch cap was 1. It is not correct now: the scheduler dispatches MTP
//! whenever `active.len() <= speculative::mtp_max_seqs()`
//! (`scheduler/mod.rs`, `spec_width_ok`), and that cap DEFAULTS TO 32
//! (`speculative/ladder.rs`, raised 1 → 4 → 8 → 16 → 32 across the ladder
//! campaign). Only the n-gram and self-speculative lanes still require
//! `active.len() == 1`.
//!
//! So over the width range `2..=mtp_max_seqs()` this predicate says "no spec
//! step this tick" while the dispatcher would in fact have run one. The
//! mixed branch is then entered, and it:
//!
//! * clears `pending_drafts` / `pending_draft_conf` on EVERY active sequence,
//! * emits one plain token per sequence through `mixed_forward`, and
//! * sets `did_mixed_step`, which makes `mod.rs` skip `step_mtp` entirely
//!   for the tick.
//!
//! i.e. for the whole duration of a chunked prefill, every concurrent
//! sequence is demoted from a K-wide verify step to one plain token, and
//! then pays a re-bootstrap. That is a decode-step deficit concentrated at
//! low-to-mid concurrency, which is exactly where it is least likely to be
//! noticed on a wide ladder rung.
//!
//! # Why this file does NOT change the value
//!
//! Widening the predicate to the real cap is a one-line change, but it is
//! not obviously the right one: it trades the demotion above for the gap
//! the C=1 rule was introduced to close — "one 8K prefill chunk froze every
//! active decoder for the whole chunk … the single largest scheduler-level
//! concurrency gap" (`run_standard`, 2026-07-25). Which side wins at
//! C=2..8 is a throughput question, and no measurement of it exists. Under
//! measurement discipline that A/B is a GPU leg, so this module names the
//! divergence, gives it one home, and pins its extent in CI instead of
//! guessing at the flip.

/// Whether an in-flight speculative step forbids fusing a prefill chunk
/// into this tick's decode.
///
/// The C=1 rule, unchanged in value — see the module doc for the range
/// over which it disagrees with the scheduler's real dispatch gate.
pub(super) fn mixing_blocked_by_spec(n_active: usize, any_spec: bool) -> bool {
    any_spec && n_active == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_speculation_mixing_is_never_blocked() {
        for n in 0..40usize {
            assert!(!mixing_blocked_by_spec(n, false), "n={n}");
        }
    }

    #[test]
    fn the_c1_rule_blocks_exactly_one_active_sequence() {
        assert!(!mixing_blocked_by_spec(0, true));
        assert!(mixing_blocked_by_spec(1, true));
        for n in 2..40usize {
            assert!(!mixing_blocked_by_spec(n, true), "n={n}");
        }
    }

    /// The guard this module exists for: pin the EXTENT of the divergence
    /// between this predicate and the scheduler's real MTP dispatch width
    /// gate, so it is a tracked, bounded exposure rather than a stale
    /// comment. If someone moves the cap, this test reports the new range;
    /// if the cap ever returns to 1 the divergence is empty and the three
    /// call sites can go back to being trivially correct.
    #[test]
    fn divergence_from_the_real_dispatch_gate_is_exactly_two_through_the_cap() {
        // CI sets neither override; skip rather than assert a value we do
        // not control (same discipline as the ladder's default-shape tests).
        if std::env::var_os("ATLAS_MTP_MAX_SEQS").is_some()
            || std::env::var_os("ATLAS_NO_MTP_K_LADDER").is_some()
        {
            return;
        }
        let cap = spark_model::speculative::mtp_max_seqs();
        // The scheduler dispatches MTP at `active.len() <= cap` (mod.rs).
        let dispatch_would_run = |n: usize| n >= 1 && n <= cap;
        let diverges: Vec<usize> = (0..cap + 8)
            .filter(|&n| dispatch_would_run(n) != mixing_blocked_by_spec(n, true))
            .collect();
        let expected: Vec<usize> = (2..=cap).collect();
        assert_eq!(
            diverges, expected,
            "the widths where mixing is allowed but a spec step would have run \
             are no longer 2..={cap}; re-read this module's doc before changing it"
        );
        // And the divergence must be non-empty on the shipped default, or
        // the module doc is describing a problem that no longer exists.
        assert!(
            cap >= 32,
            "dispatch cap fell to {cap}; divergence range shrank"
        );
    }
}
