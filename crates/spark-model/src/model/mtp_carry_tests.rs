// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the cross-turn drafter carry.
//!
//! Split out of `mtp_carry.rs` to keep that file under the repository's
//! 500-LoC cap; the module is included via `#[path]` below its code, the
//! same pattern `toolcall_tests.rs` and `bfcl_tests.rs` use.

use super::*;

use super::*;

/// The session every fixture below belongs to.
const SESSION: u64 = 0x5E55_1014;
/// A different live session, used as the intruder.
const OTHER: u64 = 0x0DD0_0DD0;

fn carried(tokens: &[u32], rows: usize, last_pair_key: Option<usize>) -> CarriedDrafter {
    CarriedDrafter {
        block_table: vec![1, 2, 3],
        rows,
        last_pair_key,
        tokens: tokens.to_vec(),
        session_hash: SESSION,
    }
}

/// ★ THE CROSS-REQUEST CHANNEL. A different session may not adopt the
/// slot, no matter how much of the prompt agrees — here the entry's tokens
/// are a FULL prefix of the intruder's prompt, which is the most
/// permissive case the old prefix-only rule had.
///
/// Before the session gate this returned `Some((4, 3))`.
#[test]
fn a_foreign_session_cannot_adopt_the_slot() {
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), Some((4, 3)));
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], OTHER), None);
}

/// A request the scheduler stamped with no session cannot adopt either.
/// Zero is "unknown", not "wildcard": there is nothing to verify ownership
/// against, and blind beats poisoned.
#[test]
fn an_unstamped_request_cannot_adopt_the_slot() {
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], 0), None);
    // And an entry deposited without a stamp is not adoptable by anyone,
    // including another unstamped request.
    let mut unstamped = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    unstamped.session_hash = 0;
    assert_eq!(unstamped.usable_by(&[1, 2, 3, 4, 5, 6, 7], 0), None);
    assert_eq!(unstamped.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), None);
}

/// ★ THE REPORTED SHAPE, in the terms it was reported in: two unrelated
/// requests rendered through ONE chat template share a long leading run of
/// tokens, so the old `common >= 2` admission test passed on every pair.
/// Prefix agreement is a bound on WHICH rows are reusable; it was never an
/// identity, and this pins that it is no longer read as one.
#[test]
fn the_shared_template_prefix_is_not_an_identity() {
    // 64 tokens of identical system/tool preamble, then the two requests
    // diverge into their own user turns.
    let template: Vec<u32> = (0..64).collect();
    let mut mine = template.clone();
    mine.extend_from_slice(&[900, 901, 902]);
    let mut theirs = template.clone();
    theirs.extend_from_slice(&[700, 701, 702]);

    let c = CarriedDrafter {
        block_table: vec![1, 2, 3],
        rows: 60,
        last_pair_key: Some(59),
        tokens: mine.clone(),
        session_hash: SESSION,
    };
    // The prefix rule alone would have admitted the stranger: 64 tokens in
    // common is far more than the two it required.
    assert_eq!(c.common_prefix_len(&theirs), 64);
    assert_eq!(c.usable_by(&theirs, OTHER), None);
    // The same session's own next turn still adopts, and still gets the
    // full entry — the gate narrows nothing it should not.
    assert_eq!(c.usable_by(&mine, SESSION), Some((60, 59)));
}

/// ★ The arming rule, and the trap it exists to name: a serve can report
/// `carry=ON` from its `DrafterContext` and still never carry, because the
/// MTP dispatch cap defaults to 32 and force-disables it. Both callers of
/// this rule — the runtime gate and the startup report — must agree, which
/// is why there is exactly one function.
#[test]
fn configured_carry_is_not_armed_carry_under_a_multi_seq_cap() {
    use crate::model::drafter_context::DrafterContext;
    let both = DrafterContext {
        prefill: true,
        carry: true,
    };
    let prefill_only = DrafterContext {
        prefill: true,
        carry: false,
    };
    // Single-sequence dispatch: configured ON is armed.
    assert!(carry_armed_with(both, false));
    // The SHIPPED default (cap 32 => multi_seq): configured ON, inert.
    assert!(
        !carry_armed_with(both, true),
        "a >1 dispatch cap must force the carry off"
    );
    // Configured off is off either way.
    assert!(!carry_armed_with(prefill_only, false));
    assert!(!carry_armed_with(prefill_only, true));
}

/// ★ THE MERGE-ACROSS-OWNERS ORDERING. A's last prefill chunk lands AFTER
/// B has been admitted and written its own rows. The `alloc_sequence` reset
/// cannot catch this — the reset ran before B wrote — so this is the case
/// the old "per-sequence by construction" claim was flatly wrong about.
///
/// Written as the scheduler orders it, because either function alone is
/// trivially green; it is the COMPOSITION that used to be broken.
#[test]
fn a_late_chunk_takes_the_interval_over_instead_of_claiming_foreign_rows() {
    // What the unstamped code did, kept as the explicit negative control:
    // A's late chunk merged into B's interval and swallowed rows 0..600.
    let unstamped = merge_interval(merge_interval((0, 0), 0, 600), 500, 500);
    assert_eq!(
        unstamped,
        (0, 1000),
        "control: without a stamp the merge claims B's rows 0..600"
    );

    // Stamped, in order. A = gen 1, B = gen 2.
    let a0 = stamped_merge(StoreRange::EMPTY, 1, 0, 500);
    assert_eq!(
        a0,
        StoreRange {
            owner: 1,
            lo: 0,
            hi: 500
        }
    );
    let b0 = stamped_merge(StoreRange::EMPTY, 2, 0, 600); // B admitted, resets, writes
    assert_eq!(
        b0,
        StoreRange {
            owner: 2,
            lo: 0,
            hi: 600
        }
    );
    let a1 = stamped_merge(b0, 1, 500, 500); // A's LAST chunk
    assert_eq!(
        a1,
        StoreRange {
            owner: 1,
            lo: 500,
            hi: 1000
        },
        "a foreign interval must be taken over, never extended"
    );
    // And A reads only what A wrote.
    assert_eq!(a1.visible_to(1), (500, 1000));
}

/// ★ THE DEFERRED-CONSUME ORDERING, carried all the way to `plan_append` —
/// the function that actually decides whether foreign hiddens get used.
///
/// Warm seq A wrote [12000, 12100). B is then admitted and cold-prefills
/// 13000 tokens, leaving B's hiddens at rows 0..13000. A proposes.
#[test]
fn a_foreign_interval_cannot_satisfy_an_append_plan() {
    let a = stamped_merge(StoreRange::EMPTY, 1, 12000, 100);
    let b = stamped_merge(a, 2, 0, 13000); // B admitted and cold-prefills
    assert_eq!(
        b,
        StoreRange {
            owner: 2,
            lo: 0,
            hi: 13000
        }
    );

    // Control: this is exactly what the unstamped read produced, and it
    // yielded 99 rows of A's tokens paired with B's hiddens.
    assert_eq!(
        plan_append(11999, 12100, 0, 13000),
        Some(AppendPlan {
            first_key: 12000,
            rows: 99
        }),
        "control: the unstamped read satisfies the plan with B's rows"
    );

    // Stamped: A sees nothing of B's, and the plan is refused.
    assert_eq!(b.visible_to(1), (0, 0));
    let (lo, hi) = b.visible_to(1);
    assert_eq!(
        plan_append(11999, 12100, lo, hi),
        None,
        "a foreign interval must not satisfy an append"
    );
}

/// The guard must not refuse everything — a guard that refuses the honest
/// case too is indistinguishable from deleting the feature. Same numbers as
/// the control above, and it must still be `Some`.
#[test]
fn a_sequence_still_appends_from_its_own_rows() {
    let a = stamped_merge(StoreRange::EMPTY, 1, 12000, 100);
    let (lo, hi) = a.visible_to(1);
    assert_eq!((lo, hi), (12000, 12100));
    assert_eq!(
        plan_append(11999, 12100, lo, hi),
        Some(AppendPlan {
            first_key: 12000,
            rows: 99
        }),
        "the owner's own append must be unaffected"
    );
}

/// A cold prefill's own chunks still merge into one interval.
#[test]
fn consecutive_chunks_of_one_sequence_still_merge() {
    let r = stamped_merge(stamped_merge(StoreRange::EMPTY, 1, 0, 500), 1, 500, 500);
    assert_eq!(
        r,
        StoreRange {
            owner: 1,
            lo: 0,
            hi: 1000
        }
    );
}

/// ★ THE SENTINEL. `SequenceState`s built outside `alloc_sequence` (the
/// mock, and the test fakes) draw no ticket and carry 0. Zero must never
/// match — treating it as a wildcard would reopen the hole for exactly
/// those constructors, which is where an unstamped read would land.
#[test]
fn generation_zero_never_matches_anything() {
    let claimed = StoreRange {
        owner: 7,
        lo: 0,
        hi: 9999,
    };
    assert_eq!(
        claimed.visible_to(0),
        (0, 0),
        "a ticketless reader sees nothing"
    );
    let unclaimed = StoreRange {
        owner: 0,
        lo: 0,
        hi: 9999,
    };
    assert_eq!(unclaimed.visible_to(0), (0, 0), "0 does not match 0");
    assert_eq!(
        unclaimed.visible_to(7),
        (0, 0),
        "and nobody owns an unclaimed range"
    );
    // A writer with no ticket still stamps 0, so its rows stay invisible
    // rather than becoming readable by everyone.
    assert_eq!(stamped_merge(StoreRange::EMPTY, 0, 0, 10).owner, 0);
}

/// The two refusals must not read alike in the debug log, for the same
/// reason the session refusal must not read like a prefix mismatch.
#[test]
fn foreign_hiddens_does_not_read_like_a_coverage_miss() {
    let foreign = CarryOutcome::ForeignHiddens {
        owner: 2,
        expected: 1,
    }
    .to_string();
    assert!(foreign.contains("belong to sequence gen 2"), "{foreign}");
    assert_ne!(foreign, CarryOutcome::NoHiddens.to_string());
}

/// The gate's truth table, stated once, since two call sites read it.
#[test]
fn session_matches_is_equality_and_refuses_zero() {
    let c = carried(&[1, 2, 3], 2, Some(1));
    assert!(c.session_matches(SESSION));
    assert!(!c.session_matches(OTHER));
    assert!(!c.session_matches(0), "zero is unknown, not wildcard");
}

/// A foreign-session refusal must not be reported as a prefix mismatch.
/// Reading one as the other is precisely how the channel stayed invisible:
/// `PrefixMismatch` is the expected, benign outcome of a re-tokenized turn
/// boundary, so it draws no attention in a log.
#[test]
fn the_two_refusals_do_not_read_alike() {
    let foreign = CarryOutcome::ForeignSession {
        entry_session: SESSION,
        prompt_session: OTHER,
    }
    .to_string();
    let mismatch = CarryOutcome::PrefixMismatch {
        common: 4,
        entry_rows: 9,
    }
    .to_string();
    assert!(foreign.contains("foreign session"), "{foreign}");
    assert!(!foreign.contains("prefix mismatch"), "{foreign}");
    assert!(mismatch.contains("prefix mismatch"), "{mismatch}");
    assert_ne!(foreign, mismatch);
}

#[test]
fn usable_by_keeps_everything_when_the_whole_entry_matches() {
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    // pair key 3 consumed tokens[0..=4]; all 5 match.
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), Some((4, 3)));
}

#[test]
fn usable_by_truncates_the_tail_instead_of_refusing() {
    // Divergence at index 4 => common = 4 => highest usable key is 2, so
    // one row is dropped. This is the chat-template re-tokenization case
    // that made full-match adoption refuse every warm turn.
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 9, 6, 7], SESSION), Some((3, 2)));
    // Divergence at index 2 => common = 2 => only key 0 survives.
    assert_eq!(c.usable_by(&[1, 2, 9, 9], SESSION), Some((1, 0)));
}

#[test]
fn usable_by_declines_when_nothing_survives() {
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    // Fewer than 2 tokens in common: not even pair key 0 is usable.
    assert_eq!(c.usable_by(&[1, 9, 9], SESSION), None);
    assert_eq!(c.usable_by(&[], SESSION), None);
    // No rows, or no tracked key.
    assert_eq!(
        carried(&[1, 2, 3], 0, Some(1)).usable_by(&[1, 2, 3, 4], SESSION),
        None
    );
    assert_eq!(
        carried(&[1, 2, 3], 2, None).usable_by(&[1, 2, 3, 4], SESSION),
        None
    );
}

#[test]
fn usable_by_never_drops_more_rows_than_exist() {
    // A compacted entry: 2 rows but a far-ahead key. Truncating to a low
    // common prefix must decline rather than underflow.
    let c = carried(&[1, 2, 3, 4, 5, 6], 2, Some(4));
    assert_eq!(c.usable_by(&[1, 2, 9], SESSION), None);
}

#[test]
fn append_plan_covers_exactly_the_missing_pair_keys() {
    // Drafter holds keys 0..=97; prompt has 200 tokens => keys 0..=198.
    // Hidden store covers [97, 200).
    let p = plan_append(97, 200, 97, 200).unwrap();
    assert_eq!(
        p,
        AppendPlan {
            first_key: 98,
            rows: 101
        }
    );
}

#[test]
fn append_plan_clamps_up_to_the_hidden_store_floor() {
    // Store only reaches back to 150, so keys 98..149 are unreachable.
    // Skipping them is safe: rows are compacted and RoPE carries position.
    let p = plan_append(97, 200, 150, 200).unwrap();
    assert_eq!(
        p,
        AppendPlan {
            first_key: 150,
            rows: 49
        }
    );
}

#[test]
fn append_plan_declines_when_the_store_stops_short_of_the_last_key() {
    // Needs hidden row 198; store ends at 190 (exclusive).
    assert_eq!(plan_append(97, 200, 97, 190), None);
    // Exclusive end 198 still omits row 198; this is the adjacent boundary.
    assert_eq!(plan_append(97, 200, 97, 198), None);
}

#[test]
fn append_plan_declines_when_nothing_is_missing() {
    assert_eq!(plan_append(198, 200, 0, 200), None);
    assert_eq!(plan_append(250, 200, 0, 200), None);
}

#[test]
fn append_plan_declines_on_a_degenerate_prompt() {
    assert_eq!(plan_append(0, 1, 0, 8), None);
    assert_eq!(plan_append(0, 0, 0, 8), None);
}

#[test]
fn merge_interval_extends_on_overlap_and_abut() {
    assert_eq!(merge_interval((10, 20), 20, 5), (10, 25)); // abut
    assert_eq!(merge_interval((10, 20), 15, 10), (10, 25)); // overlap
    assert_eq!(merge_interval((10, 20), 5, 6), (5, 20)); // overlap below
}

#[test]
fn merge_interval_replaces_on_a_gap() {
    // A disjoint write must NOT claim the gap: rows in it were never
    // written for this sequence.
    assert_eq!(merge_interval((10, 20), 30, 5), (30, 35));
    assert_eq!(merge_interval((10, 20), 0, 5), (0, 5));
    assert_eq!(merge_interval((0, 0), 30, 5), (30, 35));
}

#[test]
fn common_prefix_len_is_the_validity_primitive() {
    let c = carried(&[1, 2, 3, 4], 3, Some(2));
    assert_eq!(c.common_prefix_len(&[1, 2, 3, 4, 5]), 4);
    assert_eq!(c.common_prefix_len(&[1, 2, 9, 4, 5]), 2);
    assert_eq!(c.common_prefix_len(&[]), 0);
}
