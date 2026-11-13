// SPDX-License-Identifier: AGPL-3.0-only

//! Port-sprint additions to the admission tests (master plan L05).
//!
//! 1. The long-context regression vectors of the ExLlamaV3 token-budget fix
//!    (117,760-token cache, 91,400-token prompt, 65,536 tokens requested; the
//!    stock server there rejected the job because prompt + requested did not
//!    fit the whole cache). Atlas admits by `prompt + min(max_tokens,
//!    watermark)` blocks (see `admission.rs`); these pin what that policy does
//!    with exactly those numbers, boundary included, so a later change to the
//!    gate shows up here. The vectors are numbers only; no code is taken from
//!    that project.
//! 2. The per-request no-drafts switch: `atlas_speculative: false` becomes
//!    `seq.port.no_drafts` and the MTP dispatch predicate refuses a sequence
//!    that carries it.

use super::admission::{
    SeqDemand, admit_count, blocks_for_tokens, committed_blocks, seq_commitment_blocks,
};
use super::mtp_gate::spec_dispatch_eligible;
use super::port_state::SeqPortState;

const BS: usize = 16;
/// The cache of the regression report, in tokens.
const CACHE_TOKENS: usize = 117_760;
/// 117,760 / 16.
const TOTAL_BLOCKS: usize = 7_360;
const PROMPT: usize = 91_400;
const REQUESTED: usize = 65_536;
/// The honest bound: a request can never generate past the served ceiling.
const NO_WATERMARK: usize = usize::MAX;

#[test]
fn the_cache_is_the_block_count_the_vectors_assume() {
    assert_eq!(CACHE_TOKENS / BS, TOTAL_BLOCKS);
    assert_eq!(CACHE_TOKENS % BS, 0);
}

#[test]
fn a_request_that_cannot_fit_the_whole_cache_is_clamped_to_the_ceiling() {
    let d = SeqDemand {
        current_tokens: PROMPT,
        budget_tokens: REQUESTED,
    };
    // Served ceiling = the cache: the reservation stops at it (7,360 full
    // blocks plus the slack block of the `tokens / block + 1` formula).
    assert_eq!(
        seq_commitment_blocks(&d, NO_WATERMARK, CACHE_TOKENS, BS),
        blocks_for_tokens(CACHE_TOKENS, BS)
    );
    assert_eq!(blocks_for_tokens(CACHE_TOKENS, BS), 7_361);
    // No ceiling: the reservation is the full 156,936 tokens.
    assert_eq!(seq_commitment_blocks(&d, NO_WATERMARK, 0, BS), 9_809);
}

#[test]
fn the_oversize_request_alone_is_admitted_with_the_forced_flag() {
    // 7,361 blocks against 7,360: it does not fit, but nothing else is in
    // flight, so liveness admits it (the allocator back-pressures at runtime)
    // and flags it for the warning.
    let reqs = [(PROMPT, REQUESTED)];
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 0, &reqs, NO_WATERMARK, CACHE_TOKENS, BS),
        (1, true)
    );
}

#[test]
fn the_same_request_waits_behind_anything_in_flight() {
    let reqs = [(PROMPT, REQUESTED)];
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 1, &reqs, NO_WATERMARK, CACHE_TOKENS, BS),
        (0, false)
    );
}

#[test]
fn a_request_that_fits_the_cache_is_admitted_normally() {
    // 91,400 + 20,000 = 111,400 tokens -> 6,963 blocks <= 7,360.
    let reqs = [(PROMPT, 20_000)];
    assert_eq!(blocks_for_tokens(PROMPT + 20_000, BS), 6_963);
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 0, &reqs, NO_WATERMARK, CACHE_TOKENS, BS),
        (1, false)
    );
}

#[test]
fn the_boundary_is_one_token_wide() {
    // 117,759 tokens -> 7,359 + 1 = 7,360 blocks: fits exactly.
    let fits = [(PROMPT, CACHE_TOKENS - 1 - PROMPT)];
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 0, &fits, NO_WATERMARK, 0, BS),
        (1, false)
    );
    // One more token crosses into a 7,361st block: forced.
    let over = [(PROMPT, CACHE_TOKENS - PROMPT)];
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 0, &over, NO_WATERMARK, 0, BS),
        (1, true)
    );
}

#[test]
fn a_second_long_request_queues_until_the_first_finishes() {
    // One request already holds 6,963 blocks; an identical one cannot join.
    let in_flight = [SeqDemand {
        current_tokens: PROMPT,
        budget_tokens: 20_000,
    }];
    let committed = committed_blocks(&in_flight, NO_WATERMARK, CACHE_TOKENS, BS);
    assert_eq!(committed, 6_963);
    let reqs = [(PROMPT, 20_000)];
    assert_eq!(
        admit_count(
            TOTAL_BLOCKS,
            committed,
            &reqs,
            NO_WATERMARK,
            CACHE_TOKENS,
            BS
        ),
        (0, false)
    );
    // Once it is gone, the same request is admitted.
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 0, &reqs, NO_WATERMARK, CACHE_TOKENS, BS),
        (1, false)
    );
}

#[test]
fn watermark_zero_reserves_the_prompt_only() {
    // The legacy escape hatch: 91,400 tokens -> 5,713 blocks, so the 65,536
    // requested tokens no longer count against admission.
    let reqs = [(PROMPT, REQUESTED)];
    assert_eq!(blocks_for_tokens(PROMPT, BS), 5_713);
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 0, &reqs, 0, CACHE_TOKENS, BS),
        (1, false)
    );
}

#[test]
fn a_prompt_larger_than_the_cache_is_forced_when_idle_and_queued_otherwise() {
    // 120,000-token prompt: 7,501 blocks > 7,360 whatever the budget.
    let reqs = [(120_000, 1)];
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 0, &reqs, NO_WATERMARK, 0, BS),
        (1, true)
    );
    assert_eq!(
        admit_count(TOTAL_BLOCKS, 10, &reqs, NO_WATERMARK, 0, BS),
        (0, false)
    );
}

#[test]
fn a_request_without_the_switch_keeps_drafting_enabled() {
    assert!(!SeqPortState::default().no_drafts);
    assert!(!SeqPortState::for_request(false).no_drafts);
}

#[test]
fn atlas_speculative_false_reaches_the_sequence_state() {
    assert!(SeqPortState::for_request(true).no_drafts);
}

#[test]
fn a_no_drafts_sequence_is_refused_by_the_speculative_dispatch_predicate() {
    // (inside_thinking, post_think_emitted, output_len, suppress_tool_call,
    //  disable_mtp, spec_think, resume_guard, dflash_raw_argmax)
    assert!(spec_dispatch_eligible(
        false, 100, 100, false, false, false, 0, false
    ));
    // The handler sets `disable_mtp` from `atlas_speculative: false`; the gate
    // then never lets the sequence into a speculative step.
    assert!(!spec_dispatch_eligible(
        false, 100, 100, false, true, false, 0, false
    ));
}
