// SPDX-License-Identifier: AGPL-3.0-only

//! Spec-in-think parity on the commit side (`emit_step::emit_token`): the
//! thinking-state transitions `process_decode_logits` performs per committed
//! token must happen identically here (see `think_commit`).

use super::emit_step::emit_token;
use super::sched_ctx::SchedCtx;
use super::test_support::{EOS, test_seq};
use super::think_commit::SpecThinkTrail;
use super::types::ActiveSeq;

const FENCE: u32 = 4000;
const THINK_END: u32 = 4001;
const THINK_START: u32 = 4002;

fn thinking_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq((1000..1007).collect(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a
}

fn sched() -> SchedCtx {
    let mut s = SchedCtx::for_test();
    s.limits.code_fence_token = Some(FENCE);
    s
}

#[test]
fn emit_toggles_code_fence_and_resets_it_at_think_end() {
    let s = sched();
    let mut a = thinking_seq();
    emit_token(&mut a, FENCE, None, &s);
    assert!(
        a.in_code_fence,
        "``` inside <think> opens a fence (decode twin)"
    );
    emit_token(&mut a, 2000, None, &s);
    assert!(a.in_code_fence);
    a.consecutive_confident = 9;
    emit_token(&mut a, THINK_END, None, &s);
    assert!(!a.in_code_fence && a.consecutive_confident == 0);
}

#[test]
fn emit_runs_the_think_loop_watchdog() {
    let s = sched();
    let mut a = thinking_seq();
    for i in 0..200u32 {
        emit_token(&mut a, 3000 + (i % 4), None, &s);
        if a.force_end_thinking {
            break;
        }
    }
    assert!(
        a.force_end_thinking,
        "a period-4 reasoning loop arms </think>"
    );
    assert_eq!(a.think_watchdog_fires, 1);
    assert!(a.thinking_tokens < 200 && a.thinking_budget.is_none());
}

#[test]
fn emit_discards_a_suppressed_eos_inside_think_from_history() {
    let s = sched();
    let mut a = thinking_seq();
    let before = a.output_tokens.clone();
    emit_token(&mut a, EOS[0], None, &s);
    assert!(!a.finished);
    assert_eq!(
        a.output_tokens, before,
        "decode never keeps a suppressed EOS"
    );
    assert_eq!(
        a.thinking_tokens, 1,
        "…but it still counts as a thinking token"
    );
}

#[test]
fn emit_clears_require_tool_call_after_512_tokens() {
    let s = sched();
    let mut a = thinking_seq();
    a.require_tool_call = true;
    a.output_tokens = (0..513).collect();
    emit_token(&mut a, 2000, None, &s);
    assert!(!a.require_tool_call);
}

#[test]
fn emit_spontaneous_think_budget_decays_like_decode() {
    let s = sched();
    let mut a = thinking_seq();
    a.inside_thinking = false;
    a.think_ended = true;
    a.spontaneous_think_budget = 1024;
    a.think_watchdog_fires = 2;
    emit_token(&mut a, THINK_START, None, &s);
    assert!(a.inside_thinking);
    assert_eq!(a.thinking_budget, Some(256));
}

#[test]
fn emit_applies_only_a_matching_trail_entry() {
    let s = sched();
    let mut a = thinking_seq();
    let len = a.output_tokens.len();
    let entry = |tok, out_len| SpecThinkTrail {
        tok,
        out_len,
        consecutive_confident: 7,
        sentence_defer_count: 3,
        force_end_thinking: false,
    };
    a.spec_think_trail.push_back(entry(2000, len));
    a.spec_think_trail.push_back(entry(2001, len + 1));
    emit_token(&mut a, 2000, None, &s);
    assert_eq!((a.consecutive_confident, a.sentence_defer_count), (7, 3));
    assert_eq!(a.spec_think_trail.len(), 1);
    // Rejected tail: a different token commits → stale trail dropped.
    a.consecutive_confident = 0;
    emit_token(&mut a, 2555, None, &s);
    assert_eq!(a.consecutive_confident, 0);
    assert!(a.spec_think_trail.is_empty());
}

#[test]
fn thinking_rows_never_take_the_raw_argmax_verdict() {
    // GPU class (A)+(B) under DFlash (`dflash_verify_raw_argmax`): the raw
    // verdict skips the whole pipeline (mid-word `</think>` mask, forced
    // close, A4 floor) AND uses the GPU argmax's tie order, while spec-off
    // decodes every thinking row on the host pipeline (last-wins ties).
    let mut a = thinking_seq();
    a.logit_bias.clear();
    assert!(super::sample_step::speculative_raw_argmax_forbidden(&a));
    a.inside_thinking = false;
    a.think_ended = true;
    a.grammar_state = None;
    assert!(!super::sample_step::speculative_raw_argmax_forbidden(&a));
}
