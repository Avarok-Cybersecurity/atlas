// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::logit_processors::SamplingLevers;
use crate::scheduler::test_support::test_seq;

fn rows(values: [[f32; 4]; 2]) -> Rows {
    Rows {
        bytes: values
            .into_iter()
            .flatten()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        vocab: 4,
    }
}

fn context(sched: &SchedCtx) -> LogitsContext<'_> {
    LogitsContext {
        scratch: &sched.scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers {
            mtp_verify_sample: true,
            mtp_minp: true,
            ..SamplingLevers::default()
        },
        timing: sched.timing.clone(),
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    }
}

#[test]
fn greedy_verify_uses_serial_tie_breaking() {
    let sched = SchedCtx::for_test();
    let ctx = context(&sched);
    let rows = rows([[0.0, 8.0, 8.0, 0.0], [0.0, 0.0, 8.0, 8.0]]);
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.finished = false;
    let first = rows.pick(0, &mut seq, &ctx);
    assert_eq!(first.0, 2);
    assert_eq!(rows.emit_accepted(first, &mut seq, &sched, &ctx), Some(3));
    assert_eq!(seq.output_tokens, [2, 3]);
}

#[test]
fn accepted_prefix_updates_presence_penalty_and_logprobs() {
    let sched = SchedCtx::for_test();
    let ctx = context(&sched);
    let rows = rows([[0.0, 8.0, 0.0, 0.0], [0.0, 3.0, 2.0, 0.0]]);
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.finished = false;
    seq.presence_penalty = 1.5;
    seq.top_logprobs = Some(2);
    let first = rows.pick(0, &mut seq, &ctx);
    assert_eq!(first.0, 1);
    assert_eq!(rows.emit_accepted(first, &mut seq, &sched, &ctx), Some(2));
    assert_eq!(seq.output_tokens, [1, 2]);
    assert_eq!(seq.logprobs_data[1].token_id, 2);
    let expected =
        crate::scheduler::logprobs::extract_logprobs_from_f32(&[0.0, 1.5, 2.0, 0.0], 2, 2);
    assert_eq!(seq.logprobs_data[1].logprob, expected.logprob);
}

#[test]
fn thinking_close_activates_bonus_tool_pin() {
    let sched = SchedCtx::for_test();
    let mut ctx = context(&sched);
    ctx.think_end_token = Some(1);
    ctx.tool_call_start_token = Some(3);
    let rows = rows([[0.0, 20.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0]]);
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.finished = false;
    seq.inside_thinking = true;
    seq.think_end_token = Some(1);
    seq.tool_call_start_token = Some(3);
    seq.thinking_tokens = 20;
    seq.require_tool_call = true;
    let first = rows.pick(0, &mut seq, &ctx);
    assert_eq!(first.0, 1);
    assert_eq!(rows.emit_accepted(first, &mut seq, &sched, &ctx), Some(3));
    assert!(!seq.inside_thinking);
    assert!(seq.tool_call_opened);
}

#[test]
fn finished_first_token_never_processes_bonus_row() {
    let sched = SchedCtx::for_test();
    let ctx = context(&sched);
    let rows = rows([[0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0]]);
    let (mut seq, _rx) = test_seq(Vec::new(), 1, None, 10);
    seq.finished = false;
    seq.inside_thinking = true;
    seq.force_end_thinking = true;
    let first = rows.pick(0, &mut seq, &ctx);
    let after_first = seq.sentence_defer_count;
    assert_eq!(rows.emit_accepted(first, &mut seq, &sched, &ctx), None);
    assert_eq!(seq.sentence_defer_count, after_first);
    assert_eq!(seq.output_tokens.len(), 1);
}

#[test]
fn rejected_draft_does_not_advance_suffix_counters() {
    let sched = SchedCtx::for_test();
    let ctx = context(&sched);
    let rows = rows([[0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0]]);
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.finished = false;
    seq.inside_thinking = true;
    seq.force_end_thinking = true;
    let (token, lp) = rows.pick(0, &mut seq, &ctx);
    assert_ne!(token, 3, "the draft must be rejected");
    emit_token(&mut seq, token, lp, &sched);
    assert_eq!(seq.sentence_defer_count, 1);
    assert_eq!(seq.output_tokens, [1]);
}
