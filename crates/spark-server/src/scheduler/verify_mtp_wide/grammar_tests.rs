// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::test_support::test_seq;

fn context(sched: &SchedCtx) -> LogitsContext<'_> {
    LogitsContext {
        scratch: &sched.scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: crate::scheduler::logit_processors::SamplingLevers::default(),
        timing: sched.timing.clone(),
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    }
}

fn json_state() -> GrammarState {
    let vocab = ["{", "}", "x", "<eos>", "</think>"].map(String::from);
    let mut engine = GrammarEngine::new(&vocab, &[3]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    GrammarState::new(&compiled, vocab.len())
        .unwrap()
        .with_stop_tokens(&[3])
}

fn logits(favorites: &[usize]) -> Vec<u8> {
    favorites
        .iter()
        .flat_map(|&favorite| {
            (0..5).flat_map(move |token| {
                let value: f32 = if token == favorite { 20.0 } else { 0.0 };
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
        })
        .collect()
}

#[test]
fn fresh_grammar_masks_malformed_suffix_and_keeps_matcher_attached() {
    let sched = SchedCtx::for_test();
    let ctx = context(&sched);
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.finished = false;
    seq.min_tokens = 0;
    seq.eos_tokens = vec![3];
    seq.grammar_state = Some(json_state());
    // After '{', an unmasked 'x' draft is malformed. The target must mask it
    // to '}', reject it, and leave grammar ready for EOS without sampling row2.
    let picks = sample_and_emit(&logits(&[0, 2, 2]), 5, &[0, 2], &mut seq, &sched, &ctx);
    assert_eq!(picks, [0, 1]);
    let gs = seq
        .grammar_state
        .as_mut()
        .expect("valid grammar must remain active");
    assert_eq!(gs.num_history_steps(), 2);
    assert!(gs.stop_legal(&[3]));
}

#[test]
fn valid_grammar_transition_ends_at_eos_without_rejected_suffix() {
    let sched = SchedCtx::for_test();
    let ctx = context(&sched);
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.finished = false;
    seq.min_tokens = 0;
    seq.eos_tokens = vec![3];
    seq.grammar_state = Some(json_state());
    let picks = sample_and_emit(
        &logits(&[0, 1, 3, 2]),
        5,
        &[0, 1, 3],
        &mut seq,
        &sched,
        &ctx,
    );
    assert_eq!(picks, [0, 1, 3]);
    assert!(seq.finished);
    assert!(seq.grammar_state.is_some());
}

#[test]
fn thinking_close_resumes_grammar_on_next_verification_row() {
    let sched = SchedCtx::for_test();
    let mut ctx = context(&sched);
    ctx.think_end_token = Some(4);
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.finished = false;
    seq.min_tokens = 0;
    seq.eos_tokens = vec![3];
    seq.grammar_state = Some(json_state());
    seq.inside_thinking = true;
    seq.think_end_token = Some(4);
    seq.thinking_tokens = 20;
    let picks = sample_and_emit(
        &logits(&[4, 2, 1, 3]),
        5,
        &[4, 0, 1],
        &mut seq,
        &sched,
        &ctx,
    );
    assert_eq!(picks, [4, 0, 1, 3]);
    assert!(!seq.inside_thinking);
    assert!(seq.finished);
    assert!(seq.grammar_state.is_some());
}

#[test]
fn length_finish_does_not_advance_grammar_past_emission() {
    let sched = SchedCtx::for_test();
    let ctx = context(&sched);
    let (mut seq, _rx) = test_seq(Vec::new(), 1, None, 10);
    seq.finished = false;
    seq.grammar_state = Some(json_state());
    let picks = sample_and_emit(&logits(&[0, 1, 3]), 5, &[0, 1], &mut seq, &sched, &ctx);
    assert_eq!(picks, [0]);
    assert!(seq.finished);
    let gs = seq.grammar_state.as_mut().unwrap();
    assert_eq!(gs.num_history_steps(), 1);
    assert!(!gs.stop_legal(&[3]));
}

#[test]
fn only_qwen_proposer_state_allows_wider_grammar_proposals() {
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.grammar_state = Some(json_state());
    assert_eq!(grammar::drafts(&seq, 3, false), 1);
    seq.seq.proposer_state = Some(Box::new(
        spark_model::layers::qwen4_exp_mtp_proposer::Qwen4ExpMtpProposerState {
            inner: spark_model::layers::qwen4_exp_mtp::Qwen4ExpMtpState {
                block_table: Vec::new(),
                seq_len: 0,
                body_state: Box::new(spark_model::layer::EmptyLayerState),
                pending_draft: None,
                last_num_drafted: 0,
                pending_rewind: 0,
                pre_draft_aux: None,
            },
        },
    ));
    assert_eq!(grammar::drafts(&seq, 3, false), 3);
    assert_eq!(grammar::drafts(&seq, 3, true), 1);
    // The dispatcher passes one when concurrency grows after proposal.
    assert_eq!(grammar::drafts(&seq, 1, false), 1);
}
