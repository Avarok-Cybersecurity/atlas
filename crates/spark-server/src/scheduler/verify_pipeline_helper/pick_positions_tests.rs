// SPDX-License-Identifier: AGPL-3.0-only

//! Reasoning-boundary tests for the K-position verify pick loop.
//!
//! A verify row may cross `</think>`: position i closes the reasoning span,
//! so position i+1 is the FIRST content token and must be masked from the
//! pristine grammar — exactly what a fresh decode step would do. The loop
//! used to read `inside_thinking` as it stood at step start, so every
//! position after the close was picked unmasked and the unconstrained draft
//! X was then fed to the matcher by `emit_token`, disengaging a `required`
//! grammar on its very first content token. These tests drive the real loop
//! over synthetic host logits and a real xgrammar matcher.

use super::pick_positions::pick_positions_from_host;
use crate::grammar::tests::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::logit_processors::{LogitsContext, SamplingLevers};
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;

const VOCAB: usize = 131;
const TOOL_CALL_OPEN: u32 = 128;
const TOOL_CALL_CLOSE: u32 = 129;
const EOS: u32 = 130;
/// The model's free (prose) pick — grammar-illegal as a first content token.
const HELLO: u32 = b'h' as u32;
/// In-vocab ids standing in for `</think>` / `<think>`. The matcher must
/// never be fed either, so any id the grammar would refuse is a fair probe.
const THINK_END: u32 = 127;
const THINK_START: u32 = 126;

fn required_tool_grammar() -> GrammarState {
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[EOS as i32]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS])
}

/// A sequence mid-reasoning with a `required` tool grammar armed.
fn thinking_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.tool_call_start_token = Some(TOOL_CALL_OPEN);
    a.grammar_state = Some(required_tool_grammar());
    a
}

/// One logits row: zeros except the named ids.
fn row(hot: &[(u32, f32)]) -> Vec<f32> {
    let mut r = vec![0.0f32; VOCAB];
    for &(id, v) in hot {
        r[id as usize] = v;
    }
    r
}

/// Rows → the little-endian BF16 `[K, vocab]` buffer the verify path D2H-copies.
fn bf16_rows(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flat_map(|r| {
            r.iter().flat_map(|&v| {
                let b = v.to_bits();
                [(b >> 16) as u8, (b >> 24) as u8]
            })
        })
        .collect()
}

fn with_ctx<R>(f: impl FnOnce(&LogitsContext) -> R) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let dumps = crate::scheduler::dumps::RunDumps::default();
    let ctx = LogitsContext {
        scratch: &scratch,
        dumps: &dumps,
        stats: std::sync::Arc::new(crate::scheduler::spec_stats::SpecStats::new()),
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers::default(),
        timing: std::sync::Arc::default(),
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL_OPEN),
        tool_call_end_token: Some(TOOL_CALL_CLOSE),
    };
    f(&ctx)
}

#[test]
fn verify_row_crossing_think_end_masks_the_first_post_think_position() {
    let mut a = thinking_seq();
    // Row 0: the model closes the span. Row 1: its free pick is prose, with
    // `<tool_call>` a distant second — the unmasked draft X the old loop let
    // through.
    let buf = bf16_rows(&[
        row(&[(THINK_END, 10.0)]),
        row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 5.0)]),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 2, &mut a, ctx));
    assert_eq!(picks[0], THINK_END, "position 0 closes the reasoning span");
    assert_eq!(
        picks[1], TOOL_CALL_OPEN,
        "the first post-think position must be picked under the pristine grammar, not free-run"
    );
    // The loop only PICKS; `emit_token` owns the real transition and advance.
    assert!(
        a.inside_thinking && !a.think_ended,
        "sequence state restored after the loop"
    );
    let gs = a
        .grammar_state
        .as_mut()
        .expect("grammar untouched by the loop");
    assert_eq!(
        gs.num_history_steps(),
        0,
        "</think> never fed; speculative advances rolled back"
    );
}

#[test]
fn verify_row_that_stays_inside_thinking_is_not_masked() {
    // Control: no `</think>` in the row → every position stays free-run and
    // the grammar stays paused (no over-constraint of reasoning tokens).
    let mut a = thinking_seq();
    let buf = bf16_rows(&[
        row(&[(HELLO, 10.0)]),
        row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 5.0)]),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 2, &mut a, ctx));
    assert_eq!(picks, vec![HELLO, HELLO]);
    assert!(a.inside_thinking);
    assert_eq!(a.grammar_state.as_ref().unwrap().num_history_steps(), 0);
}
