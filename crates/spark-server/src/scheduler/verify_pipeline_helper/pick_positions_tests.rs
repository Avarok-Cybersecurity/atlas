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
use super::verify_pick_all_with_pipeline;
use crate::grammar::tests::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::logit_processors::{LogitsContext, SamplingLevers};
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;

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
        // Inert here: this file's `pick_positions_from_host` never reads
        // `verify_pos`. The real verify path rebuilds the context per
        // position (`LogitsContext { verify_pos, ..ctx.clone() }` in
        // verify_pipeline_helper.rs), so the value carried by a shared
        // context cannot reach a position's mask. 0 matches every other
        // non-verify construction site.
        verify_pos: 0,
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

// ── Post-think structural guard on the FAST paths (A143 sibling) ─────────
//
// `verify_pick_all_with_pipeline`'s two GPU-argmax-only fast-greedy blocks
// (grammar and grammarless) used to return `argmax_ids` straight through
// with no post-think check at all — unlike the slow path exercised above,
// which always runs `PostCloseThinkMask` via `pick_positions_from_host`.
// These tests drive the real `verify_pick_all_with_pipeline` entry point
// (not just `pick_positions_from_host`) so a regression in the NEW guard's
// wiring — not just the guard function itself — fails a test.

/// A sequence past `</think>` (`think_ended`), no grammar armed — the
/// grammarless fast-greedy arm's eligibility regime.
fn post_think_grammarless_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = false;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.think_ended = true;
    a.grammar_state = None;
    a
}

/// Like [`with_ctx`], but arms `fast_greedy_chat` — the grammarless
/// fast-greedy arm's kill switch — so the arm is actually eligible absent
/// the post-think guard, making this a real test of the guard rather than
/// a vacuous one.
fn with_ctx_fast_greedy_chat<R>(f: impl FnOnce(&LogitsContext) -> R) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let dumps = crate::scheduler::dumps::RunDumps::default();
    let ctx = LogitsContext {
        scratch: &scratch,
        dumps: &dumps,
        stats: std::sync::Arc::new(crate::scheduler::spec_stats::SpecStats::new()),
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers {
            fast_greedy_chat: true,
            ..SamplingLevers::default()
        },
        timing: std::sync::Arc::default(),
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL_OPEN),
        tool_call_end_token: Some(TOOL_CALL_CLOSE),
        verify_pos: 0,
    };
    f(&ctx)
}

/// Minimal `Model`: only `vocab_size` / `logits_buffer_ptr` /
/// `copy_logits_to_host` are functional (the slow-path D2H the guard must
/// route to when it fires); everything else is unreachable because a
/// grammarless, temp-0 verify call with the guard's precondition never
/// reaches it. Same "functional slice + `unreachable!()` rest" pattern as
/// `PrefillStubModel` (`prefill_fifo_tests.rs`) and `PreemptStubModel`
/// (`test_support.rs`).
struct FastPathStubModel {
    vocab: usize,
    /// `[K, vocab]` BF16 bytes, same layout `bf16_rows` produces.
    buf: Vec<u8>,
}

impl FastPathStubModel {
    fn new(vocab: usize, rows: &[Vec<f32>]) -> Self {
        Self {
            vocab,
            buf: bf16_rows(rows),
        }
    }
}

impl Model for FastPathStubModel {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let off = logits_ptr.0 as usize;
        dst.copy_from_slice(&self.buf[off..off + dst.len()]);
        Ok(())
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no prefill in this harness")
    }
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _chunk_start: usize,
        _chunk_len: usize,
        _is_last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("no prefill in this harness")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        unreachable!("no MTP in this harness")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _mask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        unreachable!("no MTP in this harness")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        unreachable!("no MTP in this harness")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no decode in this harness")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("no decode in this harness")
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        unreachable!("no speculation in this harness")
    }
    fn argmax_on_device(&self, _logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        unreachable!("no on-device argmax in this harness")
    }
    fn argmax_batch(&self, _logits_ptr: DevicePtr, _n: usize, _stream: u64) -> Result<Vec<u32>> {
        unreachable!("not the decode fast path")
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        unreachable!("no MTP in this harness")
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        unreachable!("no SSM in this harness")
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        unreachable!("no SSM in this harness")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _n: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        unreachable!("no speculation in this harness")
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _new_slot: usize) -> Result<()> {
        unreachable!("no compaction in this harness")
    }
    fn detach_slot_for_reuse(&self, _seq: &mut SequenceState) {}
    fn save_hidden_for_mtp(&self, _token_idx: usize, _st: u64) -> Result<()> {
        unreachable!("no MTP in this harness")
    }
}

#[test]
fn verify_fast_path_bails_on_reopened_think_end_after_think_ended() {
    // The GPU-graphed argmax for this (single-position) verify window
    // re-opens `</think>` — a live token in the vocabulary right up until
    // `PostCloseThinkMask` masks it. `HELLO` is the runner-up. With
    // `think_ended` true and `fast_greedy_chat` armed, the grammarless
    // fast-greedy arm would (absent the new guard) return `THINK_END`
    // straight through with no D2H at all.
    let mut a = post_think_grammarless_seq();
    let model = FastPathStubModel::new(VOCAB, &[row(&[(THINK_END, 10.0), (HELLO, 5.0)])]);
    let argmax_ids = [THINK_END];
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &argmax_ids, &mut a, ctx, 0)
    });
    assert_eq!(
        picks,
        vec![HELLO],
        "post-think structural guard must force the slow path so \
         PostCloseThinkMask masks the reopened </think> and the runner-up \
         wins, instead of the fast path returning the raw GPU argmax"
    );
}

#[test]
fn verify_fast_path_takes_the_fast_path_when_no_structural_hit() {
    // Control: the raw argmax is an ordinary content token, not a
    // structural id — the guard must NOT force the slow path (which would
    // still produce the same pick here, but this proves the fast arm is
    // actually reachable in this fixture, so the guarded test above is not
    // vacuously passing via some unrelated ineligibility).
    let mut a = post_think_grammarless_seq();
    let model = FastPathStubModel::new(VOCAB, &[]);
    let argmax_ids = [HELLO];
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &argmax_ids, &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
}

// ── A144: speculative paths apply the SAME `logit_bias` as decode ─────────
//
// Server-side, a tools-active request carries `(<tool_call>, +3.0)` in
// `ActiveSeq.logit_bias` (`sampling_setup.rs`). Decode applied it; verify
// passed an EMPTY bias, so spec-on picked the raw prose token where spec-off
// opened a call (GPU probe: spec-off + client bias -3 reproduced K3 4/4).

/// Post-think, grammarless, greedy, tools-active: `repetition_penalty` 1.05
/// keeps decode on the HOST pipeline for this row (the `think_ended` GPU
/// admission needs exactly-neutral penalties), so decode APPLIES the bias.
fn tools_present_seq() -> ActiveSeq {
    let mut a = post_think_grammarless_seq();
    a.min_tokens = 0;
    a.repetition_penalty = 1.05;
    a.tool_call_start_token = Some(TOOL_CALL_OPEN);
    a.tool_call_end_token = Some(TOOL_CALL_CLOSE);
    a.logit_bias = vec![(TOOL_CALL_OPEN, 3.0)];
    a
}

/// `<tool_call>` 2.0 below the prose argmax: +3.0 flips it, 0.0 does not.
fn opener_near_miss_row() -> Vec<f32> {
    row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 8.0)])
}

#[test]
fn a144_verify_applies_the_same_bias_as_decode_at_a_tools_present_position() {
    use crate::scheduler::sample_step::{
        PositionKind, penalty_params_for, speculative_base_logit_bias,
    };
    let mut a = tools_present_seq();
    // Param-level parity: the Verify params carry exactly decode's bias.
    let decode = penalty_params_for(
        &a,
        PositionKind::FinalDecode,
        0.0,
        None,
        a.logit_bias.clone(),
    );
    let verify_bias = speculative_base_logit_bias(&a, 0, Some(THINK_END), || {
        unreachable!("host-regime row never probes the raw argmax")
    });
    let verify = penalty_params_for(&a, PositionKind::Verify, 0.0, None, verify_bias);
    assert_eq!(verify.logit_bias, decode.logit_bias);
    assert_eq!(verify.logit_bias, vec![(TOOL_CALL_OPEN, 3.0)]);

    // End to end through the slow path (fast paths off in `with_ctx`).
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx(|ctx| verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN],
        "verify must pick what decode picks: HELLO 10.0 < <tool_call> 8.0 + 3.0"
    );
}

#[test]
fn a144_opener_bias_is_stripped_per_position_inside_a_tool_body() {
    // One window opens a call, stays in its body, closes it, then sits at a
    // fresh opener decision. The +3.0 must be OFF at position 1 (inside the
    // body opened by position 0 — else a spurious mid-body re-open) and ON
    // again at position 3 (after position 2's `</tool_call>`). The step-start
    // state (outside a body) is wrong for positions 1 and 2.
    let mut a = tools_present_seq();
    let buf = bf16_rows(&[
        row(&[(TOOL_CALL_OPEN, 10.0)]),
        opener_near_miss_row(),
        row(&[(TOOL_CALL_CLOSE, 10.0)]),
        opener_near_miss_row(),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 4, &mut a, ctx));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN, HELLO, TOOL_CALL_CLOSE, TOOL_CALL_OPEN]
    );
    assert!(
        !a.inside_tool_body,
        "tool-body flag restored after the loop"
    );

    // Starting INSIDE a body: position 0 is stripped; after the close the
    // nudge returns.
    let mut a = tools_present_seq();
    a.inside_tool_body = true;
    let buf = bf16_rows(&[
        opener_near_miss_row(),
        row(&[(TOOL_CALL_CLOSE, 10.0)]),
        opener_near_miss_row(),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 3, &mut a, ctx));
    assert_eq!(picks, vec![HELLO, TOOL_CALL_CLOSE, TOOL_CALL_OPEN]);
    assert!(a.inside_tool_body, "tool-body flag restored after the loop");
}

#[test]
fn a144_fast_greedy_falls_back_to_host_when_bias_present() {
    // `fast_greedy_chat` armed and the penalties reduce-only: absent the
    // A144 guard the grammarless fast arm returns the raw GPU argmax (HELLO)
    // with no D2H, never seeing the bias.
    let mut a = tools_present_seq();
    assert!(crate::scheduler::sample_step::speculative_bias_forces_host(
        &a
    ));
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![TOOL_CALL_OPEN]);

    // Control: without a bias the same fixture takes the fast arm.
    let mut a = tools_present_seq();
    a.logit_bias.clear();
    assert!(!crate::scheduler::sample_step::speculative_bias_forces_host(&a));
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
}

#[test]
fn a144_bias_skipped_exactly_where_decode_gpu_argmax_skips_it() {
    use crate::scheduler::sample_step::{
        speculative_base_logit_bias, speculative_bias_forces_host,
    };
    // Neutral penalties + think_ended + greedy + no grammar: decode admits
    // the row to its GPU argmax and never applies `logit_bias`. Parity with
    // decode means verify must not apply it either.
    let mut a = tools_present_seq();
    a.repetition_penalty = 1.0;
    assert!(!speculative_bias_forces_host(&a));
    assert!(speculative_base_logit_bias(&a, 0, Some(THINK_END), || HELLO).is_empty());
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx(|ctx| verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0));
    assert_eq!(
        picks,
        vec![HELLO],
        "decode's GPU argmax emits HELLO; so must verify"
    );

    // ...except when that GPU argmax lands on a post-think `</think>`/`<think>`:
    // decode then redoes the step on the host, bias included.
    assert_eq!(
        speculative_base_logit_bias(&a, 0, Some(THINK_END), || THINK_END),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );
    assert_eq!(
        speculative_base_logit_bias(&a, 0, Some(THINK_END), || THINK_START),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );

    // A `min_tokens` floor keeps decode on the host until it is met; the
    // floor is judged at `output_len + verify_pos`.
    a.min_tokens = 2;
    assert_eq!(
        speculative_base_logit_bias(&a, 1, Some(THINK_END), || HELLO),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );
    assert!(speculative_base_logit_bias(&a, 2, Some(THINK_END), || HELLO).is_empty());

    // Temperature > 0 always runs decode's host sampler.
    a.min_tokens = 0;
    a.temperature = 0.7;
    assert!(speculative_bias_forces_host(&a));
}

// ── A144b (2026-09-25): verify's final pick must use decode's tie-break ──
//
// decode's host greedy path (`sample_impl::greedy_pick_last_wins`) and this
// slow path both process the identical dequantised (BF16→F32, no extra
// rounding either side) logits, so an exact tie on a quantised checkpoint is
// real and common. Before this fix the slow path's final argmax
// (`argmax::argmax_first_wins`) resolved ties to the FIRST equal-valued id;
// decode always resolves to the LAST. That mismatch — not a masking or
// precision bug — is what produced the K3-vs-spec-off synonym-swap
// divergence (54/60 divergent TEB transcripts at temperature 0).

#[test]
fn a144b_verify_exact_tie_matches_decodes_last_wins_tie_break() {
    // HELLO (104) and TOOL_CALL_OPEN (128) tied at the row max, 9.0 — exactly
    // bf16-representable, so the D2H round-trip introduces no rounding that
    // could break the tie by accident.
    let mut a = post_think_grammarless_seq();
    a.min_tokens = 0;
    let buf = bf16_rows(&[row(&[(HELLO, 9.0), (TOOL_CALL_OPEN, 9.0)])]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 1, &mut a, ctx));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN],
        "TOOL_CALL_OPEN (id 128) is the LAST of the two tied ids (104, 128) — \
         decode's `greedy_pick_last_wins` must win here, not first-wins' HELLO"
    );
}
