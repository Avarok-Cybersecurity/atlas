// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `lifecycle::derive_finish_reason` and `finish_sequence`.
//!
//! Split from `lifecycle.rs` for the ≤500-line cap. Two layers:
//!  * the pure precedence matrix over `derive_finish_reason`;
//!  * a call-site proof: `finish_sequence` on a REAL `ActiveSeq` with a
//!    blocking oneshot sink and a stub `Model`, asserting the
//!    `finish_reason` that actually reaches the response — so a wiring
//!    bug (wrong field passed for the budget/count) fails a test even
//!    though the pure function is correct.

use super::lifecycle::{derive_finish_reason, finish_sequence};
use super::types::{ActiveSeq, GUARD_STOP_REQUEST_TIMEOUT, ResponseSink};
use super::{DEFAULT_LZ_PENALTY, SsmDecodeRing};
use crate::api::InferenceResponse;
use crate::ir::FINISH_REASON_TIMEOUT;
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

const EOS: &[u32] = &[151645];
const TOOL_END: Option<u32> = Some(151658);
const MAX_SEQ_LEN: usize = 8192;

/// Common-case shorthand: mid-context position (no seqlen ceiling), so
/// the budget dimension under test is the `remaining` countdown.
fn derive(guard: Option<&'static str>, last: Option<u32>, remaining: usize) -> &'static str {
    derive_finish_reason(guard, last, EOS, TOOL_END, remaining, 10, MAX_SEQ_LEN)
}

#[test]
fn length_means_budget_exhausted_only() {
    // max_tokens countdown exhausted on an ordinary content token.
    assert_eq!(derive(None, Some(42), 0), "length");
    // Served context ceiling reached with budget left — the other half
    // of the hard-ceiling stop predicate. OpenAI reports "length" for
    // the context limit too.
    assert_eq!(
        derive_finish_reason(
            None,
            Some(42),
            EOS,
            TOOL_END,
            500,
            MAX_SEQ_LEN - 1,
            MAX_SEQ_LEN
        ),
        "length"
    );
}

#[test]
fn early_stop_one_token_short_of_budget_is_not_length() {
    // The bug this wave fixes: "length" was the catch-all, so ANY stop
    // whose last token was not EOS/tool-end claimed the budget was hit
    // (observed live: `Done: 573 tokens (length)` under
    // max_new_tokens=1024). One token of budget left ⇒ not "length".
    let r = derive(None, Some(42), 1);
    assert_ne!(r, "length");
    // Early finalize with no guard = client cancel / dropped receiver /
    // shutdown drain ⇒ "stop" (generation stopped; budget not hit).
    assert_eq!(r, "stop");
}

#[test]
fn normal_stops_are_unchanged() {
    assert_eq!(derive(None, Some(151645), 100), "stop");
    assert_eq!(derive(None, Some(151658), 100), "tool_calls");
}

#[test]
fn eos_on_the_last_budgeted_token_is_stop_not_length() {
    // The model finished naturally ON the final budgeted token: the
    // sampled EOS outranks the exhausted budget (OpenAI parity).
    assert_eq!(derive(None, Some(151645), 0), "stop");
}

#[test]
fn timeout_unchanged_and_still_outranks_everything() {
    // Shipped contract (2026-08 deadline wave): a deadline cut must be
    // distinguishable from both a natural stop and a max_tokens stop,
    // even when the deadline lands on an EOS / tool-close / budget-end
    // step.
    for last in [Some(42), Some(151645), Some(151658)] {
        assert_eq!(
            derive(Some(GUARD_STOP_REQUEST_TIMEOUT), last, 100),
            FINISH_REASON_TIMEOUT
        );
    }
    assert_eq!(
        derive(Some(GUARD_STOP_REQUEST_TIMEOUT), Some(42), 0),
        FINISH_REASON_TIMEOUT
    );
}

#[test]
fn guard_cuts_report_length_because_the_model_did_not_finish() {
    // POSITIVE case. A guard cut is a server-side truncation: the model
    // was still mid-output. `"length"` is the OpenAI-spec slot for
    // "forcibly truncated" and is what every client's truncation handling
    // keys on (openai-python `LengthFinishReasonError`, aider's
    // continuation, Instructor, pydantic-ai).
    //
    // ★ This assertion was briefly INVERTED to `"stop"`, and that shipped
    // a measured regression: the agentic gate fell to 8/10 then 4/10
    // followed_directions because its `was_cut_off()` stopped firing and
    // runs ended at 3-10 turns instead of the 12-22 a recovery needs.
    // `"stop"` claims the model finished; for a mid-sentence repetition
    // cut that is false, and every client action keyed on it (accept,
    // validate, commit, end the run) is then wrong. Do not re-invert.
    for guard in [
        "fuzzy_repetition",
        "inter_tool_prose_budget",
        "tool_envelope_stuck",
        "simhash_semantic_loop",
        "token_loop_watchdog",
    ] {
        assert_eq!(
            derive(Some(guard), Some(42), 100),
            "length",
            "guard={guard}"
        );
        // A guard trip on the exact step the budget ran out is still a
        // truncation, and both paths agree — precedence is deterministic.
        assert_eq!(derive(Some(guard), Some(42), 0), "length", "guard={guard}");
    }
}

#[test]
fn non_truncating_stops_are_not_relabelled_as_length() {
    // NEGATIVE case, and the whole point of the original fix: `"length"`
    // must NOT become a catch-all again. The bug this replaced derived it
    // from "the last token wasn't EOS", sweeping in early finalizes and
    // client cancels that are not truncations at all.
    //
    // No guard, budget left (client cancel / dropped receiver / drain):
    // generation stopped, nothing was truncated ⇒ "stop", never "length".
    assert_eq!(derive(None, Some(42), 100), "stop");
    // And the timeout guard keeps its own distinct reason rather than
    // collapsing into the truncation bucket.
    assert_eq!(
        derive(Some(GUARD_STOP_REQUEST_TIMEOUT), Some(42), 100),
        FINISH_REASON_TIMEOUT
    );
}

#[test]
fn token_derived_stops_outrank_non_timeout_guards() {
    // Preserved from the old test: a guard that fires on the same step
    // the model sampled EOS / the tool-call close reports what the
    // model actually did.
    assert_eq!(
        derive(Some("tool_envelope_stuck"), Some(151645), 100),
        "stop"
    );
    assert_eq!(
        derive(Some("fuzzy_repetition"), Some(151658), 100),
        "tool_calls"
    );
}

#[test]
fn empty_output_edges() {
    // max_tokens==0 scoring path: empty output, remaining==0 ⇒ "length"
    // (the budget of zero was exhausted before the first token).
    assert_eq!(derive(None, None, 0), "length");
    // Empty output on a model with NO tool-call end token configured
    // must not satisfy `None == None` and misreport "tool_calls".
    assert_eq!(
        derive_finish_reason(None, None, EOS, None, 5, 10, MAX_SEQ_LEN),
        "stop"
    );
}

// ─── Call-site proof: finish_sequence wires the REAL fields ────────────

/// Minimal `Model`: only the paths `finish_sequence` exercises
/// (`cache_sequence`, `free_sequence`, `ep_broadcast_cmd_for_seq`
/// default) are live; everything else is unreachable in these tests.
pub(super) struct StubModel;

impl Model for StubModel {
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _cs: usize,
        _cl: usize,
        _last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _n: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _bm: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn vocab_size(&self) -> usize {
        0
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn copy_logits_to_host(&self, _p: DevicePtr, _d: &mut [u8]) -> Result<()> {
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _p: DevicePtr, _st: u64) -> Result<u32> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn argmax_batch(&self, _p: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn cache_sequence(&self, _s: &SequenceState) {}
    fn free_sequence(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _slot: usize) -> Result<()> {
        Ok(())
    }
    fn detach_slot_for_reuse(&self, _s: &mut SequenceState) {}
    fn save_hidden_for_mtp(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
}

type RespRx = tokio::sync::oneshot::Receiver<Result<InferenceResponse>>;

/// A real `ActiveSeq` with a blocking oneshot sink. `min_tokens` is
/// deliberately set to a value DIFFERENT from `remaining` (7) so a
/// call-site mutation that passes the wrong field flips a test red.
fn test_seq(
    output_tokens: Vec<u32>,
    remaining: usize,
    guard_stop: Option<&'static str>,
    seq_len: usize,
) -> (ActiveSeq, RespRx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let now = Instant::now();
    let mut seq = SequenceState::host_only(0);
    seq.seq_len = seq_len;
    let a = ActiveSeq {
        seq,
        session_hash: 0,
        last_token: output_tokens.last().copied().unwrap_or(0),
        output_tokens,
        remaining,
        min_tokens: 7,
        eos_tokens: EOS.to_vec(),
        finished: true,
        guard_stop,
        param_close_pending: 0,
        sink: ResponseSink::Blocking(Some(tx)),
        cancel_flag: None,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        inside_thinking: false,
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 0,
        thinking_tokens: 0,
        force_end_thinking: false,
        think_force_closed: false,
        sentence_defer_count: 0,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token: None,
        think_start_token: None,
        think_ended: false,
        think_just_ended: false,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        think_skip_count: 0,
        tool_call_end_token: TOOL_END,
        require_tool_call: false,
        tool_request: false,
        tools_present: false,
        tool_call_start_token: None,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        suppress_tool_call: false,
        disable_mtp: false,
        mtp_acct: Default::default(),
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(0),
        grammar_state: None,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        last_token_time: now,
        request_start: now,
        decode_start: now,
        seed: None,
        top_logprobs: None,
        logprobs_data: Vec::new(),
        timeout_at: None,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
        cached_prompt_tokens: 0,
        preempt_immune_until_tokens: 0,
    };
    (a, rx)
}

fn finish_and_recv(mut a: ActiveSeq, mut rx: RespRx) -> InferenceResponse {
    finish_sequence(&StubModel, &mut a, MAX_SEQ_LEN);
    rx.try_recv()
        .expect("finish_sequence must send the blocking response")
        .expect("response must be Ok")
}

#[test]
fn call_site_passes_the_real_budget() {
    // Budget exhausted ⇒ "length". Red if finish_sequence stops passing
    // `a.remaining` (e.g. a constant, or the decoy `min_tokens` = 7).
    let (a, rx) = test_seq(vec![5, 6, 42], 0, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
    // Budget left ⇒ NOT "length" (same decoy: min_tokens=7 ≠ remaining).
    let (a, rx) = test_seq(vec![5, 6, 42], 7, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "stop");
}

#[test]
fn call_site_passes_the_real_seq_len() {
    // Context-ceiling stop with budget left ⇒ "length". Red if the
    // call site stops passing `a.seq.seq_len` / the served ceiling.
    let (a, rx) = test_seq(vec![5, 6, 42], 500, None, MAX_SEQ_LEN - 1);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
}

#[test]
fn call_site_passes_the_real_last_token_and_eos() {
    // Red if the call site stops passing `output_tokens.last()` or
    // `a.eos_tokens`.
    let (a, rx) = test_seq(vec![5, 6, 151645], 0, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "stop");
    let (a, rx) = test_seq(vec![5, 6, 151658], 3, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "tool_calls");
}

#[test]
fn call_site_passes_the_real_guard() {
    // Timeout is the one guard with a distinct wire reason — proves
    // `a.guard_stop` reaches the decision.
    let (a, rx) = test_seq(vec![5, 6, 42], 3, Some(GUARD_STOP_REQUEST_TIMEOUT), 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, FINISH_REASON_TIMEOUT);
    // And a degeneration guard reaches the response as "length" — the
    // truncation signal. Asserted at the CALL SITE, not just over the pure
    // function, because that is where the wire value the client actually
    // receives is decided.
    let (a, rx) = test_seq(vec![5, 6, 42], 3, Some("fuzzy_repetition"), 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
}
