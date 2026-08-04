// SPDX-License-Identifier: AGPL-3.0-only

//! `PrefillInProgress` builder for `start_chunked_prefill`.
//!
//! Hoisted from `prefill_a_step.rs` to keep each file under the 500-LoC
//! file-size cap. Both the deferred-co-dispatch early return and the
//! non-last-chunk return build an identical `PrefillInProgress` (only
//! `chunk_offset` differs); this builder is that construction 1:1, so the
//! two call sites pass their already-bound locals straight through.

use std::sync::Arc;
use std::time::Instant;

use spark_model::traits::SequenceState;

use crate::api::inference_types::RepetitionDetectionParams;
use crate::grammar::GrammarState;

use super::types::{PrefillInProgress, ResponseSink};

/// Build a `PrefillInProgress` from the per-request locals. Mirrors the
/// previous inline struct literals exactly; `chunk_offset` is the only field
/// that varies between the deferred (0) and non-last-chunk (chunk_len) paths.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_prefill_in_progress(
    prompt_tokens: Arc<Vec<u32>>,
    session_hash: u64,
    seq: SequenceState,
    chunk_offset: usize,
    max_tokens: usize,
    min_tokens: usize,
    eos_tokens: Vec<u32>,
    sink: ResponseSink,
    cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    request_start: Instant,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    top_n_sigma: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    lz_penalty: f32,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: u32,
    logit_bias: Vec<(u32, f32)>,
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    repetition_detection: Option<RepetitionDetectionParams>,
    spontaneous_think_budget: u32,
    require_tool_call: bool,
    tools_present: bool,
    suppress_tool_call: bool,
    disable_mtp: bool,
    grammar_state: Option<GrammarState>,
    seed: Option<u64>,
    top_logprobs: Option<u8>,
    timeout_at: Option<Instant>,
) -> PrefillInProgress {
    PrefillInProgress {
        prompt_tokens,
        session_hash,
        seq,
        chunk_offset,
        max_tokens,
        min_tokens,
        eos_tokens,
        sink,
        cancel_flag,
        request_start,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        repetition_penalty_window: 256,
        presence_penalty,
        frequency_penalty,
        lz_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_sequence_breakers: Vec::new(),
        logit_bias,
        enable_thinking,
        thinking_budget,
        repetition_detection,
        spontaneous_think_budget,
        require_tool_call,
        tools_present,
        suppress_tool_call,
        disable_mtp,
        grammar_state,
        seed,
        top_logprobs,
        timeout_at,
    }
}
