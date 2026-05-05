// SPDX-License-Identifier: AGPL-3.0-only

//! Promote completed (fully prefilled) prefills into the active queue,
//! or surface error to the sink + free the sequence. Extracted from
//! `phase_continue_prefills` for the ≤500 LoC cap.

use spark_model::traits::Model;
use std::time::Instant;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn promote_completed_prefills(
    model: &dyn Model,
    prefilling: &mut Vec<PrefillInProgress>,
    mut completed_indices: Vec<(usize, Option<u32>)>,
    active: &mut Vec<ActiveSeq>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
) {
    // Process in reverse order so swap_remove indices stay valid.
    completed_indices.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (idx, maybe_token) in completed_indices {
        let p = prefilling.swap_remove(idx);
        let Some(first) = maybe_token else {
            // Error path: free the sequence.
            let mut seq = p.seq;
            if let Err(e) = model.free_sequence(&mut seq) {
                tracing::error!("phase_promote_prefills: free_sequence (error path): {e:#}");
            }
            if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                tracing::error!(
                    "phase_promote_prefills: ep_broadcast free+realloc (error path): {e:#}"
                );
            }
            continue;
        };
        let spontaneous_think = !p.enable_thinking && think_start_token == Some(first);
        // Only stream non-EOS tokens (OpenAI: stop seq not in output).
        if !spontaneous_think
            && !p.eos_tokens.contains(&first)
            && let ResponseSink::Streaming(ref tx) = p.sink
            && let Err(e) = tx.blocking_send(StreamEvent::Token(first))
        {
            tracing::warn!(
                "phase_promote_prefills: first-token send failed (receiver dropped): {e}"
            );
        }
        let use_legacy_tool_call =
            p.require_tool_call && p.grammar_state.is_none() && tool_call_start_token.is_some();
        let now = Instant::now();
        let cached_prompt_tok = p.seq.cached_prefix_tokens as u32;
        let immediate_finish =
            !spontaneous_think && (p.eos_tokens.contains(&first) || p.max_tokens <= 1);

        let mut a = build_active_seq_from_prefill(
            p,
            first,
            spontaneous_think,
            use_legacy_tool_call,
            cached_prompt_tok,
            immediate_finish,
            now,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
        );
        if immediate_finish {
            finish_sequence(model, &mut a);
        } else {
            active.push(a);
        }
    }
}

/// Construct an `ActiveSeq` from a finished prefill plus its first sampled
/// token. The two original branches (immediate finish + continue) used
/// near-identical field initialisation; this helper reuses the same record
/// and the caller decides whether to push onto active or finish_sequence.
#[allow(clippy::too_many_arguments)]
fn build_active_seq_from_prefill(
    p: PrefillInProgress,
    first: u32,
    spontaneous_think: bool,
    use_legacy_tool_call: bool,
    cached_prompt_tok: u32,
    immediate_finish: bool,
    now: Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
) -> ActiveSeq {
    let temperature = p.temperature;
    ActiveSeq {
        seq: p.seq,
        session_hash: p.session_hash,
        last_token: first,
        output_tokens: if !immediate_finish && spontaneous_think {
            vec![]
        } else {
            vec![first]
        },
        remaining: if immediate_finish {
            0
        } else {
            p.max_tokens - 1
        },
        min_tokens: p.min_tokens,
        eos_tokens: p.eos_tokens,
        finished: immediate_finish,
        sink: p.sink,
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        top_n_sigma: p.top_n_sigma,
        min_p: p.min_p,
        repetition_penalty: p.repetition_penalty,
        presence_penalty: p.presence_penalty,
        frequency_penalty: p.frequency_penalty,
        repetition_penalty_window: 256,
        lz_penalty: p.lz_penalty,
        dry_multiplier: p.dry_multiplier,
        dry_base: p.dry_base,
        dry_allowed_length: p.dry_allowed_length,
        dry_sequence_breakers: Vec::new(),
        logit_bias: p.logit_bias,
        pending_drafts: Vec::new(),
        inside_thinking: if immediate_finish {
            p.enable_thinking && think_end_token.is_some()
        } else {
            spontaneous_think || (p.enable_thinking && think_end_token.is_some())
        },
        enable_thinking: p.enable_thinking,
        thinking_budget: if !immediate_finish && spontaneous_think {
            Some(p.spontaneous_think_budget)
        } else {
            p.thinking_budget
        },
        spontaneous_think_budget: p.spontaneous_think_budget,
        thinking_tokens: 0,
        cached_prompt_tokens: cached_prompt_tok,
        force_end_thinking: false,
        consecutive_confident: 0,
        think_end_token,
        think_start_token,
        // When thinking is disabled but model supports thinking, the template
        // pre-closes with `<think>\n\n</think>\n\n`. Set think_ended=true so
        // the </think> logit suppression is active from the start.
        think_ended: if !immediate_finish && spontaneous_think {
            false
        } else {
            !p.enable_thinking && think_end_token.is_some()
        },
        think_just_ended: false,
        think_skip_count: 0,
        require_tool_call: use_legacy_tool_call,
        tool_call_start_token,
        tool_call_opened: false,
        inside_tool_body: false,
        suppress_tool_call: p.suppress_tool_call,
        disable_mtp: p.disable_mtp,
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        entropy_collapse_streak: 0,
        f27_fingerprint_ring: std::collections::VecDeque::with_capacity(F27_RING_CAP),
        f27_attractor_streak: 0,
        f27_last_emitted_token: 0,
        tool_call_end_token,
        grammar_state: p.grammar_state,
        last_token_time: now,
        request_start: p.request_start,
        decode_start: now,
        seed: p.seed,
        top_logprobs: p.top_logprobs,
        logprobs_data: Vec::new(),
        timeout_at: p.timeout_at,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
    }
}
