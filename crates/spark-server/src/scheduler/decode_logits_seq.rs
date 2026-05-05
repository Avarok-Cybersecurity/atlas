// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits per-sequence helper (extracted to keep parent file ≤500 LoC).

use super::*;

/// Process logits for a single active sequence: dequant, adjust, sample, return token + optional logprobs.
#[allow(clippy::too_many_arguments)]
pub fn process_seq_logits(
    _model: &dyn Model,
    a: &mut ActiveSeq,
    buf: &[u8],
    i: usize,
    vocab_size: usize,
    elem_bytes: usize,
    logits_fp32: bool,
    think_end_token: Option<u32>,
    _think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    _tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) -> (u32, Option<crate::api::TokenLogprobs>) {
    let slice = &buf[i * vocab_size * elem_bytes..(i + 1) * vocab_size * elem_bytes];
    let mut f32_logits: Vec<f32> = if logits_fp32 {
        // Direct FP32: 4 bytes/element little-endian.
        (0..vocab_size)
            .map(|j| {
                let off = j * 4;
                f32::from_le_bytes([slice[off], slice[off + 1], slice[off + 2], slice[off + 3]])
            })
            .collect()
    } else {
        // BF16 → FP32 expansion.
        (0..vocab_size)
            .map(|j| {
                let lo = slice[j * 2];
                let hi = slice[j * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };

    // F1: Reflection token suppression during thinking.
    // Penalize "wait", "however", "actually" etc. to prevent circular reasoning.
    if a.inside_thinking {
        for &rid in reflection_suppress_ids {
            if (rid as usize) < f32_logits.len() {
                f32_logits[rid as usize] -= 10.0;
            }
        }
    }

    // F2: Confidence-based early stop during thinking.
    // F2: Confidence-based early stop during thinking.
    // When top-1 prob >= 0.95 for 30 consecutive tokens, force </think>.
    // Only kicks in after 400 thinking tokens — the model needs room to
    // plan (numbered lists, step-by-step reasoning have high per-token
    // confidence but are NOT signs the model is done thinking).
    // Previous thresholds (200 tokens, 10 consecutive) were too aggressive
    // and caused premature thinking termination in agentic coding sessions.
    if a.inside_thinking && !a.force_end_thinking && a.thinking_tokens >= 400 {
        let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = f32_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        if sum_exp > 0.0 && 1.0 / sum_exp >= 0.95 {
            a.consecutive_confident += 1;
            if a.consecutive_confident >= 30 {
                a.force_end_thinking = true;
                tracing::info!(
                    "Confidence early stop: top-1 prob >= 0.95 for 30 tokens (after {} thinking tokens)",
                    a.thinking_tokens
                );
            }
        } else {
            a.consecutive_confident = 0;
        }
    }

    // After thinking is done, suppress the </think> token to prevent
    // degenerate loops where the model generates hundreds of </think>.
    if a.think_ended {
        if let Some(end_tok) = think_end_token {
            let end_idx = end_tok as usize;
            if end_idx < f32_logits.len() {
                f32_logits[end_idx] = f32::NEG_INFINITY;
            }
        }
        // F9 (2026-04-26): symmetric mask for the START token.
        // Once `think_ended` is true (watchdog forced close OR
        // model emitted </think> naturally), the model must not
        // re-enter thinking in the same response. Without this
        // mask, the spontaneous-<think> re-entry path at the
        // emit site flips `inside_thinking=true` again on any
        // sampled <think>, and the watchdog fires again ~8s
        // later — observed three rapid re-entries on
        // 2026-04-26 fix29 logs. arXiv evidence: s1
        // (2501.19393), DeepSeek-R1, Qwen3 (2505.09388),
        // Production Repetition (2512.04419) all mask the
        // open token after first close. Chain-of-Draft
        // (2502.18600) ablates penalty stacking (12% drop) vs
        // hard masking (94% drop) — masking dominates.
        if let Some(start_tok) = a.think_start_token {
            let start_idx = start_tok as usize;
            if start_idx < f32_logits.len() {
                f32_logits[start_idx] = f32::NEG_INFINITY;
            }
        }
    }

    // Suppress <tool_call> during thinking (prevents KV cache contamination
    // from think-leak bug) AND when tool call loop detected (≥4 identical
    // calls — see api.rs:548). For the loop case, use a STRONG NEGATIVE
    // BIAS (−12.0) instead of `-inf` so the model can still escape if its
    // evidence for a tool call is overwhelming (e.g. user explicitly says
    // "actually run the tests"). For thinking, hard-mask remains: tool
    // calls inside <think> are unparsable per the (canonical) qwen3_coder
    // dialect, so they must be physically blocked.
    if a.inside_thinking {
        if let Some(tc_start) = tool_call_start_token {
            let idx = tc_start as usize;
            if idx < f32_logits.len() {
                f32_logits[idx] = f32::NEG_INFINITY;
            }
        }
    } else if a.suppress_tool_call
        && let Some(tc_start) = tool_call_start_token
    {
        let idx = tc_start as usize;
        if idx < f32_logits.len() {
            f32_logits[idx] -= 12.0;
        }
    }

    // Force </think> when budget exhausted OR confidence early stop triggered.
    if a.inside_thinking
        && a.force_end_thinking
        && let Some(end_tok) = think_end_token
    {
        let end_idx = end_tok as usize;
        if end_idx < f32_logits.len() {
            for logit in f32_logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            f32_logits[end_idx] = 0.0;
        }
    }

    // Change 3b: one-shot pin-to-tool-call-start.
    // When the previous token was `</think>` AND the request
    // requires a tool call AND no tool-call has been opened yet,
    // mask all logits to -inf except `tool_call_start_token`.
    // This prevents architectures like MiniMax M2 (which always
    // thinks via the chat template) from wandering into prose
    // after `</think>` instead of emitting the structured tool
    // call. Models that don't have `require_tool_call` set
    // (i.e. the request didn't pass tools) skip this entirely.
    if a.think_just_ended
        && a.require_tool_call
        && !a.tool_call_opened
        && !a.inside_thinking
        && let Some(start_tok) = tool_call_start_token
    {
        let idx = start_tok as usize;
        if idx < f32_logits.len() {
            for logit in f32_logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            f32_logits[idx] = 0.0;
            tracing::debug!("Forced tool_call_start_token after </think> (require_tool_call set)");
        }
    }

    // F70 (2026-04-29, attempted): canonical-opener anchor
    // bias was REVERTED. xgrammar's TagDispatch is non-anchored
    // (verified by
    // `grammar.rs::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`),
    // and a flat +2.5 logit boost on `tool_call_start_token`
    // pushes the model into the tool body too aggressively —
    // observed live: 1-tool prompts produce
    // `tool_calls[0].function.arguments = {"command":""}`
    // because the model rushes through the envelope with no
    // parameter values. The proper fix is byte-level partial
    // trigger anchoring (mask trigger-breaker tokens only when
    // a partial-match suffix is actually present in recent
    // output) but that's a follow-up — for now we accept the
    // "model occasionally drifts on stressed prompts"
    // limitation and rely on F26/F2 to terminate the response
    // cleanly when it happens.

    // Apply grammar bitmask BEFORE sampling — but NOT during
    // `<think>`…`</think>`. Thinking is free-form reasoning that
    // is stripped from the final API response, so forcing it
    // through a JSON-tool-call grammar produces garbage
    // punctuation streams (observed with opencode: the assistant
    // thinking channel filled with `!.,),,,***` before the
    // model recovered after `</think>`).
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
        && gs.fill_bitmask()
    {
        gs.apply_bitmask_to_logits(&mut f32_logits);
    }

    // F72 (byte-level partial-trigger anchor) was removed — see
    // F73 / fix42. The sampler-side anchor hung the server in
    // production despite passing isolated unit tests; the
    // model's broken-envelope case is now recovered at the
    // streaming-sanitizer + parser layer (F73 + F71). The
    // xgrammar non-anchored TagDispatch limitation is pinned
    // by `grammar.rs::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`
    // for documentation only.

    // ── Adaptive sampling: update zone, observe entropy, check greedy gate ──
    // Disabled by default (--adaptive-sampling flag). Each call scans the
    // full vocab (262k) on CPU: entropy O(V) exp+log, greedy gate O(V) exp.
    // Cost: ~300-400µs per token → 2-3x throughput regression when enabled.
    let greedy_gate = if adaptive_sampling {
        a.adaptive.update_zone(
            a.tool_call_opened,
            a.inside_thinking,
            a.grammar_state.is_some(),
        );
        a.adaptive.observe_entropy(&f32_logits);
        a.adaptive.update_lz_ratio(&a.output_tokens);
        a.adaptive.should_use_greedy(&f32_logits)
    } else {
        false
    };
    let effective_temp = if adaptive_sampling {
        a.adaptive.effective_temperature()
    } else {
        a.temperature
    };

    // Unified sampling path: stochastic OR greedy (temp==0 or
    // adaptive greedy_gate) both go through
    // `sample_with_params_history`. The function applies all
    // configured penalties (repetition / presence / frequency /
    // LZ / DRY) and logit_bias BEFORE the temperature decision,
    // so greedy argmax respects MODEL.toml-configured penalties
    // — matching HF Transformers / vLLM / llama.cpp behavior.
    //
    // The earlier "Pure greedy argmax — NO penalties" bypass
    // here was the load-bearing bug for Gemma-4-31B's greedy
    // fib failure: `MODEL.toml` configures rep_penalty=1.1 but
    // the bypass dropped it. After this change, the configured
    // penalty applies at temp=0.
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    // Force temp=0 for greedy_gate path (adaptive override) so
    // sample_with_params_seeded takes the post-penalty argmax
    // branch instead of running the full stochastic pipeline.
    let sampling_temp = if greedy_gate { 0.0 } else { effective_temp };
    // Advance seed per token for deterministic but varying randomness.
    let step_seed = a.seed.map(|s| s.wrapping_add(a.output_tokens.len() as u64));
    // Phase-gated sampler scoping (P3.1, 2026-04-25):
    // inside the tool-call body (between `<tool_call>` and
    // `</tool_call>`) the JSON we emit is dense with
    // legitimate short repetitions — `":"`, `","`, key
    // tokens — that DRY/presence_penalty/frequency_penalty
    // would otherwise penalise, breaking schema validity.
    // XGrammar already guarantees structural correctness
    // here; penalties only add noise. Outside the tool
    // body (free text + `<think>`) the full preset
    // applies: this is where prose loops actually live.
    let in_tool = a.inside_tool_body && !a.inside_thinking;
    let sampled = sample_with_params_history(
        f32_bytes,
        &SamplingParams {
            temperature: sampling_temp,
            top_k: a.top_k,
            top_p: a.top_p,
            top_n_sigma: a.top_n_sigma,
            min_p: a.min_p,
            logit_bias: a.logit_bias.clone(),
            repetition_penalty: if in_tool { 1.0 } else { a.repetition_penalty },
            repetition_penalty_window: a.repetition_penalty_window,
            presence_penalty: if in_tool { 0.0 } else { a.presence_penalty },
            frequency_penalty: if in_tool { 0.0 } else { a.frequency_penalty },
            lz_penalty: if a.grammar_state.is_some() {
                0.0
            } else {
                a.lz_penalty
            },
            edt_strength: 0.0,
            edt_floor: 0.1,
            // DRY: same logic. Outside the tool body it
            // remains active to dampen `<think>` fence-narration
            // attractors. Inside the body, disabled — JSON
            // patterns repeat and that's correct.
            dry_multiplier: if in_tool { 0.0 } else { a.dry_multiplier },
            dry_base: a.dry_base,
            dry_allowed_length: a.dry_allowed_length,
            dry_sequence_breakers: a.dry_sequence_breakers.clone(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: step_seed,
        },
        &a.output_tokens,
    );

    // F26 (2026-04-26): kernel-level entropy-collapse guard.
    // The sampler thread-locally records the post-truncation
    // entropy for the just-sampled distribution; read it here.
    // If H is below threshold AND we're outside any guard
    // (thinking, tool-body, grammar, warmup), increment the
    // per-sequence streak counter. When the streak reaches
    // ENTROPY_COLLAPSE_STREAK_K consecutive collapsed samples,
    // force EOS — the model is on a distribution attractor
    // that token-level dedup may miss.
    //
    // Greedy sampling bypasses record_entropy so last_entropy()
    // returns stale data; skip the check in that case.
    if !(a.temperature == 0.0
        || greedy_gate
        || a.inside_thinking
        || a.inside_tool_body
        || a.grammar_state.is_some())
        && a.output_tokens.len() >= ENTROPY_COLLAPSE_WARMUP_TOKENS
    {
        let h = spark_runtime::sampler::last_entropy();
        if h < ENTROPY_COLLAPSE_THRESHOLD_NATS {
            a.entropy_collapse_streak = a.entropy_collapse_streak.saturating_add(1);
            #[allow(clippy::absurd_extreme_comparisons)]
            if ENTROPY_COLLAPSE_STREAK_K > 0
                && a.entropy_collapse_streak >= ENTROPY_COLLAPSE_STREAK_K
            {
                tracing::warn!(
                    entropy_nats = h,
                    streak = a.entropy_collapse_streak,
                    output_tokens = a.output_tokens.len(),
                    "F26: entropy-collapse guard fired (sustained low H outside guards); ending response"
                );
                a.finished = true;
            }
        } else {
            a.entropy_collapse_streak = 0;
        }
    } else {
        // In a guarded context — reset streak so re-entering
        // free-text region after a tool call starts fresh.
        a.entropy_collapse_streak = 0;
    }

    // F27 (2026-04-26): logit-distribution fingerprint
    // attractor detector. Same guards as F26 but uses a
    // distinct signal (logit landscape stability with token
    // diversity) so the two compose: F26 catches "same
    // distribution shape AND model is decisive"; F27
    // catches "same distribution shape AND model is sampling
    // jitter on top of it" (token varies but the ranked
    // candidates don't).
    if !(a.temperature == 0.0
        || greedy_gate
        || a.inside_thinking
        || a.inside_tool_body
        || a.grammar_state.is_some())
        && a.output_tokens.len() >= ENTROPY_COLLAPSE_WARMUP_TOKENS
    {
        let fp = fingerprint_logits_strided(&f32_logits);
        let token_diversified = sampled != a.f27_last_emitted_token;
        let dup = a.f27_fingerprint_ring.iter().any(|&p| p == fp);
        if dup && token_diversified {
            a.f27_attractor_streak = a.f27_attractor_streak.saturating_add(1);
        } else if !dup {
            a.f27_attractor_streak = 0;
        }
        if a.f27_fingerprint_ring.len() >= F27_RING_CAP {
            a.f27_fingerprint_ring.pop_front();
        }
        a.f27_fingerprint_ring.push_back(fp);
        a.f27_last_emitted_token = sampled;
        if a.f27_attractor_streak >= F27_STREAK_K {
            tracing::warn!(
                ring_len = a.f27_fingerprint_ring.len(),
                streak = a.f27_attractor_streak,
                output_tokens = a.output_tokens.len(),
                "F27: logit-fingerprint attractor (stable distribution, varying tokens); ending response"
            );
            a.finished = true;
        }
    } else {
        // Guarded — reset streak (but keep ring so cross-guard
        // entry doesn't lose recent fingerprints).
        a.f27_attractor_streak = 0;
        a.f27_last_emitted_token = sampled;
    }

    // Extract top-K logprobs from f32_logits if requested.
    let logprobs = a
        .top_logprobs
        .map(|k| extract_logprobs_from_f32(&f32_logits, sampled, k as usize));
    (sampled, logprobs)
}
