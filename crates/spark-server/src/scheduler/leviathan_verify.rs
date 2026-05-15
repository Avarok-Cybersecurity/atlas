// SPDX-License-Identifier: AGPL-3.0-only

//! Leviathan-2023 rejection sampling for MTP verify positions.
//!
//! Atlas's MTP proposer head emits drafts via pure argmax
//! (`spark-model/src/layers/mtp_head/forward.rs:382-488`), so for any
//! draft token x the proposer's probability `p_draft(x) = 1.0` at the
//! argmax token and 0 elsewhere. Leviathan's acceptance rule
//! `min(1, p_target(x) / p_draft(x))` collapses to `p_target(x)`, and
//! the residual on reject `max(0, p_target − p_draft) / Z` becomes the
//! target distribution with the draft token zeroed out and
//! renormalised. We never need to expose draft logits from the
//! proposer for this iteration; target logits at host
//! (`model.logits_buffer_ptr()` → `[K, vocab]` BF16) plus the seq's
//! penalty stack are sufficient.
//!
//! The loop-breaking property: penalties (DRY + LZ) are applied to the
//! TARGET logits before computing `p_target(draft)`. On an attractor
//! the penalised softmax assigns the attractor token ~0.1 instead of
//! ~0.9, so acceptance is probabilistic ≈10%, and on reject we sample
//! from the penalty-adjusted residual (excluding the attractor) →
//! loop breaks. On natural code with no penalty-active suffix-match
//! the penalty is a no-op, p_target(draft) ≈ 0.95-1.0, and MTP
//! acceptance stays ≥70%.
//!
//! OpenAI-style penalties (`repetition_penalty`, `presence_penalty`,
//! `frequency_penalty`) are HARD-SUPPRESSED here. Empirically
//! (2026-05-11) applying `presence_penalty=1.5` at verify shifted
//! argmax on every common code-token (`{`, `;`, `\n`) → drafts vs
//! verifier disagreed everywhere → MTP collapsed to ~0% acceptance.
//! Only DRY+LZ fire on actual suffix-match repeats and are safe at
//! the argmax position.

use super::ActiveSeq;
use anyhow::Result;
use spark_model::traits::Model;
use spark_runtime::sampler::{
    SamplingParams, apply_dry_penalty, apply_lz_penalty, sample_excluding, seeded_uniform_f32,
    softmax_token_prob,
};

use super::helpers::bf16_to_f32;

/// Outcome of running Leviathan rejection sampling over K verify
/// positions. The caller (`verify_kN_step.rs`) uses `num_accepted` to
/// drive the existing cascade arithmetic and emits `reject_token` (if
/// `Some`) at position `num_accepted` instead of the verifier's
/// argmax. On full draft acceptance, `tail_token` carries the verifier
/// tail token. It is usually the raw argmax, but guards such as
/// incomplete-HTML EOS suppression may replace it with a masked argmax.
pub struct VerifyAccept {
    pub num_accepted: usize,
    pub reject_token: Option<u32>,
    pub tail_token: Option<u32>,
}

/// Whether MTP verify needs Leviathan rejection sampling for this seq.
///
/// The loop-detecting penalties (DRY, LZ) trigger the host-side path.
/// The incomplete-HTML EOS guard also needs host logits so it can mask
/// EOS before acceptance/tail selection.
pub fn verify_needs_leviathan(a: &ActiveSeq) -> bool {
    a.dry_multiplier > 0.0
        || a.lz_penalty > 0.0
        || super::helpers::should_suppress_eos_for_html(
            &a.output_tokens,
            a.inside_thinking,
            a.inside_tool_body,
            a.grammar_state.is_some(),
        )
}

/// Run Leviathan rejection sampling over `K = argmax_tokens.len()`
/// verify positions and return the cascade outcome.
///
/// Acceptance per position: read penalised softmax probability at the
/// proposer's draft token; accept with that probability. On the first
/// reject, sample a residual token from the penalised distribution
/// with the draft excluded, and return `num_accepted = i`. On full
/// accept, return `num_accepted = K, reject_token = None`.
///
/// Falls back to the argmax cascade (current pre-rejection behaviour)
/// on D2H copy failure or when `verify_needs_leviathan(a)` is false.
pub fn verify_leviathan(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    argmax_tokens: &[u32],
) -> VerifyAccept {
    if !verify_needs_leviathan(a) {
        return cascade_argmax(drafts, argmax_tokens);
    }
    let k = argmax_tokens.len();
    let vocab = model.vocab_size();
    let mut buf = vec![0_u8; k * vocab * 2];
    if model
        .copy_logits_to_host(model.logits_buffer_ptr(), &mut buf)
        .is_err()
    {
        tracing::debug!("verify_leviathan: D2H copy failed; falling back to argmax cascade");
        return cascade_argmax(drafts, argmax_tokens);
    }
    match run_leviathan_inner(a, drafts, argmax_tokens, &buf, vocab) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("verify_leviathan: inner failed ({e:#}); falling back to argmax");
            cascade_argmax(drafts, argmax_tokens)
        }
    }
}

fn cascade_argmax(drafts: &[u32], argmax_tokens: &[u32]) -> VerifyAccept {
    for (i, (&draft, &argmax)) in drafts.iter().zip(argmax_tokens.iter()).enumerate() {
        if draft != argmax {
            return VerifyAccept {
                num_accepted: i,
                reject_token: Some(argmax),
                tail_token: None,
            };
        }
    }
    VerifyAccept {
        num_accepted: drafts.len(),
        reject_token: None,
        tail_token: argmax_tokens.get(drafts.len()).copied(),
    }
}

fn run_leviathan_inner(
    a: &ActiveSeq,
    drafts: &[u32],
    argmax_tokens: &[u32],
    bf16_buf: &[u8],
    vocab: usize,
) -> Result<VerifyAccept> {
    let in_tool = a.inside_tool_body && !a.inside_thinking;
    let dry_active = a.dry_multiplier > 0.0 && !in_tool && a.grammar_state.is_none();
    let lz_active = a.lz_penalty > 0.0 && !in_tool && a.grammar_state.is_none();
    let dry_allowed = a.dry_allowed_length;
    let html_suppresses_eos = super::helpers::should_suppress_eos_for_html(
        &a.output_tokens,
        a.inside_thinking,
        a.inside_tool_body,
        a.grammar_state.is_some(),
    );
    // Penalties are applied to `logits` directly below; `params` here is
    // only for the `sample_excluding` residual sample and MUST NOT
    // re-apply them (`sample_with_params_history` would otherwise
    // double-penalise, distorting the residual distribution).
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        edt_strength: 0.0,
        edt_floor: 0.1,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: a.seed,
    };

    // Only the first `drafts.len()` argmax positions correspond to
    // actual proposer drafts — the final argmax slot is the natural
    // tail token emitted on full accept. K=2 verify: 1 draft → 2
    // argmax positions; K=3 verify: 2 drafts → 3 positions; K=4: 3
    // drafts → 4 positions. We only do rejection sampling over the
    // draft positions.
    // Window the penalty history to the last 256 tokens — matches
    // the sampler's `repetition_penalty_window` default
    // (`decode_logits_seq.rs:267`). At long contexts (10k+ tokens)
    // a full-history scan finds spurious 5-grams that aren't
    // structural loops but happen to recur once in 10k tokens,
    // shifting penalised argmax away from the proposer's draft and
    // tanking acceptance. The window covers the practical
    // loop-attractor range (5-50 token cycles, 50-500 token paraphrases);
    // longer attractors are out of scope of DRY/LZ semantics anyway.
    const VERIFY_PENALTY_WINDOW: usize = 256;
    let hist_full = &a.seq.tokens;
    let window_start = hist_full.len().saturating_sub(VERIFY_PENALTY_WINDOW);
    let mut hist: Vec<u32> = hist_full[window_start..].to_vec();
    for i in 0..drafts.len() {
        if i > 0 {
            hist.push(drafts[i - 1]);
        }
        let slice = &bf16_buf[i * vocab * 2..(i + 1) * vocab * 2];
        let mut logits: Vec<f32> = (0..vocab)
            .map(|j| bf16_to_f32(slice[j * 2], slice[j * 2 + 1]))
            .collect();

        if lz_active && hist.len() >= 4 {
            apply_lz_penalty(&mut logits, &hist, a.lz_penalty);
        }
        if dry_active && hist.len() >= 3 {
            apply_dry_penalty(
                &mut logits,
                &hist,
                a.dry_multiplier,
                a.dry_base,
                dry_allowed,
                &a.dry_sequence_breakers,
            );
        }
        if html_suppresses_eos {
            mask_eos(&mut logits, &a.eos_tokens);
        }
        for &(tid, bias) in &a.logit_bias {
            if (tid as usize) < vocab {
                logits[tid as usize] += bias;
            }
        }

        let draft = drafts[i];
        let p_target = softmax_token_prob(&logits, draft);
        let seed = a
            .seed
            .unwrap_or(0)
            .wrapping_add(a.output_tokens.len() as u64)
            .wrapping_add(i as u64);
        let u = seeded_uniform_f32(seed);
        let accept = u <= p_target;
        // Sampled per-position diagnostics: every 50 emitted tokens
        // and only when the proposer's draft DISAGREES with the
        // verifier's argmax (the interesting case for tuning).
        // Power-user verbose: `RUST_LOG=spark::scheduler::leviathan_verify=debug`.
        if a.output_tokens.len().is_multiple_of(50) && draft != argmax_tokens[i] {
            tracing::info!(
                "MTP leviathan: pos={} draft={} argmax={} p_target={:.3} u={:.3} accept={} (DRY={} LZ={})",
                i,
                draft,
                argmax_tokens[i],
                p_target,
                u,
                accept,
                dry_active,
                lz_active,
            );
        }
        if accept {
            continue;
        }
        // EOS-class tokens are suppressed in the rejection residual
        // ONLY when generation is in a structural state where EOS
        // would be premature (mid-HTML / inside a tool body / under
        // an unterminated grammar). For free chat completion, EOS
        // must pass through the rejection residual — otherwise the
        // model is unable to terminate via MTP K=2 once the
        // proposer drafts a non-EOS token, since MTP heads almost
        // never draft EOS, and the accept-path-only termination
        // route is rarely taken. 2026-05-14: previously this loop
        // masked EOS unconditionally, which dragged short answers
        // like "19" into chat-template-leakage tails ("19\n\n\n19").
        let eos_suppressed = html_suppresses_eos
            || a.inside_tool_body
            || a
                .grammar_state
                .as_ref()
                .is_some_and(|gs| !gs.is_terminated())
            || a.require_tool_call;
        if eos_suppressed {
            for &eos in &a.eos_tokens {
                if (eos as usize) < logits.len() {
                    logits[eos as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let reject = sample_excluding(&logits, &params, &hist, draft);
        return Ok(VerifyAccept {
            num_accepted: i,
            reject_token: Some(reject),
            tail_token: None,
        });
    }
    let tail_token = guarded_tail_token(
        a,
        bf16_buf,
        vocab,
        drafts,
        &hist,
        argmax_tokens.get(drafts.len()).copied(),
        dry_active,
        lz_active,
        dry_allowed,
        html_suppresses_eos,
    );
    Ok(VerifyAccept {
        num_accepted: drafts.len(),
        reject_token: None,
        tail_token,
    })
}

#[allow(clippy::too_many_arguments)]
fn guarded_tail_token(
    a: &ActiveSeq,
    bf16_buf: &[u8],
    vocab: usize,
    drafts: &[u32],
    hist_before_last_draft: &[u32],
    original_tail: Option<u32>,
    dry_active: bool,
    lz_active: bool,
    dry_allowed: u32,
    html_suppresses_eos: bool,
) -> Option<u32> {
    let tail_pos = drafts.len();
    let tail = original_tail?;
    if !html_suppresses_eos || !a.eos_tokens.contains(&tail) {
        return Some(tail);
    }
    let slice_start = tail_pos.checked_mul(vocab)?.checked_mul(2)?;
    let slice_end = slice_start.checked_add(vocab.checked_mul(2)?)?;
    let slice = bf16_buf.get(slice_start..slice_end)?;
    let mut logits: Vec<f32> = (0..vocab)
        .map(|j| bf16_to_f32(slice[j * 2], slice[j * 2 + 1]))
        .collect();
    let mut hist = hist_before_last_draft.to_vec();
    if let Some(&last_draft) = drafts.last() {
        hist.push(last_draft);
    }
    if lz_active && hist.len() >= 4 {
        apply_lz_penalty(&mut logits, &hist, a.lz_penalty);
    }
    if dry_active && hist.len() >= 3 {
        apply_dry_penalty(
            &mut logits,
            &hist,
            a.dry_multiplier,
            a.dry_base,
            dry_allowed,
            &a.dry_sequence_breakers,
        );
    }
    mask_eos(&mut logits, &a.eos_tokens);
    for &(tid, bias) in &a.logit_bias {
        if (tid as usize) < vocab {
            logits[tid as usize] += bias;
        }
    }
    let best = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(tail);
    tracing::debug!(
        original_tail = tail,
        replacement = best,
        "HTML completion guard: replaced MTP tail EOS before </html>"
    );
    Some(best)
}

fn mask_eos(logits: &mut [f32], eos_tokens: &[u32]) {
    for &eos in eos_tokens {
        if (eos as usize) < logits.len() {
            logits[eos as usize] = f32::NEG_INFINITY;
        }
    }
}
