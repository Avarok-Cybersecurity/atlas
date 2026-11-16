// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: decide whether one sequence's verify span may be picked on the device (lane L12),
//! and translate the host pre-sample pipeline's effect on its logits into the device edit lists.
//!
//! The host slow path (`pick_positions_from_host` -> `verify_pick_with_pipeline`) runs, per verify
//! position: the nine-stage pipeline (`logit_processors::run_pipeline_with_path`), then
//! `apply_penalties_and_bias`, then an argmax or a sample. For a row with no grammar, no
//! `logit_bias`, no live LZ or DRY penalty and no stateful thinking stage, every remaining stage is a
//! function of the sequence's step-start state only, so its effect is a fixed edit per touched token:
//!
//! | stage | edit |
//! |---|---|
//! | `MinTokensEosMask` | `eos` on each EOS id; the request's `eos_until = min_tokens - output_len` |
//! | `MidWordThinkEndMask`, `PostCloseThinkMask`, `ToolCallDuringThinkingMask` (inside) | `mask` |
//! | `ToolCallDuringThinkingMask` (outside, `suppress_tool_call`) | `sub12` |
//! | repetition penalty | `rep_count` = occurrences in the window |
//! | presence and frequency | `delta = freq * count as f32 + pres` |
//! | `logit_bias` of `penalty_params_for` (the A4 floor) | `bias`, in order |
//!
//! The one state change inside the span is a `</think>` pick while thinking, which
//! `pick_positions_from_host` turns into "outside thinking, think ended, just ended" for every later
//! position. That is regime 1: its own edit list (built from the flipped flags, exactly as the host
//! rebuilds `penalty_params_for` per position), selected by the walk after the first `</think>`.
//!
//! Stages this planner refuses (the row keeps the host pipeline, counted by reason): any grammar
//! (bitmask, forced token), the F2 confidence stop (reads the softmax and mutates the sequence), the
//! forced `</think>` injector when armed (blanket mask or a per-position counter), the legacy
//! `<tool_call>` pin, LZ and DRY, AdaDec recording and `AVAROK_FORCE_TEMP_ZERO`.

use std::collections::BTreeMap;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use spark_model::port::spec_accept::metrics::Fallback;
use spark_model::port::spec_accept::{
    AcceptMode, CAND_CAP, EditDesc, MAX_BIAS, MAX_POS, SampleParams,
};
use spark_runtime::sampler::SamplingParams;

use crate::scheduler::ActiveSeq;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::sample_step::{
    PositionKind, effective_min_p, penalty_history_scope, penalty_params_for,
};

/// One verify call, ready for the device.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::scheduler::verify_pipeline_helper) struct Plan {
    pub(in crate::scheduler::verify_pipeline_helper) mode: AcceptMode,
    /// Regime 0 (step-start state) and, when the row is inside thinking, regime 1 (after `</think>`).
    pub(in crate::scheduler::verify_pipeline_helper) regimes: Vec<Vec<EditDesc>>,
    pub(in crate::scheduler::verify_pipeline_helper) eos_until: u32,
    pub(in crate::scheduler::verify_pipeline_helper) rep_penalty: f32,
    pub(in crate::scheduler::verify_pipeline_helper) rng_key: u64,
    pub(in crate::scheduler::verify_pipeline_helper) pos_base: u64,
}

/// The thinking flags `pick_positions_from_host` carries across a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Flags {
    inside_thinking: bool,
    think_ended: bool,
    think_just_ended: bool,
}

/// Plan the verify call of `k` positions over a `vocab`-wide BF16 logits row set, or say why not.
///
/// `a` is only read; its thinking flags are flipped while regime 1's penalties are built and restored
/// before returning, exactly as the host loop does.
pub(in crate::scheduler::verify_pipeline_helper) fn build(
    vocab: usize,
    k: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Result<Plan, Fallback> {
    if k == 0 || k > MAX_POS {
        return Err(Fallback::Positions);
    }
    let edit_cap = spark_model::port::spec_accept::max_edits(vocab);
    if edit_cap == 0 {
        return Err(Fallback::Vocab);
    }
    if a.grammar_state.is_some() {
        return Err(Fallback::Grammar);
    }
    if !a.logit_bias.is_empty() {
        return Err(Fallback::LogitBias);
    }
    if ctx.sampling.force_temp_zero || ctx.sampling.adadec_diagnostic || ctx.dumps.adadec.is_some()
    {
        return Err(Fallback::Diagnostic);
    }
    let start = Flags {
        inside_thinking: a.inside_thinking,
        think_ended: a.think_ended,
        think_just_ended: a.think_just_ended,
    };
    // F2 confidence early stop: reads the row's softmax and mutates `consecutive_confident`.
    let f2_active = !ctx.sampling.disable_watchdogs
        && a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens >= 400
        && ctx.watchdog.confidence_early_stop;
    // ForcedThinkEndInjector acts only when armed: a blanket mask, or a per-position defer counter.
    if f2_active || (a.inside_thinking && a.force_end_thinking) {
        return Err(Fallback::ThinkStage);
    }
    let crosses = a.inside_thinking && ctx.think_end_token.is_some();
    let pin_ready = a.require_tool_call && !a.tool_call_opened;
    let pin_now = start.think_just_ended && pin_ready && !start.inside_thinking;
    if pin_now || (crosses && pin_ready) {
        return Err(Fallback::ToolPin);
    }
    let mode = mode_for(a, ctx, vocab)?;

    let history = penalty_history_scope(&a.output_tokens, ctx.tool_call_end_token).to_vec();
    let mut regimes = Vec::with_capacity(2);
    let mut rep_penalty = 1.0f32;
    let mut flag_sets = vec![start];
    if crosses {
        flag_sets.push(Flags {
            inside_thinking: false,
            think_ended: true,
            think_just_ended: true,
        });
    }
    for (r, flags) in flag_sets.iter().enumerate() {
        let pen = with_flags(a, *flags, |a| {
            penalty_params_for(a, PositionKind::Verify, 0.0, None, Vec::new())
        });
        if (pen.lz_penalty > 0.0 && history.len() >= 4)
            || (pen.dry_multiplier > 0.0 && history.len() >= 3)
        {
            return Err(Fallback::LzDry);
        }
        if r == 0 {
            rep_penalty = pen.repetition_penalty;
        }
        let edits = edits_for(*flags, &pen, &history, a, ctx, vocab)?;
        if edits.len() > edit_cap {
            return Err(Fallback::Edits);
        }
        regimes.push(edits);
    }
    let out_len = a.output_tokens.len();
    let eos_until = a.min_tokens.saturating_sub(out_len).min(u32::MAX as usize) as u32;
    Ok(Plan {
        mode,
        regimes,
        eos_until,
        rep_penalty,
        rng_key: a.seed.unwrap_or_else(fresh_key),
        pos_base: out_len as u64,
    })
}

/// Greedy unless the host would sample (`mtp_verify_sample` and temperature > 0), with the sampler
/// shape `verify_pick_with_pipeline` builds; refuse the parameters the kernel does not carry.
fn mode_for(a: &ActiveSeq, ctx: &LogitsContext, vocab: usize) -> Result<AcceptMode, Fallback> {
    // `verify_pick_with_pipeline` step 4a; `force_temp_zero` was refused by `build`.
    let samples = ctx.sampling.mtp_verify_sample && a.temperature > 0.0;
    if !samples {
        return Ok(AcceptMode::Greedy);
    }
    let p = SampleParams {
        temperature: a.temperature,
        top_k: a.top_k,
        top_p: a.top_p,
        min_p: effective_min_p(a.min_p, &ctx.sampling),
    };
    let top_k_over_cap = p.top_k as usize > CAND_CAP && (p.top_k as usize) < vocab;
    let sane = p.temperature.is_finite()
        && p.temperature >= 1e-3
        && p.top_p.is_finite()
        && p.min_p.is_finite();
    if a.top_n_sigma > 0.0 || top_k_over_cap || !sane {
        return Err(Fallback::SamplerParams);
    }
    Ok(AcceptMode::Sample(p))
}

/// Run `f` with the thinking flags set to `flags`, then restore them.
fn with_flags<R>(a: &mut ActiveSeq, flags: Flags, f: impl FnOnce(&ActiveSeq) -> R) -> R {
    let saved = (a.inside_thinking, a.think_ended, a.think_just_ended);
    (a.inside_thinking, a.think_ended, a.think_just_ended) = (
        flags.inside_thinking,
        flags.think_ended,
        flags.think_just_ended,
    );
    let r = f(a);
    (a.inside_thinking, a.think_ended, a.think_just_ended) = saved;
    r
}

/// One regime's edits, ascending by token id; ids at or above `vocab` are dropped (the host stages
/// skip them too).
fn edits_for(
    flags: Flags,
    pen: &SamplingParams,
    history: &[u32],
    a: &ActiveSeq,
    ctx: &LogitsContext,
    vocab: usize,
) -> Result<Vec<EditDesc>, Fallback> {
    let mut map: BTreeMap<u32, EditDesc> = BTreeMap::new();

    // 1. MinTokensEosMask: masked at positions below `eos_until` (output_len + pos < min_tokens).
    if a.min_tokens > 0 && a.output_tokens.len() < a.min_tokens {
        for &eos in &a.eos_tokens {
            edit(&mut map, vocab, eos, |e| e.eos = true);
        }
    }
    // 2. F2: refused by `build` whenever it would act.
    // 3. MidWordThinkEndMask.
    if !ctx.sampling.disable_watchdogs
        && flags.inside_thinking
        && let Some(end_tok) = ctx.think_end_token
        && let Some(prev) = a.output_tokens.last().copied()
        && let Some(mask) = ctx.mid_word_mask.as_deref()
        && mask.get(prev as usize).copied().unwrap_or(false)
    {
        edit(&mut map, vocab, end_tok, |e| e.mask = true);
    }
    // 4. PostCloseThinkMask.
    if flags.think_ended {
        if let Some(end_tok) = ctx.think_end_token {
            edit(&mut map, vocab, end_tok, |e| e.mask = true);
        }
        if let Some(start_tok) = a.think_start_token {
            edit(&mut map, vocab, start_tok, |e| e.mask = true);
        }
    }
    // 5. ToolCallDuringThinkingMask.
    if flags.inside_thinking {
        if let Some(tc) = ctx.tool_call_start_token {
            edit(&mut map, vocab, tc, |e| e.mask = true);
        }
    } else if a.suppress_tool_call
        && let Some(tc) = ctx.tool_call_start_token
    {
        edit(&mut map, vocab, tc, |e| e.sub12 = true);
    }
    // 6-9. ForcedThinkEndInjector, PinToToolCallStart, ForcedTokenFastPath, GrammarBitmaskApply:
    // refused by `build` whenever they would act.

    // Penalties, as `apply_penalties_and_bias` applies them (same window for both).
    let window = pen.repetition_penalty_window as usize;
    let effective: &[u32] = if window > 0 && window < history.len() {
        &history[history.len() - window..]
    } else {
        history
    };
    let rep = pen.repetition_penalty;
    if rep != 1.0 && rep > 0.0 && !history.is_empty() {
        for &tid in effective {
            edit(&mut map, vocab, tid, |e| e.rep_count += 1);
        }
    }
    let (freq, pres) = (pen.frequency_penalty, pen.presence_penalty);
    if (freq != 0.0 || pres != 0.0) && !history.is_empty() {
        let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
        for &tid in effective {
            *counts.entry(tid).or_insert(0) += 1;
        }
        for (tid, count) in counts {
            let delta = freq * count as f32 + pres;
            edit(&mut map, vocab, tid, |e| e.delta = Some(delta));
        }
    }
    let mut overflow = false;
    for &(tid, bias) in &pen.logit_bias {
        edit(&mut map, vocab, tid, |e| {
            if (e.n_bias as usize) < MAX_BIAS {
                e.bias[e.n_bias as usize] = bias;
                e.n_bias += 1;
            } else {
                overflow = true;
            }
        });
    }
    if overflow {
        return Err(Fallback::Edits);
    }
    Ok(map.into_values().collect())
}

/// Apply `f` to `tok`'s edit, creating it; ids at or above `vocab` are skipped (no host stage touches
/// them either).
fn edit(map: &mut BTreeMap<u32, EditDesc>, vocab: usize, tok: u32, f: impl FnOnce(&mut EditDesc)) {
    if (tok as usize) < vocab {
        f(map.entry(tok).or_insert_with(|| EditDesc {
            token: tok,
            ..Default::default()
        }));
    }
}

/// A fresh RNG key for a request without a seed: a per-call `RandomState` hash of a process nonce.
fn fresh_key() -> u64 {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(NONCE.fetch_add(1, Ordering::Relaxed));
    h.finish()
}
