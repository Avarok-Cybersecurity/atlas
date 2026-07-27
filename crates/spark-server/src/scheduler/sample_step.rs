// SPDX-License-Identifier: AGPL-3.0-only

//! Token sampling helpers (resample + sample + grammar-constrained sample).

use super::*;

/// Which decode position a [`penalty_params_for`] /
/// [`crate::scheduler::logit_processors::process_position_logits`] call is
/// building for. The single discriminant that distinguishes the non-MTP
/// final-decode site (`decode_logits_seq::process_seq_logits`) from the MTP
/// verify / bootstrap sites — replacing the two divergent inline
/// `SamplingParams { .. }` literals (and the two divergent stage blocks)
/// with one SSOT.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PositionKind {
    FinalDecode,
    Verify,
}

impl PositionKind {
    /// AdaDec diagnostic path label — only tags the env-gated
    /// `ATLAS_ADADEC_DIAGNOSTIC` JSONL record; never alters a transform.
    pub(super) fn adadec_label(self) -> &'static str {
        match self {
            PositionKind::FinalDecode => "decode",
            PositionKind::Verify => "verify",
        }
    }
}

/// #192: penalty history for the CURRENT tool-call segment — the token
/// history slice that repetition / presence / frequency / LZ / DRY penalties
/// see, cut to AFTER the last completed `</tool_call>`.
///
/// With parallel tool calls the full-generation history compounds the
/// repetition penalty on the SECOND call's structural scaffold: the divide is
/// per OCCURRENCE (`logit /= rep_penalty` once per history hit, so 1.1^k), and
/// by call 2 the exact scaffold tokens from call 1 (`\n`=198, `</`=510,
/// `>`=29, `<tool_call>`) are crushed and lose to space-prefixed BPE variants
/// (` </`=672, ` >`=835) — breaking the close literal and garbling the call
/// (live 2026-07-02, hermes on Qwen3.6-27B: call 2 = `Berlin </ parameter >`
/// drift, and the penalized inter-call `<tool_call>`/`\n` suppressed the
/// third call outright). Scoping the history to the current segment gives
/// every call the same penalty landscape the FIRST call had — the one the
/// presets were tuned on — while keeping the intra-call anti-attractor role
/// (A1) fully intact. Whole-block call repetition is still bounded by the
/// post-completion open cap and the loop watchdogs.
///
/// No completed call in the history (single-call turns, plain chat,
/// `tool_call_end_token=None`) => the full history, byte-identical to the
/// previous behavior.
pub(super) fn penalty_history_scope(
    output_tokens: &[u32],
    tool_call_end_token: Option<u32>,
) -> &[u32] {
    match tool_call_end_token.and_then(|t| output_tokens.iter().rposition(|&x| x == t)) {
        Some(p) => &output_tokens[p + 1..],
        None => output_tokens,
    }
}

/// #192: drop the standing `<tool_call>` OPENER nudge while INSIDE a tool
/// body. `sampling_setup` arms a +3.0 exponential-decay bias on the opener
/// for tools-active requests to encourage a call to START; inside a body it
/// is pure distortion — observed live (2026-07-02, hermes): at a borderline
/// mid-value position it flipped the argmax to a spurious re-open
/// (`... </ parameter > \n<tool_call> ...`), garbling the second parallel
/// call. Negative (anti-repeat, -5/-10) opener values still apply everywhere.
/// Pure over the bias vec so it is unit-testable without an `ActiveSeq`.
pub(super) fn strip_in_tool_opener_bias(
    logit_bias: &mut Vec<(u32, f32)>,
    in_tool: bool,
    opener: Option<u32>,
) {
    if !in_tool {
        return;
    }
    let Some(tc_open) = opener else {
        return;
    };
    logit_bias.retain(|&(id, delta)| id != tc_open || delta <= 0.0);
}

/// P1-4 (2026-07-09): kill-switch for threading the sequence's resolved
/// `min_p` (request value, already floored by MODEL.toml `min_p_floor` in
/// `sampling_setup`) into the MTP bootstrap / first-token / verify sample
/// sites that previously passed hardcoded `0.0` literals — bypassing the
/// exact FP8/NVFP4 argmax-flip safety net the floor exists for. Default ON
/// (SSOT wiring fix); set `ATLAS_NO_MTP_MINP=1` to restore the 0.0 literals.
/// One-time read, mirroring `hint_injector::bash_wander_enabled`.
pub(super) fn mtp_minp_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_NO_MTP_MINP").ok().as_deref() != Some("1"))
}

/// P1-4 (2026-07-09): the min_p actually handed to the sampler at the
/// previously-hardcoded sites — the caller's resolved value, or 0.0 under
/// the `ATLAS_NO_MTP_MINP=1` kill-switch.
pub(super) fn effective_min_p(min_p: f32) -> f32 {
    if mtp_minp_enabled() { min_p } else { 0.0 }
}

/// Build the penalty/bias-carrying [`SamplingParams`] for one sequence —
/// the SINGLE source of truth for the repetition / presence / frequency /
/// LZ / DRY penalty gates + the A4 floor shared by the non-MTP decode path
/// and the MTP bootstrap + verify paths (the root-cause fix for
/// repetition_penalty / dry never reaching MTP-emitted tokens).
///
/// SSOT: the in-tool DRY gate (`dry_multiplier` zeroed inside a tool body)
/// and the grammar LZ gate (`lz_penalty` zeroed when a grammar is active)
/// are computed once here and match the pre-unification `process_seq_logits`
/// literal exactly.
///
/// Position-specific inputs:
///  * `FinalDecode` → the caller passes the effective `temperature`, the
///    per-token `seed` and the base `logit_bias` (`ActiveSeq.logit_bias`,
///    cloned) it computed for this step.
///  * `Verify` → the MTP verify/bootstrap emission is a penalty-aware
///    greedy ARGMAX, so callers pass `temperature = 0.0`, `seed = None`,
///    empty base bias.
pub(super) fn penalty_params_for(
    a: &ActiveSeq,
    kind: PositionKind,
    temperature: f32,
    seed: Option<u64>,
    base_logit_bias: Vec<(u32, f32)>,
) -> SamplingParams {
    // `Verify` positions are a penalty-aware greedy ARGMAX, so the contract
    // is temperature 0.0, no seed, no caller-supplied base bias. Pin it so a
    // future caller can't silently pass stochastic params on the speculative
    // path. The A4 floor below is appended for BOTH kinds (intended delta).
    debug_assert!(
        kind != PositionKind::Verify
            || (temperature == 0.0 && seed.is_none() && base_logit_bias.is_empty()),
        "Verify positions must pass temperature=0.0, seed=None, empty base bias"
    );
    let in_tool = a.inside_tool_body && !a.inside_thinking;
    let mut logit_bias = base_logit_bias;

    // #192: the `<tool_call>` opener nudge must not act INSIDE a tool body
    // (spurious mid-value re-open — see `strip_in_tool_opener_bias`).
    strip_in_tool_opener_bias(&mut logit_bias, in_tool, a.tool_call_start_token);

    // A4 (2026-05-26) POST_THINK_MIN_REASONING floor — moved here from the
    // inline `process_seq_logits` block (STEP 3). Suppress the `</think>`
    // token until at least MIN_REASONING_TOKENS thinking tokens have been
    // emitted, closing the reasoning-collapse cascade documented in
    // research2_probe_forensics.md (reasoning_content length decays 233→0
    // chars over 14 assistant turns). When the model emits a vanishingly
    // short `<think>` block, the downstream tool emission lacks planning
    // context and drifts to phantom paths / leaked control characters.
    //
    // Bias is -8.0 (firm but not infinite). If `reasoning_budget` is set
    // very small (<16) the request opted out of meaningful thinking and the
    // floor doesn't apply.
    //
    // R3: A4 is appended as a `logit_bias` ENTRY (NOT a pre-penalty direct
    // mask) so the `apply_penalties_and_bias` ordering stays byte-identical
    // on the non-MTP path. INTENDED DELTA: because the builder is now the
    // SSOT for BOTH paths, A4 is ALSO active on the MTP verify path (where
    // it was previously dead — the verify path never ran the inline floor).
    const A4_MIN_REASONING_TOKENS: u32 = 16;
    if a.inside_thinking
        && a.thinking_tokens < A4_MIN_REASONING_TOKENS
        && a.thinking_budget.unwrap_or(A4_MIN_REASONING_TOKENS) >= A4_MIN_REASONING_TOKENS
        && let Some(end_tok) = a.think_end_token
    {
        logit_bias.push((end_tok, -8.0f32));
    }

    SamplingParams {
        temperature,
        top_k: a.top_k,
        top_p: a.top_p,
        top_n_sigma: a.top_n_sigma,
        min_p: a.min_p,
        logit_bias,
        // A1: full penalty INSIDE tool body too (stops attractor patterns:
        // mismatched-paren runaway, `lean://` prefix loop, same-tool-call
        // repetition). Matches `process_seq_logits`.
        repetition_penalty: a.repetition_penalty,
        repetition_penalty_window: a.repetition_penalty_window,
        presence_penalty: a.presence_penalty,
        frequency_penalty: a.frequency_penalty,
        lz_penalty: if a.grammar_state.is_some() {
            0.0
        } else {
            a.lz_penalty
        },
        // DRY stays disabled inside the tool body (its short n-gram window
        // fights legitimate JSON structural repetition `","`/`":"`).
        dry_multiplier: if in_tool { 0.0 } else { a.dry_multiplier },
        dry_base: a.dry_base,
        dry_allowed_length: a.dry_allowed_length,
        dry_sequence_breakers: a.dry_sequence_breakers.clone(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed,
    }
}

/// Re-sample verify tokens from the logits buffer when temperature > 0.
///
/// After `decode_verify_graphed`, the logits buffer still contains valid
/// BF16 logits for each verified position (`[k, vocab_size]`). The CUDA
/// graph bakes in argmax, but when the request has temperature > 0 we need
/// stochastic sampling. This copies the logits to host and samples per
/// position, returning the temperature-sampled tokens.
///
/// Falls back to `argmax_tokens` if the D2H copy fails.
///
/// P1-3 (2026-07-09): SUPERSEDED (still dead code). This standalone helper
/// sampled the RAW verify logits, so a resurrected call here would bypass
/// the grammar bitmask + pipeline masks. Its sampling logic now lives
/// INSIDE the pipeline path (`verify_pipeline_helper::verify_pick_with_pipeline`,
/// step 4a) where it runs over the already-masked-and-penalised logits —
/// the pick is grammar-legal by construction. Kept for reference only.
#[allow(dead_code)]
pub fn verify_resample(model: &dyn Model, argmax_tokens: &[u32], temperature: f32) -> Vec<u32> {
    if temperature == 0.0 {
        return argmax_tokens.to_vec();
    }
    let k = argmax_tokens.len();
    let vocab = model.vocab_size();
    let total_bytes = k * vocab * 2;
    let mut buf = vec![0u8; total_bytes];
    if model
        .copy_logits_to_host(model.logits_buffer_ptr(), &mut buf)
        .is_err()
    {
        return argmax_tokens.to_vec();
    }
    let params = SamplingParams {
        temperature,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty_window: 0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: DEFAULT_DRY_MULTIPLIER,
        dry_base: DEFAULT_DRY_BASE,
        dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    (0..k)
        .map(|i| {
            let slice = &buf[i * vocab * 2..(i + 1) * vocab * 2];
            sample_with_params(slice, &params)
        })
        .collect()
}

/// Sample one token from device logits, applying temperature/top-k/top-p/min-p if non-greedy.
///
/// `suppress_ids`: token IDs to mask to -inf before sampling (e.g. EOS on first token).
///
/// P1-4 (2026-07-09): `min_p` is the sequence's RESOLVED min_p (request
/// value with the MODEL.toml `min_p_floor` already applied by
/// `sampling_setup`). This site previously hardcoded `min_p: 0.0` in its
/// internal `SamplingParams`, so the only stochastic first-token sample
/// under MTP bypassed the FP8 argmax-flip safety net the floor documents
/// (min_p_floor = 0.05 on this model family). Kill-switch:
/// `ATLAS_NO_MTP_MINP=1` restores the old 0.0 literal via [`effective_min_p`].
pub fn sample_token(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    suppress_ids: &[u32],
) -> Result<u32> {
    if temperature == 0.0 && suppress_ids.is_empty() {
        return model.argmax_on_device(logits, 0);
    }
    let vocab_size = model.vocab_size();
    // Read logits from device. Gemma-4 dense single-token decode produces FP32
    // logits via the FP32 lm_head + softcap path (margin between top-1 and
    // top-2 sits on a BF16 representable boundary at value 16-32, so storing
    // BF16 there flips the greedy argmax). Other paths still produce BF16
    // and need expansion. Dispatch by `logits_ptr_is_fp32`.
    let mut f32_logits: Vec<f32> = if model.logits_ptr_is_fp32(logits) {
        let mut buf = vec![0u8; vocab_size * 4];
        model.copy_logits_to_host(logits, &mut buf)?;
        // SAFETY: buf has length vocab_size * 4 and the device kernel wrote
        // little-endian f32 values; reinterpret is byte-equivalent on x86/arm.
        let f32_slice: &[f32] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, vocab_size) };
        f32_slice.to_vec()
    } else {
        let mut bf16_buf = vec![0u8; vocab_size * 2];
        model.copy_logits_to_host(logits, &mut bf16_buf)?;
        (0..vocab_size)
            .map(|i| {
                let lo = bf16_buf[i * 2];
                let hi = bf16_buf[i * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };
    // Suppress EOS tokens on first token by setting to -inf.
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    if temperature == 0.0 {
        // Greedy argmax over FP32
        let best = f32_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(best);
    }
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    Ok(sample_with_params(
        f32_bytes,
        &SamplingParams {
            temperature,
            top_k,
            top_p,
            top_n_sigma: 0.0,
            // P1-4 (2026-07-09): caller's resolved min_p, not a 0.0 literal.
            min_p: effective_min_p(min_p),
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty_window: 0,
            lz_penalty: DEFAULT_LZ_PENALTY,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        },
    ))
}

/// Sample one token from device logits with optional grammar constraint.
///
/// Like `sample_token` but also applies grammar bitmask when `grammar_state`
/// is provided. Always uses host-side sampling when grammar is active (can't
/// use GPU argmax since grammar bitmask is CPU-side).
///
/// `penalties` + `history` carry the sequence's configured repetition /
/// presence / frequency / LZ / DRY penalties (built via [`penalty_params_for`])
/// and the output-token history. These are applied via the shared
/// [`apply_penalties_and_bias`] helper AFTER the grammar bitmask + EOS
/// suppression and BEFORE the temperature decision — the same order the
/// non-MTP `process_seq_logits` path uses — so MTP-bootstrap-emitted tokens
/// see the same penalties as the non-MTP path. Backward-compatible: a
/// no-op when the penalties are neutral (rep==1.0, dry==0.0, etc.).
pub fn sample_token_with_grammar(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    suppress_ids: &[u32],
    mut grammar_state: Option<&mut GrammarState>,
    penalties: &SamplingParams,
    history: &[u32],
) -> Result<u32> {
    // ── FAST PATH (#3, 2026-06-02): on-GPU greedy pick under grammar ──
    // The MTP bootstrap sample (~1 token/step) otherwise D2Hs + dequants the
    // full 248k vocab + applies the bitmask on host. When greedy (temp=0 or
    // ATLAS_FORCE_TEMP_ZERO), penalties neutral, and no suppress list, the
    // masked-greedy pick == the GPU argmax whenever that argmax is grammar-
    // allowed (global max ∩ allowed-set = the max). Emit it directly; fall back
    // to the host path below only when the argmax is grammar-disallowed.
    // Mirrors the verify-path fast path. Kill-switch ATLAS_DISABLE_FAST_GREEDY=1.
    //
    // #237 (fix 4a): penalty-neutrality relaxed to the SSOT `fast_greedy`
    // gate shared with the verify helper — reduce-only penalties cannot flip
    // an argmax that is not in the scoped `history` and has a positive raw
    // logit (proof in `fast_greedy` module docs). `history` here is already
    // the scoped span the slow path feeds to `apply_penalties_and_bias`.
    if crate::scheduler::verify_pipeline_helper::fast_greedy_grammar_enabled()
        && suppress_ids.is_empty()
        && (temperature == 0.0 || crate::scheduler::decode_logits_seq::force_temp_zero_enabled())
    {
        let gate = crate::scheduler::fast_greedy::classify_penalties(penalties);
        if gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
            let top1 = model.argmax_on_device(logits, 0)?;
            let immune = gate == crate::scheduler::fast_greedy::PenaltyGate::Neutral
                || crate::scheduler::fast_greedy::argmax_immune(top1, history, || {
                    crate::scheduler::fast_greedy::logit_is_positive(
                        model,
                        logits,
                        0,
                        model.vocab_size(),
                        top1,
                    )
                });
            if immune {
                let allowed = match grammar_state.as_mut() {
                    Some(gs) => {
                        if gs.is_terminated() {
                            true
                        } else {
                            gs.fill_bitmask();
                            gs.is_token_allowed(top1)
                        }
                    }
                    None => true,
                };
                if allowed {
                    return Ok(top1);
                }
            }
        }
    }

    let vocab_size = model.vocab_size();
    let mut bf16_buf = vec![0u8; vocab_size * 2];
    model.copy_logits_to_host(logits, &mut bf16_buf)?;
    let mut f32_logits: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let lo = bf16_buf[i * 2];
            let hi = bf16_buf[i * 2 + 1];
            bf16_to_f32(lo, hi)
        })
        .collect();
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    // Apply grammar bitmask (when a grammar is active).
    if let Some(gs) = grammar_state {
        gs.fill_bitmask();
        gs.apply_bitmask_to_logits(&mut f32_logits);
    }
    // SSOT penalties + bias on the post-mask logits, using the seq's
    // output-token history — identical stage to the non-MTP path.
    apply_penalties_and_bias(&mut f32_logits, penalties, history);
    if temperature == 0.0 {
        let best = f32_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(best);
    }
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    // Penalties already applied in place above; pass neutral penalty params
    // to `sample_with_params` (which re-runs the helper with empty history,
    // a guaranteed no-op) so the stochastic top-k/top-p/min-p pipeline runs.
    Ok(sample_with_params(
        f32_bytes,
        &SamplingParams {
            temperature,
            top_k,
            top_p,
            top_n_sigma: 0.0,
            // P1-4 (2026-07-09): thread the sequence's resolved min_p from
            // the SSOT `penalties` struct (built by `penalty_params_for`,
            // which copies `a.min_p` — request value + MODEL.toml
            // `min_p_floor` applied in `sampling_setup`). This was a
            // hardcoded 0.0, so the MTP BOOTSTRAP token — one of only two
            // stochastic sample points under MTP — bypassed the FP8
            // argmax-flip safety net the floor documents. Kill-switch:
            // ATLAS_NO_MTP_MINP=1 restores the 0.0 literal.
            min_p: effective_min_p(penalties.min_p),
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty_window: 0,
            lz_penalty: DEFAULT_LZ_PENALTY,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        },
    ))
}

/// Sample the FIRST generated token after prefill, applying the grammar
/// constraint when one is active.
///
/// `#131`: the very first decode token is produced from the prefill's final
/// logits in [`crate::scheduler::prefill_b_step`], OUTSIDE the per-step
/// `process_seq_logits` / `emit_step` decode loop. The plain [`sample_token`]
/// used there does NOT mask against the grammar, so the model's first token is
/// free — under a `json_schema` (or any) grammar it emits a leading prose token
/// (e.g. `Here` / the schema `name`) BEFORE the grammar's opening `{`, which
/// breaks strict parsing and bleeds prose into the first string value. The
/// per-step `accept_token` in `emit_step` also only covers tokens `2..N`, so
/// the matcher would still sit at its start state when token 2 decodes.
///
/// This helper closes both gaps for the first token: it masks the logits with
/// the grammar bitmask (forcing a grammar-legal first token — leading
/// whitespace or `{`) and then advances the matcher with the accepted token,
/// exactly mirroring the mask-then-`accept_token` order the decode loop uses
/// for every subsequent token. With no grammar it is byte-identical to
/// [`sample_token`] (PCND: no behavior change on the non-grammar path).
///
/// Penalties are neutral here (empty history on the first token makes every
/// penalty a no-op), matching the existing non-grammar first-token contract.
///
/// P1-4 (2026-07-09): `min_p` is the sequence's RESOLVED min_p (request
/// value with the MODEL.toml `min_p_floor` already applied by
/// `sampling_setup`), threaded from the prefill call sites that previously
/// let a hardcoded 0.0 reach the sampler. The first token is the other of
/// the only two stochastic sample points under MTP (with the bootstrap),
/// so the unfloored min_p let the FP8/NVFP4 degenerate logit tail be
/// sampled exactly where the floor was designed to block it. Kill-switch:
/// `ATLAS_NO_MTP_MINP=1` restores the 0.0 literals via [`effective_min_p`].
pub fn sample_first_token(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    suppress_ids: &[u32],
    grammar_state: Option<&mut GrammarState>,
) -> Result<u32> {
    let Some(gs) = grammar_state else {
        return sample_token(
            model,
            logits,
            temperature,
            top_k,
            top_p,
            min_p,
            suppress_ids,
        );
    };
    let neutral = SamplingParams {
        temperature,
        top_k,
        top_p,
        top_n_sigma: 0.0,
        // P1-4 (2026-07-09): resolved min_p, consumed via `penalties.min_p`
        // inside `sample_token_with_grammar` (kill-switch applied there).
        min_p,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty_window: 0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: DEFAULT_DRY_BASE,
        dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    let tok = sample_token_with_grammar(
        model,
        logits,
        temperature,
        top_k,
        top_p,
        suppress_ids,
        Some(gs),
        &neutral,
        &[],
    )?;
    // Advance the matcher past the first token (the emit_step accept_token
    // only runs for tokens 2..N). A grammar-disallowed first token here would
    // indicate the mask was not applied — keep going rather than abort; the
    // emit_step disengage path handles any later desync gracefully.
    gs.accept_token(tok);
    Ok(tok)
}

#[cfg(test)]
mod penalty_scope_tests {
    //! #192: the per-tool-call-segment penalty history scope and the in-tool
    //! opener-bias strip — the two levers that stop Atlas's own sampling
    //! machinery from garbling the SECOND parallel tool call (live 2026-07-02,
    //! hermes on Qwen3.6-27B-NVFP4: call 2 scaffold flipped to space-prefixed
    //! BPE variants `Berlin </ parameter >` + a spurious mid-value
    //! `<tool_call>` re-open; the third call was penalty-suppressed outright).
    use super::{penalty_history_scope, strip_in_tool_opener_bias};

    const CLOSE: u32 = 248059; // </tool_call>

    #[test]
    fn scope_without_completed_call_is_full_history() {
        let toks = vec![1, 2, 3, 4];
        assert_eq!(penalty_history_scope(&toks, Some(CLOSE)), &toks[..]);
        // No configured end token (legacy/none) — also full history.
        assert_eq!(penalty_history_scope(&toks, None), &toks[..]);
    }

    #[test]
    fn scope_cuts_after_last_completed_call() {
        //             call 1                 sep  call 2 (open)
        let toks = vec![10, 11, 12, CLOSE, 198, 20, 21];
        assert_eq!(penalty_history_scope(&toks, Some(CLOSE)), &[198, 20, 21]);
        // Two completed calls — only the segment after the LAST close counts.
        let toks = vec![10, CLOSE, 198, 20, CLOSE, 30];
        assert_eq!(penalty_history_scope(&toks, Some(CLOSE)), &[30]);
        // Close is the final token — the next position starts a fresh segment.
        let toks = vec![10, 11, CLOSE];
        assert_eq!(penalty_history_scope(&toks, Some(CLOSE)), &[] as &[u32]);
    }

    /// The live failure mechanism, demonstrated at the sampler level: the
    /// repetition penalty divides PER OCCURRENCE, so a structural token used
    /// k times across previous calls is at 1/1.1^k by the next call — enough
    /// to lose to an unpenalized space-prefixed variant. The scoped history
    /// restores the first-call landscape.
    #[test]
    fn scoped_history_prevents_cross_call_penalty_compounding() {
        use crate::scheduler::{SamplingParams, apply_penalties_and_bias};
        const NL: u32 = 198; // '\n' — exact scaffold token
        const NL_VARIANT: u32 = 695; // ' \n' — space-prefixed BPE variant

        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            logit_bias: Vec::new(),
            // MODEL.toml [sampling.tools] default for Qwen3.6-27B.
            repetition_penalty: 1.1,
            repetition_penalty_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            lz_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        };

        // Call 1 used '\n' six times; the variant never appeared. The model
        // slightly prefers the exact token (as at T=0 on the real scaffold).
        let full_history: Vec<u32> = vec![NL; 6]
            .into_iter()
            .chain([10, 11, CLOSE, NL]) // close + separator newline
            .collect();

        let mut logits = vec![0.0f32; 1000];
        logits[NL as usize] = 10.0;
        logits[NL_VARIANT as usize] = 8.5;
        apply_penalties_and_bias(&mut logits, &params, &full_history);
        assert!(
            logits[NL_VARIANT as usize] > logits[NL as usize],
            "unscoped: 1.1^7 compounding must flip the scaffold token (the bug): {} vs {}",
            logits[NL as usize],
            logits[NL_VARIANT as usize],
        );

        let mut logits = vec![0.0f32; 1000];
        logits[NL as usize] = 10.0;
        logits[NL_VARIANT as usize] = 8.5;
        let scoped = penalty_history_scope(&full_history, Some(CLOSE));
        assert_eq!(scoped, &[NL], "segment = separator newline only");
        apply_penalties_and_bias(&mut logits, &params, scoped);
        assert!(
            logits[NL as usize] > logits[NL_VARIANT as usize],
            "scoped: one occurrence must NOT flip the scaffold token: {} vs {}",
            logits[NL as usize],
            logits[NL_VARIANT as usize],
        );
    }

    #[test]
    fn opener_bias_stripped_only_inside_tool_body() {
        const OPEN: u32 = 248058; // <tool_call>
        // Inside a body: the +3.0 nudge goes, negative anti-repeat stays,
        // unrelated entries stay.
        let mut bias = vec![(OPEN, 3.0f32), (42, -8.0f32)];
        strip_in_tool_opener_bias(&mut bias, true, Some(OPEN));
        assert_eq!(bias, vec![(42, -8.0f32)]);

        let mut bias = vec![(OPEN, -5.0f32)];
        strip_in_tool_opener_bias(&mut bias, true, Some(OPEN));
        assert_eq!(bias, vec![(OPEN, -5.0f32)], "anti-repeat bias survives");

        // Outside a body: untouched.
        let mut bias = vec![(OPEN, 3.0f32)];
        strip_in_tool_opener_bias(&mut bias, false, Some(OPEN));
        assert_eq!(bias, vec![(OPEN, 3.0f32)]);

        // No opener token configured: untouched.
        let mut bias = vec![(OPEN, 3.0f32)];
        strip_in_tool_opener_bias(&mut bias, true, None);
        assert_eq!(bias, vec![(OPEN, 3.0f32)]);
    }
}
