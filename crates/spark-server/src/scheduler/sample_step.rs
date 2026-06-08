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

/// Sample one token from device logits, applying temperature/top-k/top-p if non-greedy.
///
/// `suppress_ids`: token IDs to mask to -inf before sampling (e.g. EOS on first token).
pub fn sample_token(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
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
    if crate::scheduler::verify_pipeline_helper::fast_greedy_grammar_enabled()
        && suppress_ids.is_empty()
        && (temperature == 0.0 || crate::scheduler::decode_logits_seq::force_temp_zero_enabled())
        && penalties.repetition_penalty == 1.0
        && penalties.presence_penalty == 0.0
        && penalties.frequency_penalty == 0.0
        && penalties.lz_penalty == 0.0
        && penalties.dry_multiplier == 0.0
    {
        let top1 = model.argmax_on_device(logits, 0)?;
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
        },
    ))
}
