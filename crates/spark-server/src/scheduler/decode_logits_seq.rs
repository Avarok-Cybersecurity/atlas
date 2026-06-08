// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits per-sequence helper (extracted to keep parent file ≤500 LoC).

use super::*;

// B1 (2026-05-26) margin-ratio drift detector moved to
// `logit_processors::b1_margin` (STEP 5) so the single unified
// `process_position_logits` fn owns it. It is gated to the FINAL decode
// position there.

/// Process logits for a single active sequence: dequant, adjust, sample, return token + optional logprobs.
#[allow(clippy::too_many_arguments)]
/// ATLAS_FORCE_TEMP_ZERO=1 — diagnostic mode that bypasses all drift
/// mitigation (AM1/A4/B1/C4) and just returns argmax of raw
/// logits. Used together with VLLM_FORCE_TEMP_ZERO on vLLM for
/// apples-to-apples layer-cosine comparison.
pub(crate) fn force_temp_zero_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_FORCE_TEMP_ZERO")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

pub fn process_seq_logits(
    _model: &dyn Model,
    a: &mut ActiveSeq,
    buf: &[u8],
    i: usize,
    vocab_size: usize,
    elem_bytes: usize,
    logits_fp32: bool,
    ctx: &crate::scheduler::logit_processors::LogitsContext,
    adaptive_sampling: bool,
) -> (u32, Option<crate::api::TokenLogprobs>) {
    // All special tokens are now carried in `ctx` (SSOT with the MTP verify
    // path) and read inside the pipeline stages / the A4 floor in
    // `penalty_params_for`.
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

    // ── Adaptive sampling: update zone, observe entropy, check greedy gate ──
    // Disabled by default (--adaptive-sampling flag). Each call scans the
    // full vocab (262k) on CPU: entropy O(V) exp+log, greedy gate O(V) exp.
    // Cost: ~300-400µs per token → 2-3x throughput regression when enabled.
    //
    // NOTE (STEP 4 ordering): with the unified `process_position_logits`
    // the pipeline masking + penalties run together below, so the entropy
    // observation now reads the dequantised logits BEFORE the pipeline mask
    // (it previously read the masked-but-unpenalised buffer). This only
    // affects the OPT-IN `--adaptive-sampling` path: when the flag is off
    // (the default, including qwen3.6-35b-a3b) `greedy_gate` is always
    // `false` and `effective_temp == a.temperature` regardless of logit
    // values, so the default decode is byte-identical.
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

    // Force temp=0 for greedy_gate path (adaptive override) so the sampler
    // takes the post-penalty argmax branch instead of the full stochastic
    // pipeline.
    let sampling_temp = if greedy_gate { 0.0 } else { effective_temp };
    // Advance seed per token for deterministic but varying randomness.
    let step_seed = a.seed.map(|s| s.wrapping_add(a.output_tokens.len() as u64));

    // ── SSOT post-processing (STEP 4 unification) ──
    // Build this position's penalty/bias params via the single
    // `penalty_params_for` builder (shared with the MTP verify/bootstrap
    // paths). `FinalDecode` carries the effective temperature, per-token
    // seed and base `logit_bias` (`a.logit_bias`); the penalty gates
    // (rep/presence/freq/LZ/DRY, in-tool DRY zero, grammar LZ zero) and the
    // A4 floor are computed inside the builder.
    let params = crate::scheduler::sample_step::penalty_params_for(
        a,
        crate::scheduler::sample_step::PositionKind::FinalDecode,
        sampling_temp,
        step_seed,
        a.logit_bias.clone(),
    );

    // Run the unified per-position pipeline: ATLAS_FORCE_TEMP_ZERO bypass →
    // pre-sample pipeline (8 masking stages + AdaDec `"decode"` diagnostic,
    // forced-token short-circuit) → B1 margin observer (FinalDecode only) →
    // penalties + bias applied in place on `f32_logits`. A `Some(tok)`
    // return is the force-temp-zero argmax OR the forced-token fast-path:
    // emit it directly with no sampling. The forced-token fast-path is
    // gated on `top_logprobs.is_none()` so its logprobs are always `None`;
    // for force-temp-zero `f32_logits` is the raw dequant buffer — so the
    // uniform `top_logprobs.map(..)` below reproduces the previous
    // per-branch logprob behaviour byte-for-byte.
    //
    // R1: this call NEVER advances/rolls back the grammar matcher — the
    // sampled token is fed to `gs.accept_token` later in
    // `decode_logits_step::process_decode_logits`.
    if let Some(tok) = crate::scheduler::logit_processors::process_position_logits(
        &mut f32_logits,
        a,
        ctx,
        &params,
        crate::scheduler::sample_step::PositionKind::FinalDecode,
    ) {
        let logprobs = a
            .top_logprobs
            .map(|k| extract_logprobs_from_f32(&f32_logits, tok, k as usize));
        return (tok, logprobs);
    }

    // F72 (byte-level partial-trigger anchor) was removed — see
    // F73 / fix42. The sampler-side anchor hung the server in
    // production despite passing isolated unit tests; the
    // model's broken-envelope case is now recovered at the
    // streaming-sanitizer + parser layer (F73 + F71). The
    // xgrammar non-anchored TagDispatch limitation is pinned
    // by `grammar.rs::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`
    // for documentation only.

    // `process_position_logits` has already applied the penalties + bias in
    // place. Sample from the now-masked-and-penalised `f32_logits` with the
    // configured temperature / top-k / top-p / min-p / seed, passing NEUTRAL
    // penalty params (rep=1.0, empty bias, empty history) so the sampler's
    // internal `apply_penalties_and_bias` is a no-op — the penalties are not
    // double-applied. This is the same split the MTP bootstrap path uses in
    // `sample_token_with_grammar`, and is byte-identical to the previous
    // single-shot `sample_with_params_history(.., params, history)` because
    // the buffer feeding temperature/top-k/p is identical either way.
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    let sampler_shape = SamplingParams {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        top_n_sigma: params.top_n_sigma,
        min_p: params.min_p,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: params.dry_base,
        dry_allowed_length: params.dry_allowed_length,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: params.seed,
    };
    let sampled = sample_with_params_history(f32_bytes, &sampler_shape, &[]);

    // Complete per-step logit dump (#222): ATLAS_LOGIT_DUMP=<file>. Captures
    // top-K + every applied bias + sampled, for Atlas↔vLLM divergence
    // analysis. Inert unless the env var is set. NOTE: with the unified
    // pipeline `f32_logits` is now masked AND penalised here (the penalties
    // were folded in by `process_position_logits`); the bias field reports
    // `params.logit_bias` (base + A4). This only changes the env-gated dump
    // output, never the emitted token.
    if super::logit_dump::enabled() {
        super::logit_dump::record(
            a.output_tokens.len(),
            a.inside_parameter_body,
            a.param_body_chars_emitted as usize,
            &f32_logits,
            &params.logit_bias,
            sampled,
        );
    }

    // Extract top-K logprobs from f32_logits if requested.
    let logprobs = a
        .top_logprobs
        .map(|k| extract_logprobs_from_f32(&f32_logits, sampled, k as usize));
    (sampled, logprobs)
}
